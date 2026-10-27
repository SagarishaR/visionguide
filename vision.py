"""OpenCV 5 tools + agent logic for a voice-guided reader (medicine, food expiry, bills, forms) for low-vision users."""
import re, datetime, calendar, cv2, numpy as np, pytesseract
from pytesseract import Output

TH = dict(blur=60.0, dark=60, glare=0.03, cover_min=0.30, cover_max=0.92, conf=60, item_conf=55)

MSG = {  # key: (English, Tamil)
 "dark":   ("It is too dark. Move to a brighter place or switch on a light.", "மிகவும் இருட்டாக உள்ளது. வெளிச்சமான இடத்திற்குச் செல்லுங்கள்."),
 "glare":  ("There is glare on the {side}. Tilt the phone a little.", "பளபளப்பு உள்ளது. போனை சிறிது சாய்க்கவும்."),
 "blur":   ("The picture is blurry. Hold the phone still.", "படம் மங்கலாக உள்ளது. போனை அசையாமல் பிடிக்கவும்."),
 "nodoc":  ("I cannot see the item. Point the camera at it.", "பொருள் தெரியவில்லை. கேமராவை அதன் மேல் காட்டவும்."),
 "closer": ("Move the phone closer.", "போனை அருகில் கொண்டு வாருங்கள்."),
 "farther":("Move the phone a little farther away.", "போனை சற்றுத் தள்ளி பிடிக்கவும்."),
 "ok":     ("Good. Hold still.", "நன்றாக உள்ளது. அசையாமல் இருங்கள்."),
 "turn":   ("I could not find a date. Turn the item over, and show me the bottom, the side or the back.",
            "தேதி கிடைக்கவில்லை. பொருளைத் திருப்பி, அடிப்பகுதி, பக்கம் அல்லது பின்புறத்தைக் காட்டுங்கள்."),
 "closer_date": ("I can see the expiry label but cannot read it. Bring the phone closer to the date and hold still.",
            "காலாவதி லேபிள் தெரிகிறது, ஆனால் படிக்க முடியவில்லை. போனை தேதிக்கு அருகில் கொண்டு வந்து அசையாமல் பிடிக்கவும்."),
 "unsure": ("I am not sure what it says. Please ask someone to check.", "எனக்கு உறுதியாகத் தெரியவில்லை. யாரிடமாவது சரிபார்க்கச் சொல்லுங்கள்."),
}

def find_doc(img):
    """Find a plausible large document/package outline, even with imperfect edges."""
    h, w = img.shape[:2]
    scale = min(1.0, 700.0 / w)
    small = cv2.resize(
        img, (max(1, int(w * scale)), max(1, int(h * scale)))
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    sh, sw = small.shape[:2]
    frame_area = sw * sh
    candidates = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if area < 0.05 * frame_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

        # Prefer a four-corner outline, but tolerate irregular packaging edges.
        if len(approx) == 4 and cv2.isContourConvex(approx):
            score = area * 1.2
            candidates.append((score, approx.reshape(4, 2)))
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            rect_area = cv2.contourArea(box.astype(np.float32))
            if rect_area >= 0.08 * frame_area:
                score = area * 0.7
                candidates.append((score, box))

    if not candidates:
        return None, 0.0

    _, points = max(candidates, key=lambda item: item[0])
    points = points.astype(np.float32) / scale

    # Return a quadrilateral only when it is reasonably inside the image.
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)

    fraction = cv2.contourArea(points) / float(w * h)
    return points, float(fraction)

def warp(img, q):
    """Perspective-correct the document so text is straight."""
    s = q.sum(1); d = np.diff(q, axis=1).ravel()
    r = np.array([q[s.argmin()], q[d.argmin()], q[s.argmax()], q[d.argmax()]], np.float32)
    W = int(max(np.linalg.norm(r[0] - r[1]), np.linalg.norm(r[3] - r[2])))
    H = int(max(np.linalg.norm(r[0] - r[3]), np.linalg.norm(r[1] - r[2])))
    M = cv2.getPerspectiveTransform(r, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(img, M, (max(W, 1), max(H, 1)))

def quality(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (640, max(1, int(640 * g.shape[0] / g.shape[1]))))
    mask = g > 245
    glare = float(mask.mean())
    side = "middle"
    if mask.any():
        cx = np.nonzero(mask)[1].mean() / 640
        side = "left" if cx < 0.4 else "right" if cx > 0.6 else "middle"
    _, cover = find_doc(img)
    return dict(blur=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1), brightness=round(float(g.mean()), 1),
                glare=round(glare, 3), glare_side=side, cover=round(cover, 2))

def guide(q):
    """Decision step 1: what we SEE decides what the person is told to do next."""
    if q["brightness"] < TH["dark"]: return "dark"
    if q["glare"] > TH["glare"]: return "glare"
    if q["cover"] == 0: return "nodoc"
    if q["blur"] < TH["blur"]: return "blur"
    if q["cover"] < TH["cover_min"]: return "closer"
    if q["cover"] > TH["cover_max"]: return "farther"
    return "ok"

def message(key, side="middle"):
    en, ta = MSG[key]
    return en.format(side=side), ta

# ---------- OCR ----------
def _ocr(gray, psm):
    d = pytesseract.image_to_data(gray, config=f"--psm {psm}", output_type=Output.DICT)
    return [dict(t=d["text"][i].strip(), c=float(d["conf"][i]), x=d["left"][i], y=d["top"][i], w=d["width"][i], h=d["height"][i],
                 line=(d["block_num"][i], d["par_num"][i], d["line_num"][i]))
            for i in range(len(d["text"])) if d["text"][i].strip() and float(d["conf"][i]) >= 0]

def _enhance(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.shape[1] < 1000:
        f = 1000 / g.shape[1]; g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)

def _fix_weak(g, words, trace):
    """Zoom into low-confidence words (3x + Otsu) and re-read them."""
    for w in [w for w in words if w["c"] < TH["conf"] and len(w["t"]) > 1][:8]:
        p = 6
        crop = g[max(0, w["y"] - p):w["y"] + w["h"] + p, max(0, w["x"] - p):w["x"] + w["w"] + p]
        if crop.size == 0: continue
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        r = _ocr(crop, 7)
        if r and min(x["c"] for x in r) > w["c"]:
            new, nc = " ".join(x["t"] for x in r), min(x["c"] for x in r)
            trace.append(f"Unsure of '{w['t']}' ({w['c']:.0f}%): zoomed 3x + re-read -> '{new}' ({nc:.0f}%)")
            w["t"], w["c"] = new, nc
        else:
            trace.append(f"Unsure of '{w['t']}' ({w['c']:.0f}%): zoomed re-read did not help")

def _lines(words):
    d = {}
    for w in words: d.setdefault(w["line"], []).append(w)
    return [(" ".join(x["t"] for x in v), sum(x["c"] for x in v) / len(v)) for v in d.values()]

# ---------- extraction (rules per document type) ----------
MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
YR = r"(?:20(?:1[5-9]|2\d|3\d)|(?:1[5-9]|2\d|3\d))"
DATE_RES = [re.compile(rf"\b(?:\d{{1,2}}[/.\-])?(?:0?[1-9]|1[0-2])[/.\-]{YR}\b", re.I),
            re.compile(rf"\b(?:\d{{1,2}}\s*)?{MON}[\s.,'/\-]*{YR}\b", re.I)]
EXP_K = ("xp", "exp", "use by", "best before", "before", "bbe", "valid")
MFG_K = ("fg", "mfg", "mfd", "manuf", "packed", "pkd")
AMT = re.compile(r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")

def _fixdigits(t):
    f = lambda m: m.group().translate(str.maketrans("OIlSB;)", "01158//")) if sum(c.isdigit() for c in m.group()) >= 3 else m.group()
    return re.sub(r"[0-9OIlSB/.\-;)]{5,12}", f, t)

BAD = re.compile(rf"\b[0-9A-Za-z]([0-2])[/.\-]{YR}\b")  # month's first digit garbled by OCR: 42/2020, A2/2020

def _dates(lines, due_mode=False):
    out = []
    for i, (txt, conf) in enumerate(lines):
        fixed = _fixdigits(txt); got = False
        ctx_of = lambda src, p: ((lines[i - 1][0][-14:] if i and p < 3 else "") + src[max(0, p - 14):p]).lower()
        for rx, src in ((DATE_RES[0], fixed), (DATE_RES[1], txt)):
            for m in rx.finditer(src):
                s_ = m.group(); ctx = ctx_of(src, m.start())
                if due_mode and "due" in ctx: kind = "due date"
                elif any(k in ctx for k in MFG_K): kind = "made"
                elif any(k in ctx for k in EXP_K): kind = "expiry"
                elif re.search(r"20\d\d", s_) or re.search(MON, s_, re.I): kind = "date"
                else: continue
                got = got or kind == "expiry"
                out.append(dict(kind=kind, value=s_.strip(), conf=round(conf)))
        if not got:  # repair "EXP.42/2020" -> 12/2020, but never claim high confidence
            for m in BAD.finditer(fixed):
                if any(k in ctx_of(fixed, m.start()) for k in EXP_K):
                    out.append(dict(kind="expiry", value="1" + m.group(1) + "/" + re.split(r"[/.\-]", m.group())[-1], conf=min(round(conf), 45)))
    return out

def extract(mode, lines):
    if mode in ("medicine", "food"):
        return [d for d in _dates(lines) if d["kind"] != "made"] or []
    if mode == "bill":
        items = _dates(lines, True)
        for txt, conf in lines:
            low = txt.lower()
            if any(k in low for k in ("total", "amount due", "payable", "balance", "amount")) and AMT.findall(txt):
                items.insert(0, dict(kind="total", value=AMT.findall(txt)[-1], conf=round(conf))); break
        return items
    items = []  # form
    for txt, conf in lines:
        label = txt.split(":")[0].strip() if ":" in txt else re.split(r"_{2,}|\.{4,}", txt)[0].strip() if re.search(r"_{2,}|\.{4,}", txt) else ""
        letters = sum(ch.isalpha() or ch == " " for ch in label) / max(1, len(label))
        if 2 < len(label) <= 30 and letters >= 0.9 and conf >= 50 and all(len(wd) >= 2 for wd in label.split()):
            items.append(dict(kind="field", value=label, conf=round(conf)))
    return items

def _detect(lines):
    t = " ".join(l for l, _ in lines).lower()
    sc = dict(medicine=sum(k in t for k in ("tablet", "capsule", " mg", "dosage", "batch no")),
              food=sum(k in t for k in ("exp", "best before", "mfg", "mrp", "use by", "ingredients", "net weight", "net wt", "nutrition", "kcal", "manufactured")),
              bill=sum(k in t for k in ("invoice", "total", "bill", "gst", "receipt", "amount due")),
              form=sum(k in t for k in ("signature", "date of birth", "applicant", "father", "mobile no", "application")))
    if sc["form"] < 2: sc["form"] = 0   # a form needs strong evidence
    if sc["bill"] < 2: sc["bill"] = 0
    best = max(sc, key=sc.get)
    return best if sc[best] else "food"

MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}

def _parse_date(txt):
    t = txt.lower(); nums = [int(x) for x in re.findall(r"\d+", t)]; nm = re.search(r"[a-z]{3}", t)
    try:
        if nm and nm.group() in MONTHS: mo, yr, day = MONTHS[nm.group()], nums[-1], (nums[0] if len(nums) > 1 else None)
        elif len(nums) >= 3: day, mo, yr = nums[-3:]
        else: mo, yr = nums[-2:]; day = None
        yr += 2000 if yr < 100 else 0
        return datetime.date(yr, mo, day or calendar.monthrange(yr, mo)[1])
    except Exception:
        return None

def _is_anchor(t):
    k = re.sub(r"[^a-z]", "", t.lower())
    return k.startswith("exp") or k in ("pkd", "prd", "mfg", "mfd", "useby") or k.startswith("bestb")

def _anchor_zoom(g, ws, mode):
    """A label word like EXP was SEEN: zoom 4x onto the text beside it, with 3 different cleanups, and read again."""
    anchors = sorted([w for w in ws if _is_anchor(w["t"])], key=lambda w: not re.sub(r"[^a-z]", "", w["t"].lower()).startswith("exp"))
    for a in anchors[:4]:
        h = a["h"]; pad = int(h * .8)
        crop = g[max(0, a["y"] - pad):a["y"] + h + pad, max(0, a["x"] - h // 2):a["x"] + a["w"] + h * 14]
        if crop.size == 0: continue
        c4 = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        for v in (c4, cv2.threshold(c4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                  cv2.adaptiveThreshold(c4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)):
            for psm in (6, 7):
                items = extract(mode, _lines(_ocr(v, psm)))
                if items: return items, anchors
    return [], anchors

def _ink_scan(img, mode):
    """Coloured stamp ink (violet/blue) stands out on foil: find the densest ink cluster with the blue-minus-green map,
    crop it, read it in several orientations/cleanups, and VOTE across the readings."""
    b, g, _ = [c.astype(np.int16) for c in cv2.split(img)]
    ink = np.clip((b - g) * 3, 0, 255).astype(np.uint8)
    m = (ink > 90).astype(np.uint8) * 255
    if m.sum() / 255 < 800: return [], None
    mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    cn, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max((cv2.boundingRect(c) for c in cn), key=lambda r: m[r[1]:r[1] + r[3], r[0]:r[0] + r[2]].sum())
    p = 25; y0, x0 = max(0, y - p), max(0, x - p)
    gray = cv2.cvtColor(img[y0:y + h + p, x0:x + w + p], cv2.COLOR_BGR2GRAY); inkc = ink[y0:y + h + p, x0:x + w + p]
    bh = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))), None, 0, 255, cv2.NORM_MINMAX)
    variants = [255 - cv2.threshold(inkc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 255 - inkc, 255 - cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    rots = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE] if h > w else [None, cv2.ROTATE_180]
    votes = {}
    for v in variants:
        for sc in (2, 3):
            v2 = cv2.resize(v, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            for r in rots:
                vv = v2 if r is None else cv2.rotate(v2, r)
                for psm in (11, 6):
                    for d in _dates(_lines(_ocr(vv, psm))):
                        e = votes.setdefault((d["kind"], d["value"]), [0, 0]); e[0] += 1; e[1] = max(e[1], d["conf"])
        if any(k[0] == "expiry" and e[0] >= 3 for k, e in votes.items()): break  # enough agreeing readings
    cands = [(k, n, c) for k, (n, c) in votes.items() if k[0] != "made"]
    exp = [c for c in cands if c[0][0] == "expiry"]
    if exp:
        (kind, val), n, c = max(exp, key=lambda c: (c[1], c[2])); return [dict(kind=kind, value=val, conf=c)], (x, y, w, h)
    if cands:  # no readable label: the LATEST date on a stamp is normally the expiry; say so with low confidence
        (kind, val), n, c = max(cands, key=lambda c: _parse_date(c[0][1]) or datetime.date.min)
        return [dict(kind="date", value=val, conf=min(c, 50))], (x, y, w, h)
    return [], (x, y, w, h)

def _summary(mode, items, status, mean):
    if status == "NEED_MORE": return MSG["turn"]
    if status == "NEED_CLOSER": return MSG["closer_date"]
    if status == "UNSURE" and (not items or (mode in ("bill", "form") and mean < 50)): return MSG["unsure"]
    first = lambda k: next((i["value"] for i in items if i["kind"] == k), None)
    if mode in ("medicine", "food"):
        k = "expiry" if first("expiry") else "date"; val = first(k)
        en = f"{'Expiry date' if k == 'expiry' else 'A date, not labelled'}: {val}."
        ta = f"{'காலாவதி தேதி' if k == 'expiry' else 'ஒரு தேதி'}: {val}."
        d = _parse_date(val)
        if d and d < datetime.date.today(): en += " Warning: this item has expired. Do not use it."; ta += " எச்சரிக்கை: இது காலாவதியாகிவிட்டது. பயன்படுத்த வேண்டாம்."
        if mode == "medicine": en += " Please confirm with your pharmacist."; ta += " மருந்தாளரிடம் உறுதிசெய்யவும்."
    elif mode == "bill":
        parts = [f"{i['kind']}: {i['value']}" for i in items[:3]]
        en = "Bill. " + ". ".join(parts) + "."; ta = "பில். " + ". ".join(parts) + "."
    else:
        f = ", ".join(i["value"] for i in items[:6])
        en = f"This form asks for: {f}."; ta = f"இந்தப் படிவம் கேட்பது: {f}."
    if status == "UNSURE": en = "I am not fully sure. " + en; ta = "எனக்கு முழு உறுதியில்லை. " + ta
    return en, ta

def _tile_scan(g, n=3):
    """Look region by region: overlapping tiles, enlarged 2x, adaptive-threshold, read each."""
    h, w = g.shape; th, tw = h // n, w // n; out = []
    for r in range(n):
        for c in range(n):
            y0, x0 = max(0, r * th - th // 4), max(0, c * tw - tw // 4)
            tile = g[y0:min(h, (r + 1) * th + th // 4), x0:min(w, (c + 1) * tw + tw // 4)]
            tile = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            tile = cv2.adaptiveThreshold(tile, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
            out += _lines(_ocr(tile, 11))
    return out

ROT = [(90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]

def read_document(img, mode="auto"):
    """Agent loop: straighten -> read -> zoom weak words -> if the date is missing CHANGE HOW WE LOOK
    (sparse read, zoom on the EXP label, tile scan, rotations) -> ask the person to move closer / turn the item."""
    mode = "food" if mode == "food_expiry" else mode
    trace = []
    if max(img.shape[:2]) > 2000:
        f = 2000 / max(img.shape[:2]); img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        trace.append("Photo resized to 2000px for speed")
    q, cov = find_doc(img)
    if q is not None:
        img = warp(img, q); trace.append(f"Item found ({cov:.0%} of frame): straightened with perspective transform")
    else:
        trace.append("No clear outline: reading the whole frame")
    g = _enhance(img)
    words = _ocr(g, 6)
    trace.append(f"Pass 1 (contrast-enhanced): {len(words)} words")
    _fix_weak(g, words, trace)
    lines = _lines(words)
    if mode == "auto":
        mode = _detect(lines); trace.append(f"Document type guessed: {mode}")
    items = extract(mode, lines)
    need = mode in ("medicine", "food"); closer = False
    if need and not items:
        trace.append("No date found: changing how I look")
        ws = _ocr(g, 11); l2 = _lines(ws); items = extract(mode, l2)
        trace.append(f"Re-look with sparse-text mode: {'DATE FOUND' if items else 'nothing'}")
        if items: words, lines = ws, l2
        if not items:
            items, box = _ink_scan(img, mode)
            if box: trace.append(f"Found a coloured-ink stamp region {box[2]}x{box[3]}px: tried 4 orientations x 3 cleanups -> {'DATE FOUND' if items else 'nothing'}")
        if not items:
            items, anchors = _anchor_zoom(g, ws, mode); seen = [w["t"] for w in anchors]
            if items: trace.append(f"Zoomed 4x on label word(s) {seen}: DATE FOUND")
            elif any(re.sub(r"[^a-z]", "", w["t"].lower()).startswith("exp") for w in anchors):
                closer = True; trace.append(f"Saw label word(s) {seen} but could not read the date: asking the person to move closer")
        if not items and not closer:
            l3 = _tile_scan(g); items = extract(mode, l3)
            trace.append(f"Re-look with tile-by-tile zoom scan: {'DATE FOUND' if items else 'nothing'}")
            if items: lines = l3
        if not items and not closer:
            gs = g if max(g.shape) <= 1400 else cv2.resize(g, None, fx=1400 / max(g.shape), fy=1400 / max(g.shape))
            for a, c in ROT:
                l4 = _lines(_ocr(cv2.rotate(gs, c), 6)); items = extract(mode, l4)
                trace.append(f"Re-look with rotated {a} degrees: {'DATE FOUND' if items else 'nothing'}")
                if items: lines = l4; break
    weak = sum(1 for w in words if w["c"] < TH["conf"] and len(w["t"]) > 1)
    mean = round(sum(w["c"] for w in words) / len(words), 1) if words else 0.0
    if need:
        status = "NEED_CLOSER" if closer else "NEED_MORE" if not items else "DONE" if min(i["conf"] for i in items) >= TH["item_conf"] else "UNSURE"
    else:
        bad = len(words) < 3 or weak / len(words) > 0.25 or not items
        status = "UNSURE" if bad or min(i["conf"] for i in items) < TH["item_conf"] else "DONE"
    en, ta = _summary(mode, items, status, mean)
    trace.append(f"Decision: {status} (type {mode}, mean confidence {mean}, {len(items)} key items)")
    return dict(text="\n".join(l for l, _ in lines), mode=mode, items=items, status=status, decision=status,
                summary_en=en, summary_ta=ta, mean_conf=mean, weak_words=weak, trace=trace)
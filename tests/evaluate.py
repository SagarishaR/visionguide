"""Run every image in a folder through the agent and print the decision-path table for your report.
Usage: python tests/evaluate.py photos/ [mode]
Name files <condition>_<n>.jpg: clear, dark, glare, blur, far, close, tilt  (e.g. dark_03.jpg) to get accuracy per condition."""
import sys, os, csv, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vision
EXPECT = {"clear": {"ok"}, "tilt": {"ok"}, "dark": {"dark"}, "glare": {"glare"}, "blur": {"blur", "nodoc"}, "far": {"closer", "nodoc"}, "close": {"farther"}}
mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
rows, stats = [], {}
for f in sorted(os.listdir(sys.argv[1])):
    img = cv2.imread(os.path.join(sys.argv[1], f))
    if img is None: continue
    q = vision.quality(img); key = vision.guide(q)
    if key != "ok": path = f"COACH: {key}"
    else:
        r = vision.read_document(img, mode)
        path = "READ -> " + ("re-look -> " if any("Re-look" in t for t in r["trace"]) else "") + ("zoom -> " if any("zoomed" in t for t in r["trace"]) else "") + r["status"] + " " + "; ".join(f"{i['kind']}={i['value']}" for i in r["items"][:2])
    cond = f.split("_")[0]; right = key in EXPECT[cond] if cond in EXPECT else ""
    if cond in EXPECT: s = stats.setdefault(cond, [0, 0]); s[0] += bool(right); s[1] += 1
    rows.append([f, q["blur"], q["brightness"], q["glare"], q["cover"], path, right])
with open("results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "blur", "brightness", "glare", "cover", "path", "coaching_correct"]); w.writerows(rows)
print("| file | blur | bright | glare | cover | decision path | coaching correct |\n|---|---|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(map(str, r)) + " |")
print("\n| condition | correct | total |\n|---|---|---|")
for k, (a, b) in stats.items(): print(f"| {k} | {a} | {b} |")

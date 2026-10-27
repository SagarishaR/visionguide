import os, json, time, logging, cv2, numpy as np
from flask import Flask, request, jsonify, render_template
import vision

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")  # JSON lines -> CloudWatch / journald
BUCKET = os.environ.get("RESULT_BUCKET")  # optional: save results to S3


def load():
    arr = np.frombuffer(request.files["frame"].read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bad image")
    return img


def log(event, **kw):
    logging.info(json.dumps(dict(event=event, ts=time.time(), **kw), ensure_ascii=False))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(ok=True, opencv=cv2.__version__)


@app.route("/analyze", methods=["POST"])
def analyze():
    q = vision.quality(load())
    key = vision.guide(q)
    en, ta = vision.message(key, q["glare_side"])
    log("analyze", key=key, **q)
    return jsonify(key=key, quality=q, en=en, ta=ta)


@app.route("/read", methods=["POST"])
def read():
    img = load()
    q = vision.quality(img)
    res = vision.read_document(img, request.form.get("mode", "auto"))
    res.update(quality=q, guide=vision.guide(q))
    log("read", mode=res["mode"], status=res["status"], mean_conf=res["mean_conf"], items=len(res["items"]))
    if BUCKET:
        try:
            import boto3
            boto3.client("s3").put_object(Bucket=BUCKET, Key=f"results/{int(time.time())}.json",
                                          Body=json.dumps(res, ensure_ascii=False))
        except Exception as e:
            log("s3_error", err=str(e))
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
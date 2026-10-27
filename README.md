# VisionGuide: voice-guided reader for low-vision users (medicine, food expiry, bills, forms)

A person who cannot see the phone screen well points the camera at a medicine box. OpenCV 5 judges every frame
(darkness, glare, blur, box position/size) and the system speaks ONE instruction at a time (English/Tamil).
When the picture is good it straightens the box (perspective transform), enhances contrast (CLAHE), reads the text (Tesseract),
zooms into low-confidence words and re-reads them, then either reads aloud or says "ask a pharmacist".
It never gives medical advice.

## Run locally (camera works on localhost)
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    sudo apt install tesseract-ocr                         # Windows: install from github.com/UB-Mannheim/tesseract
    python -c "import cv2; print(cv2.__version__)"         # must start with 5
    python app.py                                          # open http://localhost:5001

## Test + evidence table
    python tests/evaluate.py photos/     # prints the decision-path table, writes results.csv

## Deploy on AWS (EC2, Ubuntu)
1. Launch a small Ubuntu EC2 instance (check Free Plan eligibility); open port 8443 in the security group.
2. `sudo apt update && sudo apt install -y python3-venv tesseract-ocr openssl`
3. Copy this folder over, then: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
4. Phone cameras need HTTPS: `openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 30 -subj "/CN=visionguide"`
5. `RESULT_BUCKET=your-bucket gunicorn -b 0.0.0.0:8443 --certfile cert.pem --keyfile key.pem app:app` (RESULT_BUCKET optional; give the instance an IAM role with S3 write)
6. Open `https://<public-ip>:8443` and accept the certificate warning.
7. Logs are JSON lines on stdout: run under systemd and ship with the CloudWatch agent (observability).
8. Pin versions for submission: `pip freeze > requirements.lock`

## Files
app.py (web server, logging, S3) | vision.py (OpenCV tools + guidance logic) | templates/index.html (camera, voice) | tests/evaluate.py (evidence)

## Modes (dropdown in the app)
- Medicine / Food expiry: finds the EXPIRY date. If none is found the agent changes how it looks (sparse-text OCR, 90/180/270 degree rotation); if still none it tells the person to turn the item over, then looks again (max 3 attempts).
- Bill: finds the total amount and due date. Form: reads out the field labels (Name, Date of Birth ...).
- Auto-detect guesses the type from keywords.

## Evaluation naming
Name test photos `clear_01.jpg, dark_02.jpg, glare_03.jpg, blur_04.jpg, far_05.jpg, close_06.jpg, tilt_07.jpg`, then `python tests/evaluate.py photos/ food`.

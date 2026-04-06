"""
lab2.py — CEG4195 Lab 2
------------------------
Flask REST API for aerial building segmentation using a trained UNet model.
Extends Lab 1's pattern: os.getenv config, jsonify responses, module-level model load.

Endpoints:
    GET  /          Info page with endpoint list
    GET  /health    Status check (model path, threshold, device)
    POST /segment   Accepts an image file, returns binary building mask as base64 PNG

Run:
    python lab2.py

Requires:
    models/unet_resnet34.pth  (produced by train.py)

Environment variables (via .env or docker-compose env_file):
    MODEL_PATH      path to trained checkpoint   (default: models/unet_resnet34.pth)
    SEG_THRESHOLD   pixel decision boundary       (default: 0.5)
    PORT            Flask listen port             (default: 5000)
"""

import os
import io
import base64

import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from PIL import Image

import segmentation_models_pytorch as smp

# ── Load secrets / config from .env before anything else ──────────────────────
load_dotenv()

MODEL_PATH    = os.getenv("MODEL_PATH",    "models/unet_resnet34.pth")
SEG_THRESHOLD = float(os.getenv("SEG_THRESHOLD", "0.5"))
PORT          = int(os.getenv("PORT", "5000"))

# ImageNet normalisation — must match the values used during training
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Model loading (once at startup, mirrors Lab 1's pipeline(...) pattern) ─────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'. "
        "Run train.py first, then restart the server."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # weights already embedded in the .pth file
    in_channels=3,
    classes=1,
    activation=None,        # raw logits; sigmoid applied at inference time
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "message": "CEG4195 Lab 2: Building Segmentation Service",
        "endpoints": {
            "health":  "GET /health",
            "segment": "POST /segment  (multipart: file=<image>)",
        }
    })


@app.get("/health")
def health():
    return jsonify({
        "status":    "ok",
        "model":     MODEL_PATH,
        "threshold": SEG_THRESHOLD,
        "device":    str(device),
    })


@app.post("/segment")
def segment():
    if "file" not in request.files:
        return jsonify({"error": "No 'file' field in request. Send multipart/form-data with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename. Attach an image file."}), 400

    # ── Preprocess
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not decode image. Send a valid PNG/JPEG/TIFF file."}), 400

    original_size = list(img.size)               # (W, H) before resize
    img_resized   = img.resize((256, 256))
    img_array     = np.array(img_resized, dtype=np.float32) / 255.0
    img_norm      = (img_array - _MEAN) / _STD   # ImageNet normalisation

    # HWC → CHW → batch dimension
    tensor = (
        torch.from_numpy(img_norm.transpose(2, 0, 1))
             .unsqueeze(0)
             .float()
             .to(device)
    )

    # ── Inference
    with torch.no_grad():
        logits = model(tensor)                            # (1, 1, 256, 256)
        probs  = torch.sigmoid(logits)                    # (1, 1, 256, 256)
        mask   = (probs > SEG_THRESHOLD).squeeze().cpu().numpy().astype(np.uint8) * 255

    # ── Encode mask as PNG → base64
    mask_img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    building_fraction = float(mask.sum()) / (255.0 * 256 * 256)

    return jsonify({
        "mask_png_base64":  mask_b64,
        "building_fraction": round(building_fraction, 4),
        "threshold_used":   SEG_THRESHOLD,
        "input_size":        original_size,
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)

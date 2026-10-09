"""AI Detector — Flask Web Application"""

import dataclasses
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"}


def _result_to_dict(result) -> dict:
    """Convert dataclass result to JSON-serialisable dict."""
    d = dataclasses.asdict(result)
    return d


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze/text", methods=["POST"])
def analyze_text_route():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    from detector.text_detector import analyze_text
    result = analyze_text(text)
    return jsonify(_result_to_dict(result))


@app.route("/analyze/image", methods=["POST"])
def analyze_image_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    image_bytes = f.read()
    from detector.image_detector import analyze_image
    result = analyze_image(image_bytes, f.filename)
    return jsonify(_result_to_dict(result))


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Detector — Python Edition")
    print("  http://127.0.0.1:5000")
    print("  Models load on first request (~30s on first run)")
    print("=" * 55)
    app.run(debug=False, host="127.0.0.1", port=5000)

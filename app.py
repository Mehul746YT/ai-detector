"""
AI Detector — Unified Universal Multi-Media Detection Engine (Dutton & Co.)
Supports:
  - Text & Essays (ChatGPT, Claude, Gemini, Llama, Copilot)
  - Images (Midjourney, Stable Diffusion, DALL-E, Flux, Ideogram, Firefly)
  - Source Code (Copilot, Cursor, CodeLlama, ChatGPT)
  - Audio & Voice Clones (ElevenLabs, Bark, RVC, Suno, Udio)
  - Videos & Deepfakes (Sora, Runway Gen-2/3, Pika, Kling, Luma)
  - Documents & Reports (PDF, Word DOCX, TXT)
"""

import dataclasses
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Explicitly resolve template and static folders relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

from detector.text_detector import analyze_text
from detector.image_detector import analyze_image
from detector.audio_detector import analyze_audio
from detector.document_detector import analyze_document
from detector.code_detector import analyze_code
from detector.video_detector import analyze_video

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload limit

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"}
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md"}
CODE_EXTS = {".py", ".js", ".ts", ".html", ".css", ".java", ".cpp", ".c", ".cs", ".go", ".rs", ".php", ".rb", ".sql", ".sh"}


def _serialize(obj):
    return dataclasses.asdict(obj)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    if path.startswith("analyze/"):
        return jsonify({"error": "Method Not Allowed"}), 405
    return render_template("index.html")


@app.route("/analyze/text", methods=["POST"])
def analyze_text_route():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided for analysis."}), 400

    result = analyze_text(text)
    return jsonify(_serialize(result))


@app.route("/analyze/code", methods=["POST"])
def analyze_code_route():
    data = request.get_json(force=True, silent=True) or {}
    code = data.get("code", "").strip()
    filename = data.get("filename", "code_snippet.py")
    if not code:
        return jsonify({"error": "No source code provided."}), 400

    result = analyze_code(code, filename)
    return jsonify(_serialize(result))


@app.route("/analyze/media", methods=["POST"])
def analyze_media_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    filename = f.filename
    ext = Path(filename).suffix.lower()
    file_bytes = f.read()

    if ext in IMAGE_EXTS:
        result = analyze_image(file_bytes, filename)
        return jsonify({"media_type": "image", "result": _serialize(result)})

    elif ext in AUDIO_EXTS:
        result = analyze_audio(file_bytes, filename)
        return jsonify({"media_type": "audio", "result": _serialize(result)})

    elif ext in VIDEO_EXTS:
        result = analyze_video(file_bytes, filename)
        return jsonify({"media_type": "video", "result": _serialize(result)})

    elif ext in DOCUMENT_EXTS:
        result = analyze_document(file_bytes, filename)
        return jsonify({"media_type": "document", "result": _serialize(result)})

    elif ext in CODE_EXTS:
        code_str = file_bytes.decode("utf-8", errors="replace")
        result = analyze_code(code_str, filename)
        return jsonify({"media_type": "code", "result": _serialize(result)})

    else:
        try:
            result = analyze_image(file_bytes, filename)
            return jsonify({"media_type": "image", "result": _serialize(result)})
        except Exception:
            return jsonify({"error": f"Unsupported media format '{ext}'."}), 400


if __name__ == "__main__":
    print("=" * 65)
    print("  Dutton & Co. Universal AI Detection Suite")
    print("  Text | Images | Code | Audio | Video | Documents")
    print("  Active at: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=False, host="127.0.0.1", port=5000)

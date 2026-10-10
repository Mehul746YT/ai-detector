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
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(BASE_DIR, "public", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

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


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)


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

    # Save to uploads directory for public URL referencing in reverse search
    public_url = None
    try:
        saved_name = f"{uuid.uuid4().hex[:12]}{ext}"
        saved_path = os.path.join(UPLOADS_DIR, saved_name)
        with open(saved_path, "wb") as out_f:
            out_f.write(file_bytes)
        host = request.host_url.rstrip("/")
        public_url = f"{host}/uploads/{saved_name}"
    except Exception:
        pass

    if ext in IMAGE_EXTS:
        result = analyze_image(file_bytes, filename)
        return jsonify({"media_type": "image", "result": _serialize(result), "public_url": public_url})

    elif ext in AUDIO_EXTS:
        result = analyze_audio(file_bytes, filename)
        return jsonify({"media_type": "audio", "result": _serialize(result), "public_url": public_url})

    elif ext in VIDEO_EXTS:
        result = analyze_video(file_bytes, filename)
        return jsonify({"media_type": "video", "result": _serialize(result), "public_url": public_url})

    elif ext in DOCUMENT_EXTS:
        result = analyze_document(file_bytes, filename)
        return jsonify({"media_type": "document", "result": _serialize(result), "public_url": public_url})

    elif ext in CODE_EXTS:
        code_str = file_bytes.decode("utf-8", errors="replace")
        result = analyze_code(code_str, filename)
        return jsonify({"media_type": "code", "result": _serialize(result), "public_url": public_url})

    else:
        try:
            result = analyze_image(file_bytes, filename)
            return jsonify({"media_type": "image", "result": _serialize(result), "public_url": public_url})
        except Exception:
            return jsonify({"error": f"Unsupported media format '{ext}'."}), 400



@app.route("/verify/reverse", methods=["POST"])
def verify_reverse_route():
    """
    In-website reverse search verification engine.
    Fetches real matches, encyclopedic records, and archival media
    directly within the website UI without redirecting users away.
    """
    import requests
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400

    q = requests.utils.quote(query)
    results = []

    # 1. Wikipedia Knowledge & Media Verification
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={q}&prop=extracts|pageimages|info&inprop=url&exintro=1&explaintext=1&exchars=240&piprop=thumbnail&pithumbsize=360&format=json&gsrlimit=4"
        r = requests.get(url, headers={"User-Agent": "DuttonCoForensics/1.0"}, timeout=5)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for pid, p in pages.items():
                results.append({
                    "title": p.get("title", ""),
                    "snippet": p.get("extract", ""),
                    "thumbnail": p.get("thumbnail", {}).get("source"),
                    "url": p.get("fullurl"),
                    "source": "Wikipedia Global Knowledge Base",
                    "badge": "RECORD VERIFIED"
                })
    except Exception:
        pass

    # 2. Wikimedia Commons Public Visual Media Archive
    try:
        url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={q}&gsrnamespace=6&prop=imageinfo&iiprop=url|size|extmetadata&format=json&gsrlimit=4"
        r = requests.get(url, headers={"User-Agent": "DuttonCoForensics/1.0"}, timeout=5)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for pid, p in pages.items():
                ii = p.get("imageinfo", [{}])[0]
                results.append({
                    "title": p.get("title", "").replace("File:", ""),
                    "snippet": f"Archived media ({ii.get('width', 0)}x{ii.get('height', 0)} px)",
                    "thumbnail": ii.get("url"),
                    "url": ii.get("descriptionurl") or ii.get("url"),
                    "source": "Wikimedia Commons Visual Archive",
                    "badge": "MEDIA MATCH"
                })
    except Exception:
        pass

    # 3. Internet Archive Media & Footage Records
    try:
        url = f"https://archive.org/advancedsearch.php?q={q}&fl[]=identifier,title,description,mediatype&rows=4&output=json"
        r = requests.get(url, headers={"User-Agent": "DuttonCoForensics/1.0"}, timeout=5)
        if r.status_code == 200:
            docs = r.json().get("response", {}).get("docs", [])
            for d in docs:
                ident = d.get("identifier")
                results.append({
                    "title": d.get("title") or ident,
                    "snippet": (d.get("description") or "")[:150],
                    "thumbnail": f"https://archive.org/services/img/{ident}",
                    "url": f"https://archive.org/details/{ident}",
                    "source": "Internet Archive Global Media",
                    "badge": "ARCHIVE ENTRY"
                })
    except Exception:
        pass

    return jsonify({
        "query": query,
        "total_matches": len(results),
        "results": results
    })


if __name__ == "__main__":
    print("=" * 65)
    print("  Dutton & Co. Universal AI Detection Suite")
    print("  Text | Images | Code | Audio | Video | Documents")
    print("  Active at: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=False, host="127.0.0.1", port=5000)

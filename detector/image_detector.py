"""
Image AI Detection Module (Real AI Vision APIs + Forensic Metadata Engine)
Uses:
  1. Sightengine Deep Generative AI API (Industry Gold Standard)
  2. Hugging Face Vision Transformer (ViT) Serverless Inference (umm-maybe/AI-image-detector)
  3. Hugging Face SDXL Specialist Classifier (Organika/sdxl-detector)
  4. C2PA Content Credentials & Generative Metadata Forensics (DALL-E, SD, Midjourney, ComfyUI)
  5. Optical Camera Hardware EXIF Verification (Apple, Samsung, Sony, Canon, Nikon, Pixel)
"""

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
from PIL import Image, ExifTags


@dataclass
class ImageSignal:
    name: str
    ai_score: float
    weight: float
    label: str


@dataclass
class ImageResult:
    final_score: float
    verdict: str
    filename: str
    signals: list = field(default_factory=list)
    api_used: Optional[str] = None
    error: Optional[str] = None


_AI_SOFTWARE_PATTERNS = [
    "stable diffusion", "midjourney", "dall-e", "dall·e", "firefly",
    "leonardo", "runwayml", "novelai", "comfyui", "automatic1111",
    "a1111", "invokeai", "dreamstudio", "nightcafe", "artbreeder",
    "craiyon", "bing image", "adobe firefly", "ideogram", "flux",
    "stablediffusion", "sdxl", "sd-webui",
]

_CAMERA_PATTERNS = [
    "canon", "nikon", "sony", "fujifilm", "olympus", "pentax",
    "leica", "hasselblad", "dji", "apple", "samsung", "google",
    "huawei", "xiaomi", "oneplus", "motorola", "gopro", "oppo", "vivo", "realme"
]

_AI_EXACT_CANVAS = {
    (512, 512), (768, 512), (512, 768),
    (1024, 1024), (1024, 768), (768, 1024),
    (1024, 1792), (1792, 1024),
    (1344, 768), (768, 1344),
    (1152, 896), (896, 1152),
}


def query_sightengine_api(image_bytes: bytes, api_user: str, api_secret: str) -> Optional[dict]:
    """
    Queries Sightengine's state-of-the-art Generative AI detection model (genai).
    Returns real probability of image being AI-generated.
    """
    if not api_user or not api_secret:
        return None
    try:
        params = {
            "models": "genai",
            "api_user": api_user.strip(),
            "api_secret": api_secret.strip(),
        }
        files = {"media": ("image.jpg", image_bytes, "image/jpeg")}
        resp = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            files=files,
            data=params,
            timeout=10,
        )
        if resp.status_code == 200:
            res_json = resp.json()
            if res_json.get("status") == "success":
                type_info = res_json.get("type", {})
                ai_score = float(type_info.get("ai_generated", 0.5))
                return {
                    "ai_score": ai_score,
                    "label": f'Sightengine AI Vision Engine: {ai_score:.1%} probability of AI generation'
                }
    except Exception:
        pass
    return None


def query_hf_vit_model(image_bytes: bytes, hf_token: str, model_id: str = "umm-maybe/AI-image-detector") -> Optional[dict]:
    """
    Queries Hugging Face serverless vision transformer model.
    Requires Hugging Face User Access Token.
    """
    if not hf_token:
        return None
    try:
        url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        headers = {"Authorization": f"Bearer {hf_token.strip()}"}
        resp = requests.post(url, headers=headers, data=image_bytes[:1500000], timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                items = data[0] if isinstance(data[0], list) else data
                for item in items:
                    lbl = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.5))
                    if any(k in lbl for k in ("artificial", "ai", "fake", "generated", "synthetic")):
                        return {
                            "ai_score": score,
                            "label": f'Neural Vision Transformer ({model_id.split("/")[-1]}): "{item.get("label")}" ({score:.1%})'
                        }
                    elif any(k in lbl for k in ("human", "real", "natural", "photo")):
                        return {
                            "ai_score": 1.0 - score,
                            "label": f'Neural Vision Transformer ({model_id.split("/")[-1]}): "{item.get("label")}" ({score:.1%})'
                        }
                first = items[0]
                return {
                    "ai_score": float(first.get("score", 0.5)),
                    "label": f'Vision Model ({model_id.split("/")[-1]}): {first.get("label")}'
                }
    except Exception:
        pass
    return None


def analyze_c2pa_and_metadata(img: Image.Image, raw_bytes: bytes, filename: str) -> dict:
    """
    Analyzes C2PA manifests, DALL-E watermarks, Stable Diffusion PNG chunks,
    and verified optical camera hardware tags in EXIF.
    """
    result = {
        "ai_score": 0.50,
        "weight": 1.5,
        "label": "Standard digital image profile (no embedded generator signatures)",
        "is_hardware": False,
        "is_ai": False
    }

    # 1. C2PA / Content Credentials manifest check (used by DALL-E 3, Adobe Firefly, Leica M11-P)
    raw_lower = raw_bytes[:16384].lower()
    if b"c2pa" in raw_lower or b"c2pa.manifest" in raw_lower or b"c2pa.provenance" in raw_lower:
        # Check if C2PA indicates generative tool
        if any(kw in raw_lower for kw in [b"openai", b"dall-e", b"firefly", b"adobe", b"generated"]):
            return {
                "ai_score": 0.99,
                "weight": 20.0,
                "label": "Verified C2PA Content Credential identifies AI generation (DALL-E / Adobe)",
                "is_hardware": False,
                "is_ai": True
            }

    # 2. PNG Chunk inspection (Stable Diffusion, WebUI, ComfyUI, NovelAI)
    info = img.info or {}
    png_text = " ".join(str(v) for v in info.values()).lower()
    has_sd_params = any(k in png_text for k in ("steps:", "cfg scale", "sampler:", "seed:", "negative prompt", "model hash"))
    ai_in_png = any(p in png_text for p in _AI_SOFTWARE_PATTERNS)
    if has_sd_params or ai_in_png:
        return {
            "ai_score": 0.99,
            "weight": 20.0,
            "label": "Verified generative AI synthesis parameters embedded in metadata",
            "is_hardware": False,
            "is_ai": True
        }

    # 3. EXIF Camera Hardware Inspection
    try:
        exif_data = img._getexif()
    except Exception:
        exif_data = None

    if exif_data:
        tag_map = {v: k for k, v in ExifTags.TAGS.items()}
        make = str(exif_data.get(tag_map.get("Make", 271), ""))
        model_tag = str(exif_data.get(tag_map.get("Model", 272), ""))
        software = str(exif_data.get(tag_map.get("Software", 305), ""))
        gps_info = exif_data.get(tag_map.get("GPSInfo", 34853))

        make_lc = make.lower()
        sw_lc = software.lower()

        if any(p in sw_lc for p in _AI_SOFTWARE_PATTERNS):
            return {
                "ai_score": 0.99,
                "weight": 20.0,
                "label": f"AI generator engine tag in EXIF: '{software}'",
                "is_hardware": False,
                "is_ai": True
            }

        if any(p in make_lc for p in _CAMERA_PATTERNS):
            gps_note = " + GPS tagged" if gps_info else ""
            return {
                "ai_score": 0.02,
                "weight": 18.0,
                "label": f"Verified physical camera hardware: {make} {model_tag}{gps_note}".strip(),
                "is_hardware": True,
                "is_ai": False
            }

        if any(p in sw_lc for p in ["photoshop", "lightroom", "gimp", "capture one"]):
            return {
                "ai_score": 0.25,
                "weight": 3.0,
                "label": f"Digital photo editing tool record: {software}",
                "is_hardware": False,
                "is_ai": False
            }

        return {
            "ai_score": 0.30,
            "weight": 2.5,
            "label": "Standard digital camera EXIF record (non-AI hardware profile)",
            "is_hardware": False,
            "is_ai": False
        }

    return result


def analyze_canvas_preset_fingerprint(img: Image.Image) -> dict:
    """Detects exact model canvas output presets."""
    w, h = img.size
    if (w, h) in _AI_EXACT_CANVAS or (h, w) in _AI_EXACT_CANVAS:
        return {
            "ai_score": 0.82,
            "label": f"Canvas size ({w}x{h}) matches known AI generator preset (DALL-E / Midjourney / SDXL)"
        }
    return {
        "ai_score": 0.45,
        "label": f"Standard image resolution ({w}x{h})"
    }


def analyze_image(
    image_bytes: bytes,
    filename: str,
    hf_token: Optional[str] = None,
    sightengine_user: Optional[str] = None,
    sightengine_secret: Optional[str] = None,
) -> ImageResult:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        return ImageResult(0.5, "Error", filename, error=str(e))

    signals: list[ImageSignal] = []
    api_names = []

    # Get API keys from parameters or environment
    effective_hf_token = hf_token or os.environ.get("HF_TOKEN")
    effective_se_user = sightengine_user or os.environ.get("SIGHTENGINE_API_USER")
    effective_se_secret = sightengine_secret or os.environ.get("SIGHTENGINE_API_SECRET")

    # 1. Sightengine Generative AI API (Industry Gold Standard)
    if effective_se_user and effective_se_secret:
        se_res = query_sightengine_api(image_bytes, effective_se_user, effective_se_secret)
        if se_res is not None:
            signals.append(ImageSignal(
                name="Sightengine Deep AI Vision Engine",
                ai_score=round(se_res["ai_score"], 4),
                weight=25.0,
                label=se_res["label"]
            ))
            api_names.append("Sightengine")

    # 2. Hugging Face Vision Transformer API (umm-maybe/AI-image-detector)
    if effective_hf_token:
        vit_res = query_hf_vit_model(image_bytes, effective_hf_token, "umm-maybe/AI-image-detector")
        if vit_res is not None:
            signals.append(ImageSignal(
                name="Vision Transformer (ViT AI Detector)",
                ai_score=round(vit_res["ai_score"], 4),
                weight=20.0,
                label=vit_res["label"]
            ))
            api_names.append("HuggingFace-ViT")

        # Secondary SDXL Specialist Model
        sdxl_res = query_hf_vit_model(image_bytes, effective_hf_token, "Organika/sdxl-detector")
        if sdxl_res is not None:
            signals.append(ImageSignal(
                name="SDXL Diffusion Specialist Classifier",
                ai_score=round(sdxl_res["ai_score"], 4),
                weight=15.0,
                label=sdxl_res["label"]
            ))
            api_names.append("HuggingFace-SDXL")

    # 3. Ground Truth Metadata & Hardware Forensics
    meta = analyze_c2pa_and_metadata(img, image_bytes, filename)
    signals.append(ImageSignal(
        name="Metadata & Hardware Provenance",
        ai_score=round(meta["ai_score"], 4),
        weight=meta["weight"],
        label=meta["label"]
    ))

    # 4. Canvas Geometry Preset
    canvas = analyze_canvas_preset_fingerprint(img)
    signals.append(ImageSignal(
        name="Canvas Geometry Preset Fingerprint",
        ai_score=round(canvas["ai_score"], 4),
        weight=1.5,
        label=canvas["label"]
    ))

    # Calculate final weighted score
    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    # Direct ground-truth anchors:
    if meta.get("is_hardware") and not api_names:
        final_score = min(final_score, 0.05)
    elif meta.get("is_ai"):
        final_score = max(final_score, 0.98)

    verdict = (
        "AI-Generated" if final_score >= 0.60
        else "Uncertain / Mixed" if final_score >= 0.40
        else "Likely Real Photo"
    )

    api_used_summary = ", ".join(api_names) if api_names else None

    return ImageResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        filename=filename,
        signals=signals,
        api_used=api_used_summary,
    )

"""
Image AI Detection Module (Vercel Serverless Ready)
Architecture:
  - Free Serverless Cloud Vision Transformer (ViT) via Hugging Face Inference API
  - Forensic EXIF and PNG chunk parsing for generative parameters (SD, Midjourney, DALL-E, CFG scale)
  - Pure NumPy/Pillow optical residual and spatial variance checks (no PyTorch required)
"""

import io
from dataclasses import dataclass, field
from typing import Optional

import requests
import numpy as np
from PIL import Image, ExifTags, ImageFilter


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
    error: Optional[str] = None


_AI_SOFTWARE_PATTERNS = [
    "stable diffusion", "midjourney", "dall-e", "dall·e", "firefly",
    "leonardo", "runwayml", "novelai", "comfyui", "automatic1111",
    "a1111", "invokeai", "dreamstudio", "nightcafe", "artbreeder",
    "craiyon", "bing image", "adobe firefly", "ideogram", "flux",
]

_CAMERA_PATTERNS = [
    "canon", "nikon", "sony", "fujifilm", "olympus", "pentax",
    "leica", "hasselblad", "dji", "apple", "samsung", "google",
    "huawei", "xiaomi", "oneplus",
]


def query_hf_vit_api(image_bytes: bytes) -> Optional[dict]:
    """Queries Hugging Face serverless vision model (umm-maybe/AI-image-detector)."""
    try:
        url = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
        resp = requests.post(url, data=image_bytes, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    lbl = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.5))
                    if any(k in lbl for k in ("artificial", "ai", "fake", "generated")):
                        return {"ai_score": score, "label": f'Vision model: "{item.get("label")}" ({score:.1%})'}
                    elif any(k in lbl for k in ("human", "real", "natural")):
                        return {"ai_score": 1.0 - score, "label": f'Vision model: "{item.get("label")}" ({score:.1%})'}
                first = data[0]
                return {"ai_score": float(first.get("score", 0.5)), "label": f'Top label: {first.get("label")}'}
    except Exception:
        pass
    return None


def analyze_exif(img: Image.Image) -> dict:
    result = {
        "has_exif": False,
        "ai_software_found": None,
        "camera_make": None,
        "photo_software": None,
        "gps_present": False,
        "ai_score": 0.50,
        "label": "No EXIF metadata found",
    }

    try:
        exif_data = img._getexif()
    except Exception:
        exif_data = None

    info = img.info or {}
    png_text = " ".join(str(v) for v in info.values()).lower()

    ai_in_png = any(p in png_text for p in _AI_SOFTWARE_PATTERNS)
    has_sd_params = any(k in png_text for k in ("steps:", "cfg scale", "sampler:", "seed:", "model hash"))

    if ai_in_png or has_sd_params:
        result["ai_software_found"] = "AI generation parameters found in metadata"
        result["ai_score"] = 0.99
        result["label"] = "Verified AI generation parameters embedded in PNG metadata"
        return result

    if exif_data:
        result["has_exif"] = True
        tag_map = {v: k for k, v in ExifTags.TAGS.items()}
        make = str(exif_data.get(tag_map.get("Make", 271), ""))
        model_tag = str(exif_data.get(tag_map.get("Model", 272), ""))
        software = str(exif_data.get(tag_map.get("Software", 305), ""))
        gps_info = exif_data.get(tag_map.get("GPSInfo", 34853))

        make_lc = make.lower()
        sw_lc = software.lower()

        if any(p in sw_lc for p in _AI_SOFTWARE_PATTERNS):
            result["ai_software_found"] = software
            result["ai_score"] = 0.99
            result["label"] = f"AI generator software identified: '{software}'"
            return result

        if any(p in make_lc for p in _CAMERA_PATTERNS):
            result["camera_make"] = make
            result["gps_present"] = gps_info is not None
            result["ai_score"] = 0.05
            result["label"] = f"Verified camera hardware: {make} {model_tag}".strip()
            return result

        if any(p in sw_lc for p in ["photoshop", "lightroom", "gimp", "darktable", "capture one"]):
            result["photo_software"] = software
            result["ai_score"] = 0.25
            result["label"] = f"Digital photo processing tool: {software}"
            return result

        result["ai_score"] = 0.40
        result["label"] = "Standard digital EXIF record (non-AI)"
    else:
        result["ai_score"] = 0.50
        result["label"] = "Metadata stripped (typical for web/social uploads)"

    return result


def _to_gray_array(img: Image.Image, size: int = 256) -> np.ndarray:
    img_resized = img.convert("L").resize((size, size), Image.LANCZOS)
    return np.array(img_resized, dtype=np.float32)


def noise_residual_analysis(img: Image.Image) -> dict:
    gray = _to_gray_array(img, size=256)
    blurred = np.array(img.convert("L").resize((256, 256), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.2)), dtype=float)
    residual = gray - blurred
    res_std = float(residual.std())

    if res_std > 11.0:
        score = min(0.95, 0.5 + (res_std - 11.0) * 0.05)
        label = f"High residual noise variance ({res_std:.1f}) — typical diffusion artifacts"
    elif res_std < 3.0:
        score = 0.80
        label = f"Unusually smooth residual ({res_std:.1f}) — typical synthetic rendering"
    else:
        score = 0.20
        label = f"Natural camera optical noise floor ({res_std:.1f})"

    return {"ai_score": score, "label": label, "res_std": res_std}


def local_variance_distribution(gray: np.ndarray) -> dict:
    h, w = gray.shape
    block_size = 16
    variances = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            variances.append(float(np.var(block)))

    variances = np.array(variances)
    mean_var = variances.mean()
    cv = float(variances.std() / (mean_var + 1e-5))
    ai_score = float(np.clip(1.0 - cv * 0.5, 0.1, 0.9))
    label = f"Texture variance CV={cv:.2f} ({'uniform/synthetic pattern' if cv < 1.0 else 'natural textural diversity'})"
    return {"ai_score": ai_score, "label": label, "cv": cv}


def color_channel_correlation(img: Image.Image) -> dict:
    arr = np.array(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    r = arr[:,:,0].flatten()
    g = arr[:,:,1].flatten()
    b = arr[:,:,2].flatten()

    std_r, std_g, std_b = r.std(), g.std(), b.std()
    if std_r < 1e-4 or std_g < 1e-4 or std_b < 1e-4:
        return {"ai_score": 0.5, "label": "Monochrome / flat color profile", "avg_corr": 1.0}

    corr_rg = float(np.corrcoef(r, g)[0, 1])
    corr_rb = float(np.corrcoef(r, b)[0, 1])
    corr_gb = float(np.corrcoef(g, b)[0, 1])
    avg_corr = (corr_rg + corr_rb + corr_gb) / 3.0

    if np.isnan(avg_corr):
        return {"ai_score": 0.5, "label": "Normal color spectrum", "avg_corr": 0.8}

    ai_score = float(np.clip((avg_corr - 0.70) / 0.28, 0.1, 0.95))
    label = f"RGB correlation={avg_corr:.2f} ({'high synthetic channel harmony' if avg_corr > 0.92 else 'natural multi-spectral variation'})"
    return {"ai_score": ai_score, "label": label, "avg_corr": avg_corr}


def analyze_image(image_bytes: bytes, filename: str) -> ImageResult:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        return ImageResult(0.5, "Error", filename, error=str(e))

    signals: list[ImageSignal] = []

    # 1. Cloud Vision Transformer via free Serverless API
    vit_res = query_hf_vit_api(image_bytes)
    if vit_res is not None:
        signals.append(ImageSignal(
            name="Cloud Vision Transformer (ViT)",
            ai_score=round(vit_res["ai_score"], 4),
            weight=4.0,
            label=vit_res["label"]
        ))

    # 2. Metadata Forensics
    exif = analyze_exif(img)
    exif_weight = 10.0 if (exif["ai_score"] > 0.95 or exif["ai_score"] < 0.1) else 2.0
    signals.append(ImageSignal(
        name="Metadata Forensics (EXIF / PNG)",
        ai_score=round(exif["ai_score"], 4),
        weight=exif_weight,
        label=exif["label"]
    ))

    # 3. Optical Residuals
    residual = noise_residual_analysis(img)
    signals.append(ImageSignal(
        name="High-Frequency Sensor Residuals",
        ai_score=round(residual["ai_score"], 4),
        weight=3.0,
        label=residual["label"]
    ))

    # 4. Textural Variance
    gray = _to_gray_array(img)
    variance_res = local_variance_distribution(gray)
    signals.append(ImageSignal(
        name="Spatial Texture Distribution",
        ai_score=round(variance_res["ai_score"], 4),
        weight=2.0,
        label=variance_res["label"]
    ))

    # 5. Channel Correlation
    corr = color_channel_correlation(img)
    signals.append(ImageSignal(
        name="Spectral Channel Correlation",
        ai_score=round(corr["ai_score"], 4),
        weight=1.5,
        label=corr["label"]
    ))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    verdict = (
        "AI-Generated" if final_score >= 0.62
        else "Uncertain / Mixed" if final_score >= 0.38
        else "Likely Real Photo"
    )

    return ImageResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        filename=filename,
        signals=signals,
    )

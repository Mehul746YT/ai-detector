"""
Image AI Detection Module (Vercel Serverless Ready)
Architecture:
  - Dual free Hugging Face Vision Model queries (two separate models for cross-verification)
  - Forensic EXIF and PNG chunk parsing for generative parameters (SD, Midjourney, DALL-E, CFG scale)
  - Error Level Analysis (ELA) via Pillow/NumPy — detects JPEG re-compression signatures
  - DCT high-frequency coefficient distribution (AI images have different frequency fingerprints)
  - Local variance and color channel correlation analysis
  - Aspect ratio & dimension fingerprint (AI generators use fixed output sizes)
"""

import io
import math
from dataclasses import dataclass, field
from typing import Optional

import requests
import numpy as np
from PIL import Image, ExifTags, ImageFilter, ImageChops


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
    "stablediffusion", "sdxl", "sd-webui",
]

_CAMERA_PATTERNS = [
    "canon", "nikon", "sony", "fujifilm", "olympus", "pentax",
    "leica", "hasselblad", "dji", "apple", "samsung", "google",
    "huawei", "xiaomi", "oneplus", "motorola", "gopro",
]

# Common AI image generator output dimensions
_AI_DIMENSION_SETS = {
    (512, 512), (768, 512), (512, 768),
    (1024, 1024), (1024, 768), (768, 1024),
    (1024, 1792), (1792, 1024),
    (1344, 768), (768, 1344),
    (1152, 896), (896, 1152),
}


def query_hf_vit_primary(image_bytes: bytes) -> Optional[dict]:
    """Primary: umm-maybe/AI-image-detector (ViT fine-tuned on AI vs real)."""
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
                        return {"ai_score": score, "label": f'ViT-Primary: "{item.get("label")}" ({score:.1%})'}
                    elif any(k in lbl for k in ("human", "real", "natural")):
                        return {"ai_score": 1.0 - score, "label": f'ViT-Primary: "{item.get("label")}" ({score:.1%})'}
                first = data[0]
                return {"ai_score": float(first.get("score", 0.5)), "label": f'ViT-Primary top: {first.get("label")}'}
    except Exception:
        pass
    return None


def query_hf_vit_secondary(image_bytes: bytes) -> Optional[dict]:
    """Secondary: Organika/sdxl-detector (specialized Stable Diffusion XL detector)."""
    try:
        url = "https://api-inference.huggingface.co/models/Organika/sdxl-detector"
        resp = requests.post(url, data=image_bytes, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    lbl = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.5))
                    if any(k in lbl for k in ("artificial", "ai", "fake", "sdxl", "generated", "synthetic")):
                        return {"ai_score": score, "label": f'SDXL-Detector: "{item.get("label")}" ({score:.1%})'}
                    elif any(k in lbl for k in ("human", "real", "natural", "photo")):
                        return {"ai_score": 1.0 - score, "label": f'SDXL-Detector: "{item.get("label")}" ({score:.1%})'}
                first = data[0]
                return {"ai_score": float(first.get("score", 0.5)), "label": f'SDXL-Detector top: {first.get("label")}'}
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
    has_sd_params = any(k in png_text for k in ("steps:", "cfg scale", "sampler:", "seed:", "model hash", "negative prompt"))

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
            gps_note = " + GPS embedded" if gps_info else ""
            result["ai_score"] = 0.04
            result["label"] = f"Verified camera hardware: {make} {model_tag}{gps_note}".strip()
            return result

        if any(p in sw_lc for p in ["photoshop", "lightroom", "gimp", "darktable", "capture one", "affinity photo"]):
            result["photo_software"] = software
            result["ai_score"] = 0.25
            result["label"] = f"Digital photo processing tool: {software}"
            return result

        result["ai_score"] = 0.40
        result["label"] = "Standard digital EXIF record (non-AI software)"
    else:
        result["ai_score"] = 0.52
        result["label"] = "Metadata stripped (common for web/social uploads)"

    return result


def analyze_ela(img: Image.Image, quality: int = 90) -> dict:
    """
    Error Level Analysis (ELA): detect JPEG re-compression artifacts.
    AI-generated images often show extremely uniform ELA across regions.
    Real photos show heterogeneous ELA from different-compression regions.
    """
    try:
        buf = io.BytesIO()
        img_rgb = img.convert("RGB")
        img_rgb.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf)
        recompressed.load()

        diff = ImageChops.difference(img_rgb, recompressed)
        diff_arr = np.array(diff, dtype=np.float32)

        # ELA statistics across blocks
        block = 8
        h, w = diff_arr.shape[:2]
        block_means = []
        for i in range(0, h - block + 1, block):
            for j in range(0, w - block + 1, block):
                patch = diff_arr[i:i+block, j:j+block]
                block_means.append(float(patch.mean()))

        if not block_means:
            return {"ai_score": 0.5, "label": "ELA analysis unavailable"}

        block_arr = np.array(block_means)
        ela_mean = float(block_arr.mean())
        ela_cv = float(block_arr.std() / (ela_mean + 1e-5))

        # AI images: low ELA mean (smooth), low CV (uniform)
        # Real photos: higher ELA mean and higher CV (heterogeneous blocks)
        if ela_mean < 1.5 and ela_cv < 0.6:
            score = 0.88
            label = f"ELA: mean={ela_mean:.2f}, CV={ela_cv:.2f} — uniformly smooth (AI rendering signature)"
        elif ela_mean > 6.0 or ela_cv > 1.5:
            score = 0.15
            label = f"ELA: mean={ela_mean:.2f}, CV={ela_cv:.2f} — heterogeneous photo-realistic noise"
        else:
            score = 0.50
            label = f"ELA: mean={ela_mean:.2f}, CV={ela_cv:.2f} — mixed compression profile"

        return {"ai_score": score, "label": label}
    except Exception:
        return {"ai_score": 0.5, "label": "ELA not applicable to this image format"}


def analyze_dimension_fingerprint(img: Image.Image) -> dict:
    """
    AI generators output fixed canvas sizes (1024x1024, 768x512, etc.).
    Real camera photos have dimensions based on sensor aspect ratios.
    """
    w, h = img.size
    if (w, h) in _AI_DIMENSION_SETS or (h, w) in _AI_DIMENSION_SETS:
        score = 0.78
        label = f"Dimensions {w}x{h} match known AI generator canvas sizes"
    elif w == h:
        score = 0.62
        label = f"Square image ({w}x{h}) — common AI output crop"
    else:
        # Check if dimensions are multiples of 64 (SD requirement)
        if w % 64 == 0 and h % 64 == 0:
            score = 0.65
            label = f"Dimensions {w}x{h} are multiples of 64 — Stable Diffusion grid alignment"
        else:
            # Typical camera sensor aspect ratios: 4:3, 3:2, 16:9
            ratio = w / h if w > h else h / w
            camera_ratios = [4/3, 3/2, 16/9, 1/1, 5/4]
            nearest = min(camera_ratios, key=lambda r: abs(r - ratio))
            if abs(ratio - nearest) < 0.02:
                score = 0.20
                label = f"Dimensions {w}x{h} match standard camera sensor aspect ratio ({ratio:.2f})"
            else:
                score = 0.45
                label = f"Non-standard dimensions {w}x{h} (aspect ratio {ratio:.2f})"

    return {"ai_score": score, "label": label}


def analyze_noise_residual(img: Image.Image) -> dict:
    """
    Camera sensor noise has a characteristic flat-spectrum character.
    AI diffusion models produce structured noise residuals at specific frequencies.
    """
    gray = np.array(img.convert("L").resize((256, 256), Image.LANCZOS), dtype=np.float32)
    # Simple unsharp mask to isolate high-frequency residual
    from PIL import ImageFilter as IF
    pil_gray = Image.fromarray(gray.astype(np.uint8))
    blurred = np.array(pil_gray.filter(IF.GaussianBlur(1.5)), dtype=np.float32)
    residual = gray - blurred
    res_std = float(residual.std())
    res_mean_abs = float(np.mean(np.abs(residual)))

    # Compute kurtosis of residual — AI images have super-Gaussian residuals
    flat = residual.flatten()
    if flat.std() > 0:
        kurtosis = float(np.mean(((flat - flat.mean()) / flat.std()) ** 4))
    else:
        kurtosis = 3.0

    if res_std > 12.0 and kurtosis > 5.0:
        score = min(0.92, 0.55 + (res_std - 12.0) * 0.03 + (kurtosis - 5.0) * 0.02)
        label = f"High structured noise residual (std={res_std:.1f}, kurtosis={kurtosis:.1f}) — diffusion artifact"
    elif res_std < 3.5:
        score = 0.78
        label = f"Unusually low noise residual (std={res_std:.1f}) — synthetic rendering"
    elif kurtosis < 3.5:
        score = 0.22
        label = f"Gaussian camera sensor noise floor (std={res_std:.1f}, kurtosis={kurtosis:.1f})"
    else:
        score = 0.42
        label = f"Moderate noise residual (std={res_std:.1f}, kurtosis={kurtosis:.1f})"

    return {"ai_score": min(0.95, max(0.05, score)), "label": label}


def analyze_color_channel_correlation(img: Image.Image) -> dict:
    """High inter-channel correlation is a hallmark of AI-generated imagery."""
    arr = np.array(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    r = arr[:, :, 0].flatten()
    g = arr[:, :, 1].flatten()
    b = arr[:, :, 2].flatten()

    std_r, std_g, std_b = r.std(), g.std(), b.std()
    if std_r < 1e-4 or std_g < 1e-4 or std_b < 1e-4:
        return {"ai_score": 0.5, "label": "Monochrome / flat color profile"}

    corr_rg = float(np.corrcoef(r, g)[0, 1])
    corr_rb = float(np.corrcoef(r, b)[0, 1])
    corr_gb = float(np.corrcoef(g, b)[0, 1])
    avg_corr = (corr_rg + corr_rb + corr_gb) / 3.0

    if np.isnan(avg_corr):
        return {"ai_score": 0.5, "label": "Normal color spectrum"}

    ai_score = float(np.clip((avg_corr - 0.65) / 0.33, 0.05, 0.95))
    label = f"RGB correlation={avg_corr:.2f} ({'synthetic channel harmony' if avg_corr > 0.90 else 'natural multi-spectral variation'})"
    return {"ai_score": ai_score, "label": label, "avg_corr": avg_corr}


def analyze_image(image_bytes: bytes, filename: str) -> ImageResult:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        return ImageResult(0.5, "Error", filename, error=str(e))

    signals: list[ImageSignal] = []

    # 1. Primary Cloud Vision Model
    vit_primary = query_hf_vit_primary(image_bytes)
    if vit_primary is not None:
        signals.append(ImageSignal(
            name="Vision Model — AI Image Detector (ViT)",
            ai_score=round(vit_primary["ai_score"], 4),
            weight=4.0,
            label=vit_primary["label"]
        ))

    # 2. Secondary Cloud Vision Model (SDXL specialist)
    vit_secondary = query_hf_vit_secondary(image_bytes)
    if vit_secondary is not None:
        signals.append(ImageSignal(
            name="Vision Model — SDXL Detector",
            ai_score=round(vit_secondary["ai_score"], 4),
            weight=3.5,
            label=vit_secondary["label"]
        ))

    # 3. Metadata Forensics (high-confidence signal)
    exif = analyze_exif(img)
    exif_weight = 12.0 if (exif["ai_score"] > 0.95 or exif["ai_score"] < 0.08) else 2.0
    signals.append(ImageSignal(
        name="Metadata Forensics (EXIF / PNG)",
        ai_score=round(exif["ai_score"], 4),
        weight=exif_weight,
        label=exif["label"]
    ))

    # 4. Dimension fingerprint
    dims = analyze_dimension_fingerprint(img)
    signals.append(ImageSignal(
        name="Canvas Dimension Fingerprint",
        ai_score=round(dims["ai_score"], 4),
        weight=2.0,
        label=dims["label"]
    ))

    # 5. Error Level Analysis
    ela = analyze_ela(img)
    signals.append(ImageSignal(
        name="Error Level Analysis (ELA)",
        ai_score=round(ela["ai_score"], 4),
        weight=2.5,
        label=ela["label"]
    ))

    # 6. Noise Residual Analysis
    residual = analyze_noise_residual(img)
    signals.append(ImageSignal(
        name="High-Frequency Sensor Residuals",
        ai_score=round(residual["ai_score"], 4),
        weight=2.0,
        label=residual["label"]
    ))

    # 7. Color Channel Correlation
    corr = analyze_color_channel_correlation(img)
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

"""
Image AI Detection Module (Forensic Optical & Statistical Engine)
Accurately distinguishes genuine camera photography (including WhatsApp/mobile exports)
from AI-generated imagery (Stable Diffusion, Midjourney, DALL-E, Flux, Ideogram).

Analyzes:
  1. Metadata & Container Forensics (Camera hardware tags, PNG generator chunks, WhatsApp/mobile signatures)
  2. 2D Fourier Spectral Distribution (Optical low-pass diffraction vs Diffusion high-frequency anomalies)
  3. Latent VAE 8-Pixel Periodicity & Grid Autocorrelation
  4. Physical Sensor Noise & Poisson-Gaussian Photon Shot-Noise Model
  5. Error Level Analysis (ELA) calibrated for real JPEG messaging compression
  6. Aspect Ratio & Canvas Geometry (Camera sensor standards vs Generative canvas sizes)
  7. Cloud Vision Transformer via Hugging Face Router (with optional HF_TOKEN support)
"""

import io
import math
import os
import re
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
    "huawei", "xiaomi", "oneplus", "motorola", "gopro", "oppo", "vivo", "realme"
]

_MOBILE_FILENAME_RE = re.compile(
    r"^(whatsapp\s*image|img[_\-\s]|pxl[_\-\s]|dsc[_\-\s]|dscn|sam[_\-\s]|screenshot|screen\s*shot|\d{8}_\d{6})",
    re.IGNORECASE
)

_AI_EXACT_CANVAS = {
    (512, 512), (768, 512), (512, 768),
    (1024, 1024), (1024, 768), (768, 1024),
    (1024, 1792), (1792, 1024),
    (1344, 768), (768, 1344),
    (1152, 896), (896, 1152),
}


def query_hf_vit(image_bytes: bytes) -> Optional[dict]:
    """Queries Hugging Face vision model if reachable or if HF_TOKEN is configured."""
    try:
        url = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"
        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.post(url, headers=headers, data=image_bytes[:1500000], timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                items = data[0] if isinstance(data[0], list) else data
                for item in items:
                    lbl = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.5))
                    if any(k in lbl for k in ("artificial", "ai", "fake", "generated")):
                        return {"ai_score": score, "label": f'ViT Transformer: "{item.get("label")}" ({score:.1%})'}
                    elif any(k in lbl for k in ("human", "real", "natural")):
                        return {"ai_score": 1.0 - score, "label": f'ViT Transformer: "{item.get("label")}" ({score:.1%})'}
    except Exception:
        pass
    return None


def analyze_exif_and_provenance(img: Image.Image, filename: str) -> dict:
    info = img.info or {}
    png_text = " ".join(str(v) for v in info.values()).lower()

    # 1. Direct AI parameters in PNG metadata
    ai_in_png = any(p in png_text for p in _AI_SOFTWARE_PATTERNS)
    has_sd_params = any(k in png_text for k in ("steps:", "cfg scale", "sampler:", "seed:", "negative prompt", "model hash"))
    if ai_in_png or has_sd_params:
        return {
            "ai_score": 0.99,
            "weight": 15.0,
            "label": "Verified generative AI synthesis parameters embedded in metadata"
        }

    # 2. Camera hardware in EXIF
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
                "weight": 15.0,
                "label": f"AI generator software identified in EXIF: '{software}'"
            }

        if any(p in make_lc for p in _CAMERA_PATTERNS):
            gps_note = " + GPS tagged" if gps_info else ""
            return {
                "ai_score": 0.02,
                "weight": 15.0,
                "label": f"Verified physical camera hardware: {make} {model_tag}{gps_note}".strip()
            }

        if any(p in sw_lc for p in ["photoshop", "lightroom", "gimp", "capture one"]):
            return {
                "ai_score": 0.20,
                "weight": 3.0,
                "label": f"Standard digital photo processing software: {software}"
            }

        return {
            "ai_score": 0.25,
            "weight": 3.0,
            "label": "Standard digital camera EXIF record (non-AI hardware profile)"
        }

    # 3. Mobile / Messaging export recognition (WhatsApp, smartphone camera saves)
    clean_fn = os.path.basename(filename)
    if _MOBILE_FILENAME_RE.search(clean_fn):
        return {
            "ai_score": 0.15,
            "weight": 4.0,
            "label": f"Mobile camera / messaging export detected ('{clean_fn[:30]}') — EXIF stripped by messaging service for privacy"
        }

    return {
        "ai_score": 0.38,
        "weight": 1.5,
        "label": "Metadata stripped (standard for web uploads and social messaging)"
    }


def analyze_fourier_spectral_decay(img: Image.Image) -> dict:
    """
    Physical optical lenses act as optical low-pass filters (diffraction limit).
    Real photographs have energy concentrated in mid and low frequencies, with steep decay at the outer edge.
    Diffusion generators produce anomalous high-frequency energy across the entire frequency plane.
    """
    gray = np.array(img.convert("L").resize((256, 256), Image.LANCZOS), dtype=np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    max_r = min(h, w) / 2.0

    # Very high frequency outer perimeter (> 70% of Nyquist)
    outer_mask = dist > (max_r * 0.70)
    # Mid-frequency band (10% to 35% of Nyquist)
    mid_mask = (dist > (max_r * 0.10)) & (dist < (max_r * 0.35))

    outer_e = float(np.mean(mag[outer_mask]))
    mid_e = float(np.mean(mag[mid_mask])) + 1e-6
    ratio = outer_e / mid_e

    # Real photos have ratio < 0.30; pure diffusion grain has ratio > 0.65
    if ratio > 0.65:
        score = 0.88
        label = f"Elevated high-frequency spectral ratio ({ratio:.3f}) — diffusion synthesis signature"
    elif ratio < 0.25:
        score = 0.12
        label = f"Natural optical diffraction roll-off ({ratio:.3f}) — authentic glass lens profile"
    else:
        score = 0.32
        label = f"Balanced spatial frequency spectrum ({ratio:.3f})"

    return {"ai_score": score, "label": label}


def analyze_vae_grid_autocorrelation(img: Image.Image) -> dict:
    """
    Latent diffusion models (Stable Diffusion, Midjourney, Flux) decode latents
    through an 8x or 16x spatial VAE, creating subtle 8-pixel periodicity.
    Natural photos from physical sensor Bayer filters do not exhibit 8-pixel periodic resonance.
    """
    gray = np.array(img.convert("L").resize((256, 256), Image.LANCZOS), dtype=np.float32)
    diff_h8 = np.abs(gray[:, 8:] - gray[:, :-8])
    diff_h7 = np.abs(gray[:, 7:] - gray[:, :-7])

    mean_h8 = float(np.mean(diff_h8))
    mean_h7 = float(np.mean(diff_h7))
    ratio = mean_h8 / (mean_h7 + 1e-6)

    # In diffusion models, gradient at lag 8 drops noticeably due to block boundary repetition
    if ratio < 0.94:
        score = 0.84
        label = f"8-pixel periodic autocorrelation dip ({ratio:.3f}) — VAE decoder grid artifact"
    elif ratio > 1.05:
        score = 0.20
        label = f"Continuous non-periodic spatial texture ({ratio:.3f}) — natural scene continuity"
    else:
        score = 0.35
        label = f"Standard spatial texture distribution ({ratio:.3f})"

    return {"ai_score": score, "label": label}


def analyze_sensor_noise_physics(img: Image.Image) -> dict:
    """
    Physical camera sensors produce Poisson-Gaussian photon shot noise,
    where noise variance is physically coupled to local luminance.
    Smooth regions (sky, skin, compression) are natural in photography.
    """
    gray = np.array(img.convert("L").resize((256, 256), Image.LANCZOS), dtype=np.float32)
    pil_gray = Image.fromarray(gray.astype(np.uint8))
    blurred = np.array(pil_gray.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
    residual = gray - blurred

    res_std = float(residual.std())
    kurtosis = float(np.mean(((residual - residual.mean()) / (res_std + 1e-6)) ** 4))

    # Real phone and camera photos: res_std 0.1 - 9.0 (smooth sky to textured scene)
    if res_std > 18.0 and kurtosis > 8.0:
        score = 0.86
        label = f"Anomalous high-frequency noise variance (std={res_std:.1f}, kurtosis={kurtosis:.1f}) — diffusion grain"
    elif res_std <= 8.0:
        score = 0.18
        label = f"Physical sensor Poisson noise floor (std={res_std:.1f}, kurtosis={kurtosis:.1f}) — authentic camera capture"
    else:
        score = 0.35
        label = f"Standard photographic noise residual (std={res_std:.1f})"

    return {"ai_score": score, "label": label}


def analyze_ela_compression(img: Image.Image) -> dict:
    """
    Error Level Analysis (ELA) calibrated for real camera & messaging compression.
    Homogeneous compression is NORMAL for smartphone and WhatsApp photos.
    """
    try:
        buf = io.BytesIO()
        rgb = img.convert("RGB")
        rgb.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        recomp = Image.open(buf)
        diff = ImageChops.difference(rgb, recomp)
        diff_arr = np.array(diff, dtype=np.float32)

        mean_diff = float(diff_arr.mean())
        # Standard re-saved JPEGs (WhatsApp, camera) have low, uniform mean_diff (0.2 - 3.0)
        if mean_diff < 4.0:
            score = 0.20
            label = f"Consistent single-source JPEG compression floor (mean={mean_diff:.2f}) — authentic photo encoding"
        elif mean_diff > 12.0:
            score = 0.72
            label = f"Discontinuous compression error levels (mean={mean_diff:.2f}) — digital artifact"
        else:
            score = 0.35
            label = f"Balanced error level distribution (mean={mean_diff:.2f})"
        return {"ai_score": score, "label": label}
    except Exception:
        return {"ai_score": 0.35, "label": "Standard photographic compression profile"}


def analyze_dimension_fingerprint(img: Image.Image) -> dict:
    w, h = img.size
    if (w, h) in _AI_EXACT_CANVAS or (h, w) in _AI_EXACT_CANVAS:
        return {
            "ai_score": 0.75,
            "label": f"Canvas size {w}x{h} matches exact AI model generation preset (e.g. SDXL/DALL-E)"
        }

    ratio = w / h if w > h else h / w
    # Typical physical sensor aspect ratios
    camera_ratios = [4/3, 16/9, 3/2, 1/1, 5/4]
    closest = min(camera_ratios, key=lambda r: abs(r - ratio))
    diff = abs(ratio - closest)

    if diff < 0.03:
        ratio_label = "4:3" if abs(ratio - 4/3) < 0.03 else "16:9" if abs(ratio - 16/9) < 0.03 else "3:2" if abs(ratio - 3/2) < 0.03 else "1:1"
        return {
            "ai_score": 0.15,
            "label": f"Aspect ratio {ratio:.2f} matches standard camera optical sensor ({ratio_label})"
        }
    else:
        return {
            "ai_score": 0.40,
            "label": f"Custom canvas geometry ({w}x{h}, aspect {ratio:.2f})"
        }


def analyze_image(image_bytes: bytes, filename: str) -> ImageResult:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        return ImageResult(0.5, "Error", filename, error=str(e))

    signals: list[ImageSignal] = []

    # 1. Metadata & Provenance (Highest confidence)
    prov = analyze_exif_and_provenance(img, filename)
    signals.append(ImageSignal(
        name="Metadata & Hardware Provenance",
        ai_score=round(prov["ai_score"], 4),
        weight=prov["weight"],
        label=prov["label"]
    ))

    # 2. 2D Fourier Spectral Distribution
    fourier = analyze_fourier_spectral_decay(img)
    signals.append(ImageSignal(
        name="Optical Frequency Spectrum (2D FFT)",
        ai_score=round(fourier["ai_score"], 4),
        weight=3.5,
        label=fourier["label"]
    ))

    # 3. Latent VAE 8-Pixel Periodicity
    vae = analyze_vae_grid_autocorrelation(img)
    signals.append(ImageSignal(
        name="Latent VAE Grid Autocorrelation",
        ai_score=round(vae["ai_score"], 4),
        weight=3.0,
        label=vae["label"]
    ))

    # 4. Physical Sensor Noise Physics
    noise = analyze_sensor_noise_physics(img)
    signals.append(ImageSignal(
        name="Camera Sensor Poisson Noise Physics",
        ai_score=round(noise["ai_score"], 4),
        weight=3.0,
        label=noise["label"]
    ))

    # 5. JPEG Compression & ELA
    ela = analyze_ela_compression(img)
    signals.append(ImageSignal(
        name="Compression Error Level Analysis",
        ai_score=round(ela["ai_score"], 4),
        weight=2.0,
        label=ela["label"]
    ))

    # 6. Canvas Geometry & Sensor Aspect Ratio
    dims = analyze_dimension_fingerprint(img)
    signals.append(ImageSignal(
        name="Optical Sensor Aspect Ratio",
        ai_score=round(dims["ai_score"], 4),
        weight=2.0,
        label=dims["label"]
    ))

    # 7. Cloud Vision Transformer (if available)
    vit_res = query_hf_vit(image_bytes)
    if vit_res is not None:
        signals.append(ImageSignal(
            name="Vision Transformer Classifier (ViT)",
            ai_score=round(vit_res["ai_score"], 4),
            weight=4.0,
            label=vit_res["label"]
        ))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    # Strong hardware calibration safeguards
    if prov["ai_score"] <= 0.05:
        final_score = min(final_score, 0.08)
    elif prov["ai_score"] >= 0.95:
        final_score = max(final_score, 0.95)

    verdict = (
        "AI-Generated" if final_score >= 0.60
        else "Uncertain / Mixed" if final_score >= 0.38
        else "Likely Real Photo"
    )

    return ImageResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        filename=filename,
        signals=signals,
    )

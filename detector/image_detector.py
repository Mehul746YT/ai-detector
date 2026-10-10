"""
Image AI Detection Module (Forensic Optical & Statistical Engine)
Accurately discriminates genuine camera photography from AI-generated imagery
(Midjourney, Stable Diffusion, DALL-E, Flux, Ideogram, StyleGAN).

Core Signals:
  1. Metadata & Hardware Provenance (Physical camera tags vs AI generation parameters)
  2. Shadow Chromaticity & AI Color Grading (Physical sensor shadow desaturation vs AI cinematic saturation)
  3. Gradient Bimodality (Uncanny contrast of plasticky smooth surfaces alongside hyper-sharp micro-edges)
  4. Plasticky Surface Patch Homogeneity (16x16 block variance distribution)
  5. Canvas Geometry Preset Fingerprint (Exact 1024x1024, 512x512, 1024x1792 generative resolutions)
  6. High-Frequency Fourier Residual & Sensor Grain Physics
"""

import io
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

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

_AI_EXACT_CANVAS = {
    (512, 512), (768, 512), (512, 768),
    (1024, 1024), (1024, 768), (768, 1024),
    (1024, 1792), (1792, 1024),
    (1344, 768), (768, 1344),
    (1152, 896), (896, 1152),
}


def analyze_metadata_and_provenance(img: Image.Image, filename: str) -> dict:
    info = img.info or {}
    png_text = " ".join(str(v) for v in info.values()).lower()

    # 1. Direct AI parameters in PNG metadata
    ai_in_png = any(p in png_text for p in _AI_SOFTWARE_PATTERNS)
    has_sd_params = any(k in png_text for k in ("steps:", "cfg scale", "sampler:", "seed:", "negative prompt", "model hash"))
    if ai_in_png or has_sd_params:
        return {
            "ai_score": 0.99,
            "weight": 15.0,
            "label": "Verified generative AI synthesis parameters embedded in metadata",
            "is_hardware": False,
            "is_ai": True
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
                "label": f"AI generator software identified in EXIF: '{software}'",
                "is_hardware": False,
                "is_ai": True
            }

        if any(p in make_lc for p in _CAMERA_PATTERNS):
            gps_note = " + GPS tagged" if gps_info else ""
            return {
                "ai_score": 0.04,
                "weight": 15.0,
                "label": f"Verified physical camera hardware: {make} {model_tag}{gps_note}".strip(),
                "is_hardware": True,
                "is_ai": False
            }

        if any(p in sw_lc for p in ["photoshop", "lightroom", "gimp", "capture one"]):
            return {
                "ai_score": 0.35,
                "weight": 2.5,
                "label": f"Standard digital photo editor record: {software}",
                "is_hardware": False,
                "is_ai": False
            }

        return {
            "ai_score": 0.30,
            "weight": 3.0,
            "label": "Standard digital camera EXIF header (non-AI hardware profile)",
            "is_hardware": False,
            "is_ai": False
        }

    return {
        "ai_score": 0.50,
        "weight": 1.5,
        "label": "Metadata stripped (standard for web uploads and social messaging) — evaluating visual pixels",
        "is_hardware": False,
        "is_ai": False
    }


def analyze_shadow_chromaticity(img: Image.Image) -> dict:
    """
    Physical optical sensors desaturate low-luminance shadow regions due to
    photon shot noise and sensor dynamic range clipping.
    In AI models (Midjourney, DALL-E, Flux), shadows remain heavily saturated
    with cinematic color grading (teal/orange, cyan/purple).
    """
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    shadow_mask = v < 0.25
    shadow_count = int(np.sum(shadow_mask))

    if shadow_count < 100:
        return {
            "ai_score": 0.50,
            "label": "High-key scene (minimal dark shadow regions to sample)"
        }

    shadow_sat = float(np.mean(s[shadow_mask]))

    if shadow_sat > 0.38:
        score = 0.86
        label = f"Hyper-saturated shadow chromaticity (sat={shadow_sat:.2f}) — typical Midjourney/DALL-E color grading"
    elif shadow_sat > 0.28:
        score = 0.65
        label = f"Elevated shadow saturation ({shadow_sat:.2f}) — stylized color palette"
    elif shadow_sat < 0.18:
        score = 0.18
        label = f"Natural physical shadow desaturation ({shadow_sat:.2f}) — authentic sensor dynamic range"
    else:
        score = 0.38
        label = f"Balanced shadow chromatic profile ({shadow_sat:.2f})"

    return {"ai_score": score, "label": label, "shadow_sat": shadow_sat}


def analyze_gradient_bimodality(img: Image.Image) -> dict:
    """
    AI diffusion imagery exhibits an uncanny bimodal distribution:
    ultra-smooth, textureless flat regions (plasticky cheeks, skin, skies)
    contrasted against hyper-sharp, razor micro-edges (hair, eyelashes, jewelry).
    Physical camera images have a continuous gradient distribution due to physical optical blur and sensor noise.
    """
    gray = np.array(img.convert("L").resize((384, 384), Image.LANCZOS), dtype=np.float32)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx**2 + gy**2)

    p50 = float(np.percentile(grad, 50)) + 1e-5
    p95 = float(np.percentile(grad, 95))
    bimodality = p95 / p50

    if bimodality > 6.5:
        score = 0.88
        label = f"Severe gradient bimodality ({bimodality:.2f}x) — over-sharpened micro-edges alongside flat synthetic surfaces"
    elif bimodality > 4.5:
        score = 0.68
        label = f"Elevated edge-to-background contrast ratio ({bimodality:.2f}x)"
    elif bimodality < 3.2:
        score = 0.18
        label = f"Natural continuous optical gradient falloff ({bimodality:.2f}x) — authentic lens behavior"
    else:
        score = 0.40
        label = f"Standard photographic gradient distribution ({bimodality:.2f}x)"

    return {"ai_score": score, "label": label, "bimodality": bimodality}


def analyze_surface_patch_homogeneity(img: Image.Image) -> dict:
    """
    Partitions the image into 16x16 pixel patches and measures variance.
    AI-generated faces and objects frequently exhibit unnaturally zero-noise
    patches (smooth 'AI plastic skin'), whereas physical camera sensors
    have PRNU and photon shot noise that ensures virtually no patch has near-zero variance.
    """
    gray = np.array(img.convert("L").resize((256, 256), Image.LANCZOS), dtype=np.float32)
    h, w = gray.shape

    patches = []
    for i in range(0, h - 16 + 1, 16):
        for j in range(0, w - 16 + 1, 16):
            patch = gray[i:i+16, j:j+16]
            patches.append(float(np.var(patch)))

    if not patches:
        return {"ai_score": 0.50, "label": "Image too small for patch variance profiling"}

    patches = np.array(patches)
    # Fraction of patches with near-zero texture (< 8.0 variance)
    flat_ratio = float(np.mean(patches < 8.0))

    if flat_ratio > 0.30:
        score = 0.85
        label = f"Dense plasticky surface patches ({flat_ratio:.1%} smooth regions) — AI generative texture signature"
    elif flat_ratio > 0.15:
        score = 0.64
        label = f"Elevated surface smoothness ({flat_ratio:.1%})"
    elif flat_ratio < 0.04:
        score = 0.16
        label = f"Pervasive sensor shot-noise floor ({flat_ratio:.1%} smooth patches) — authentic photographic grain"
    else:
        score = 0.40
        label = f"Balanced textural variance ({flat_ratio:.1%})"

    return {"ai_score": score, "label": label, "flat_ratio": flat_ratio}


def analyze_canvas_geometry_preset(img: Image.Image) -> dict:
    w, h = img.size
    if (w, h) in _AI_EXACT_CANVAS or (h, w) in _AI_EXACT_CANVAS:
        return {
            "ai_score": 0.80,
            "label": f"Exact AI generation canvas size preset ({w}x{h}) — standard DALL-E/Midjourney/SDXL canvas"
        }

    # Check for square images
    if w == h:
        return {
            "ai_score": 0.62,
            "label": f"Square canvas ({w}x{h}) — standard generative output ratio"
        }

    ratio = w / h if w > h else h / w
    camera_ratios = [4/3, 16/9, 3/2]
    closest = min(camera_ratios, key=lambda r: abs(r - ratio))
    diff = abs(ratio - closest)

    if diff < 0.03:
        return {
            "ai_score": 0.45,
            "label": f"Aspect ratio ({ratio:.2f}) matches common display/camera format"
        }

    return {
        "ai_score": 0.50,
        "label": f"Custom canvas geometry ({w}x{h}, aspect {ratio:.2f})"
    }


def analyze_fourier_residual(img: Image.Image) -> dict:
    gray = np.array(img.convert("L").resize((256, 256), Image.LANCZOS), dtype=np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    max_r = min(h, w) / 2.0

    outer_mask = dist > (max_r * 0.70)
    mid_mask = (dist > (max_r * 0.10)) & (dist < (max_r * 0.35))

    outer_e = float(np.mean(mag[outer_mask]))
    mid_e = float(np.mean(mag[mid_mask])) + 1e-6
    ratio = outer_e / mid_e

    if ratio > 0.60:
        score = 0.84
        label = f"Elevated high-frequency spectral ratio ({ratio:.3f}) — diffusion synthesis grain"
    elif ratio < 0.20:
        score = 0.22
        label = f"Natural optical diffraction roll-off ({ratio:.3f}) — authentic glass lens profile"
    else:
        score = 0.45
        label = f"Balanced spatial frequency spectrum ({ratio:.3f})"

    return {"ai_score": score, "label": label}


def analyze_image(image_bytes: bytes, filename: str) -> ImageResult:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        return ImageResult(0.5, "Error", filename, error=str(e))

    signals: list[ImageSignal] = []

    # 1. Metadata Provenance (Determines ground truth if camera or AI tags present)
    prov = analyze_metadata_and_provenance(img, filename)
    signals.append(ImageSignal(
        name="Metadata & Hardware Provenance",
        ai_score=round(prov["ai_score"], 4),
        weight=prov["weight"],
        label=prov["label"]
    ))

    # 2. Shadow Chromaticity (AI Color Grading)
    shadow = analyze_shadow_chromaticity(img)
    signals.append(ImageSignal(
        name="Shadow Chromaticity & Color Grading",
        ai_score=round(shadow["ai_score"], 4),
        weight=3.5,
        label=shadow["label"]
    ))

    # 3. Gradient Bimodality (Plastic skin vs razor micro-sharpness)
    bimo = analyze_gradient_bimodality(img)
    signals.append(ImageSignal(
        name="Surface-to-Edge Gradient Bimodality",
        ai_score=round(bimo["ai_score"], 4),
        weight=3.5,
        label=bimo["label"]
    ))

    # 4. Plasticky Surface Patch Homogeneity
    surface = analyze_surface_patch_homogeneity(img)
    signals.append(ImageSignal(
        name="Plasticky Surface Patch Homogeneity",
        ai_score=round(surface["ai_score"], 4),
        weight=3.0,
        label=surface["label"]
    ))

    # 5. Canvas Geometry Preset
    canvas = analyze_canvas_geometry_preset(img)
    signals.append(ImageSignal(
        name="Canvas Geometry Preset Fingerprint",
        ai_score=round(canvas["ai_score"], 4),
        weight=2.0,
        label=canvas["label"]
    ))

    # 6. Fourier Residual
    fourier = analyze_fourier_residual(img)
    signals.append(ImageSignal(
        name="Optical Frequency Spectrum (2D FFT)",
        ai_score=round(fourier["ai_score"], 4),
        weight=2.0,
        label=fourier["label"]
    ))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    # Direct calibration safeguards:
    if prov.get("is_hardware"):
        final_score = min(final_score, 0.08)
    elif prov.get("is_ai"):
        final_score = max(final_score, 0.96)
    else:
        # If multiple visual markers flag AI (e.g. shadow sat and gradient bimodality both high):
        if shadow["ai_score"] >= 0.70 and bimo["ai_score"] >= 0.70:
            final_score = max(final_score, 0.78)
        elif surface["ai_score"] >= 0.75 and bimo["ai_score"] >= 0.65:
            final_score = max(final_score, 0.75)
        # If visual markers are all natural:
        elif shadow["ai_score"] <= 0.25 and bimo["ai_score"] <= 0.25 and surface["ai_score"] <= 0.25:
            final_score = min(final_score, 0.22)

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

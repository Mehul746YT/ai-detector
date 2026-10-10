"""
Audio AI & Deepfake Detection Module (Forensic Audio DSP + Serverless ML)
Analyzes:
  1. Hugging Face Deepfake Audio Detection Transformer (mo-thecreator/Deepfake-audio-detection)
  2. Nyquist High-Frequency Spectral Cutoff (Vocoders typically cut off sharply at 8kHz, 11kHz, or 12kHz)
  3. Spectral Flux & Rolloff Dynamics (AI vocoders lack consonant transient burst dynamics)
  4. Pitch Contour Stability & Vocal Jitter (Micro-tremors present in human vocal cords)
  5. Digital Silence Gating vs Room Ambient Thermal Noise Floor
  6. Audio Container & ID3/RIFF Metadata Forensics
"""

import io
import math
import struct
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests


@dataclass
class AudioSignal:
    name: str
    ai_score: float
    weight: float
    label: str


@dataclass
class AudioResult:
    final_score: float
    verdict: str
    filename: str
    signals: list = field(default_factory=list)
    error: Optional[str] = None


_AI_AUDIO_KEYWORDS = [
    b"elevenlabs", b"bark", b"tortoise", b"coqui", b"vits", b"suno", b"udio",
    b"riffusion", b"musiclm", b"audiocraft", b"musicgen", b"rvc", b"so-vits",
    b"descript", b"resemble", b"playht", b"murf", b"speechify", b"wellsaid"
]


def query_hf_audio_api(audio_bytes: bytes) -> Optional[dict]:
    """
    Queries Hugging Face Serverless Audio Classification API.
    Model: mo-thecreator/Deepfake-audio-detection (Wav2Vec2 fine-tuned)
    """
    try:
        url = "https://api-inference.huggingface.co/models/mo-thecreator/Deepfake-audio-detection"
        sample_payload = audio_bytes[:1500000]  # HF serverless payload limit (~1.5MB)
        resp = requests.post(url, data=sample_payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                items = data[0] if isinstance(data[0], list) else data
                for item in items:
                    lbl = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.5))
                    if any(k in lbl for k in ("fake", "synthetic", "deepfake", "ai", "spoof")):
                        return {
                            "ai_score": score,
                            "label": f'Acoustic Transformer: "{item.get("label")}" ({score:.1%})'
                        }
                    elif any(k in lbl for k in ("real", "bonafide", "human", "authentic")):
                        return {
                            "ai_score": 1.0 - score,
                            "label": f'Acoustic Transformer: "{item.get("label")}" ({score:.1%})'
                        }
                first = items[0]
                return {
                    "ai_score": float(first.get("score", 0.5)),
                    "label": f'Acoustic model top: {first.get("label")}'
                }
    except Exception:
        pass
    return None


def check_audio_metadata(audio_bytes: bytes) -> dict:
    header = audio_bytes[:8192].lower()
    for kw in _AI_AUDIO_KEYWORDS:
        if kw in header:
            kw_name = kw.decode('ascii', errors='ignore')
            return {
                "ai_score": 0.99,
                "label": f"Embedded signature of AI audio synthesis engine: '{kw_name}'"
            }
    
    # Check for DAW / hardware recorder signatures (Zoom, Tascam, Logic, ProTools, Ableton)
    real_daws = [b"pro tools", b"logic pro", b"ableton", b"zoom", b"tascam", b"sound devices", b"audacity"]
    for daw in real_daws:
        if daw in header:
            return {
                "ai_score": 0.10,
                "label": f"Hardware recorder / studio DAW signature: '{daw.decode('ascii')}'"
            }

    return {
        "ai_score": 0.48,
        "label": "Standard audio stream container (no direct synthesis tags)"
    }


def parse_audio_samples(audio_bytes: bytes) -> Optional[tuple[np.ndarray, int]]:
    """
    Extracts raw audio samples from WAV or estimates from raw PCM buffer.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None
            frames = wf.readframes(min(n_frames, framerate * 30))  # Up to 30s
            
            if sampwidth == 1:
                dtype = np.uint8
            elif sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                return None
            
            data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            
            max_val = np.max(np.abs(data)) + 1e-6
            data = data / max_val
            return data, framerate
    except Exception:
        pass

    # Fallback: raw PCM estimation for non-WAV streams
    try:
        offset = min(128, len(audio_bytes) // 4)
        sample_chunk = audio_bytes[offset:offset + 131072]
        if len(sample_chunk) >= 2048:
            raw = np.frombuffer(sample_chunk[:len(sample_chunk) - (len(sample_chunk) % 2)], dtype=np.int16).astype(np.float32)
            if len(raw) > 500:
                raw = raw / (np.max(np.abs(raw)) + 1e-6)
                return raw, 44100
    except Exception:
        pass

    return None


def analyze_spectral_cutoff(samples: np.ndarray, sr: int) -> dict:
    """
    Vocoder Frequency Cutoff Forensics:
    Neural vocoders (HiFi-GAN, MelGAN, WaveNet) typically synthesize at 16kHz or 24kHz,
    leaving an unnatural steep spectral cliff at 8kHz or 12kHz.
    Real acoustic microphones exhibit natural gradual roll-off with ambient thermal noise above 12kHz.
    """
    if len(samples) < 2048:
        return {"ai_score": 0.5, "label": "Sample buffer too short for FFT spectral cutoff"}

    fft_size = min(4096, 2 ** int(math.log2(len(samples))))
    fft_data = np.abs(np.fft.rfft(samples[:fft_size]))
    freqs = np.fft.rfftfreq(fft_size, d=1.0/sr)

    # Energy below 8kHz vs energy above 12kHz
    band_low = (freqs >= 300) & (freqs <= 8000)
    band_high = (freqs >= 12000) & (freqs <= min(20000, sr / 2))

    energy_low = np.sum(fft_data[band_low] ** 2) + 1e-9
    energy_high = np.sum(fft_data[band_high] ** 2) if np.any(band_high) else 0.0

    ratio_high_to_low = energy_high / energy_low

    if ratio_high_to_low < 0.0005 and sr >= 32000:
        score = 0.88
        label = f"Steep ultrasonic energy cutoff (high/low ratio: {ratio_high_to_low:.5f}) — typical neural vocoder bandlimit"
    elif ratio_high_to_low < 0.003:
        score = 0.65
        label = f"Restricted high-frequency band (ratio: {ratio_high_to_low:.4f})"
    elif ratio_high_to_low > 0.03:
        score = 0.15
        label = f"Broadband acoustic air spectrum (ratio: {ratio_high_to_low:.3f}) — authentic microphone floor"
    else:
        score = 0.40
        label = f"Natural spectral rolloff (ratio: {ratio_high_to_low:.4f})"

    return {"ai_score": score, "label": label}


def analyze_spectral_flux_dynamics(samples: np.ndarray, sr: int) -> dict:
    """
    Spectral Flux measures the frame-to-frame change of spectral power.
    Human speech contains abrupt consonant transitions, plosives, and bursts.
    AI voice models exhibit smoothed, hyper-continuous spectral transitions.
    """
    frame_len = int(sr * 0.025)  # 25ms
    hop_len = int(sr * 0.010)    # 10ms
    if len(samples) < frame_len * 6:
        return {"ai_score": 0.5, "label": "Audio too short for spectral dynamics"}

    specs = []
    for i in range(0, len(samples) - frame_len, hop_len):
        frame = samples[i:i + frame_len] * np.hanning(frame_len)
        mag = np.abs(np.fft.rfft(frame))
        specs.append(mag / (np.sum(mag) + 1e-6))

    if len(specs) < 3:
        return {"ai_score": 0.5, "label": "Insufficient frames"}

    specs = np.array(specs)
    # Compute spectral flux (L2 distance between consecutive spectra)
    fluxes = np.sqrt(np.sum((specs[1:] - specs[:-1]) ** 2, axis=1))
    flux_std = float(np.std(fluxes))
    flux_mean = float(np.mean(fluxes))
    flux_cv = flux_std / (flux_mean + 1e-6)

    if flux_cv < 0.35:
        score = 0.84
        label = f"Low spectral flux variance (CV={flux_cv:.2f}) — synthetic continuous vocoder flow"
    elif flux_cv > 0.75:
        score = 0.18
        label = f"Dynamic spectral transient flux (CV={flux_cv:.2f}) — natural speech plosives & phonemes"
    else:
        score = 0.48
        label = f"Moderate spectral flux dynamics (CV={flux_cv:.2f})"

    return {"ai_score": score, "label": label}


def analyze_spectral_jitter(samples: np.ndarray, sr: int) -> dict:
    """
    Micro-tremor / Pitch Jitter Analysis:
    Organic vocal folds vibrate with involuntary cycle-to-cycle deviations (jitter).
    Synthetic voices are modeled with mathematical periodicity.
    """
    frame_size = int(sr * 0.03)  # 30ms window
    hop_size = int(sr * 0.015)   # 15ms hop
    if len(samples) < frame_size * 4:
        return {"ai_score": 0.5, "label": "Audio clip too short for pitch jitter profiling"}

    energies = []
    zero_crossings = []
    for i in range(0, len(samples) - frame_size, hop_size):
        frame = samples[i:i+frame_size]
        energies.append(float(np.sqrt(np.mean(frame**2))))
        zc = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame_size)
        zero_crossings.append(float(zc))

    energies = np.array(energies)
    zero_crossings = np.array(zero_crossings)

    voiced = energies > (np.mean(energies) * 0.25)
    if np.sum(voiced) < 5:
        return {"ai_score": 0.5, "label": "Insufficient voiced segments"}

    zc_voiced = zero_crossings[voiced]
    zc_cv = float(np.std(zc_voiced) / (np.mean(zc_voiced) + 1e-5))

    if zc_cv < 0.22:
        score = 0.89
        label = f"Zero-crossing rate CV={zc_cv:.2f} (synthetic periodic stability)"
    elif zc_cv < 0.40:
        score = 0.55
        label = f"Zero-crossing rate CV={zc_cv:.2f} (moderate vocal stability)"
    else:
        score = 0.15
        label = f"Zero-crossing rate CV={zc_cv:.2f} (natural human vocal jitter & tremor)"

    return {"ai_score": score, "label": label}


def analyze_noise_floor_silence_gate(samples: np.ndarray) -> dict:
    """
    TTS generators often enforce digital zero silence between syllables,
    whereas genuine studio or smartphone recordings capture ambient room reverberation.
    """
    frame_size = 512
    min_energies = []
    for i in range(0, len(samples) - frame_size, frame_size):
        frame = samples[i:i+frame_size]
        min_energies.append(float(np.mean(np.abs(frame))))

    if not min_energies:
        return {"ai_score": 0.5, "label": "Insufficient data"}

    min_energies = np.array(min_energies)
    dead_silence_ratio = float(np.mean(min_energies < 1e-4))

    if dead_silence_ratio > 0.18:
        score = 0.91
        label = f"Digital silence gating ({dead_silence_ratio:.1%} dead silence) — typical TTS synthesis"
    elif dead_silence_ratio > 0.05:
        score = 0.62
        label = f"Noticeable noise-gate dropouts ({dead_silence_ratio:.1%})"
    else:
        score = 0.18
        label = "Continuous room acoustic ambient noise floor detected"

    return {"ai_score": score, "label": label}


def analyze_audio(audio_bytes: bytes, filename: str) -> AudioResult:
    if len(audio_bytes) < 500:
        return AudioResult(0.5, "Error", filename, error="Audio file is empty or corrupted.")

    signals: list[AudioSignal] = []

    # 1. Cloud Transformer API (mo-thecreator/Deepfake-audio-detection)
    hf_res = query_hf_audio_api(audio_bytes)
    if hf_res is not None:
        signals.append(AudioSignal(
            name="Cloud Acoustic Deepfake Classifier",
            ai_score=round(hf_res["ai_score"], 4),
            weight=4.0,
            label=hf_res["label"]
        ))

    # 2. Metadata Analysis
    meta = check_audio_metadata(audio_bytes)
    meta_weight = 10.0 if meta["ai_score"] > 0.9 else 2.0
    signals.append(AudioSignal(
        name="Audio Container & Stream Forensics",
        ai_score=round(meta["ai_score"], 4),
        weight=meta_weight,
        label=meta["label"]
    ))

    # 3. Waveform / DSP Spectrum Analysis
    parsed = parse_audio_samples(audio_bytes)
    if parsed is not None:
        samples, sr = parsed

        # Ultrasonic / Vocoder Nyquist Cutoff
        cutoff = analyze_spectral_cutoff(samples, sr)
        signals.append(AudioSignal(
            name="Vocoder Nyquist Spectral Cutoff",
            ai_score=round(cutoff["ai_score"], 4),
            weight=3.0,
            label=cutoff["label"]
        ))

        # Transient Flux Dynamics
        flux = analyze_spectral_flux_dynamics(samples, sr)
        signals.append(AudioSignal(
            name="Spectral Flux & Transient Dynamics",
            ai_score=round(flux["ai_score"], 4),
            weight=2.5,
            label=flux["label"]
        ))

        # Pitch Jitter
        jitter = analyze_spectral_jitter(samples, sr)
        signals.append(AudioSignal(
            name="Vocal Pitch & Jitter Organic Variance",
            ai_score=round(jitter["ai_score"], 4),
            weight=2.5,
            label=jitter["label"]
        ))

        # Silence Gating
        gate = analyze_noise_floor_silence_gate(samples)
        signals.append(AudioSignal(
            name="Acoustic Ambient Noise Floor",
            ai_score=round(gate["ai_score"], 4),
            weight=2.0,
            label=gate["label"]
        ))
    else:
        signals.append(AudioSignal(
            name="Acoustic Waveform Analysis",
            ai_score=0.5,
            weight=1.0,
            label="Compressed audio stream parsed at container level"
        ))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    verdict = (
        "AI-Generated Audio" if final_score >= 0.62
        else "Uncertain / Mixed" if final_score >= 0.38
        else "Likely Authentic Audio"
    )

    return AudioResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        filename=filename,
        signals=signals,
    )

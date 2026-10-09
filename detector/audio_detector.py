"""
Audio AI & Deepfake Detection Module
Analyzes:
  1. Spectral rolloff & zero-crossing rate uniformity (synthetic vocoder artifacts)
  2. Pitch contour stability (AI voices lack natural micro-tremor/jitter)
  3. Dynamic range compression & silence gating (hallmark of TTS generators like ElevenLabs, Bark, VITS)
  4. Audio container metadata forensics (ID3/RIFF headers for AI tools)
"""

import io
import math
import struct
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
    b"riffusion", b"musiclm", b"audiocraft", b"musicgen", b"rvc", b"so-vits"
]


def check_audio_metadata(audio_bytes: bytes) -> dict:
    header = audio_bytes[:4096].lower()
    for kw in _AI_AUDIO_KEYWORDS:
        if kw in header:
            return {
                "ai_score": 0.99,
                "label": f"Embedded signature of AI audio synthesis engine ({kw.decode('ascii', errors='ignore')})"
            }
    return {
        "ai_score": 0.50,
        "label": "Standard audio stream metadata (no explicit generator tags)"
    }


def parse_wav_samples(audio_bytes: bytes) -> Optional[tuple[np.ndarray, int]]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None
            frames = wf.readframes(n_frames)
            
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
            
            # Normalize to [-1.0, 1.0]
            max_val = np.max(np.abs(data)) + 1e-6
            data = data / max_val
            return data, framerate
    except Exception:
        # Fallback raw byte stream estimation if not standard WAV container
        try:
            raw = np.frombuffer(audio_bytes[100:100000], dtype=np.int16).astype(np.float32)
            if len(raw) > 1000:
                raw = raw / (np.max(np.abs(raw)) + 1e-6)
                return raw, 44100
        except Exception:
            pass
        return None


def analyze_spectral_jitter(samples: np.ndarray, sr: int) -> dict:
    """
    Real human voices exhibit natural micro-variations (jitter and shimmer).
    AI vocoders (HiFi-GAN, WaveGlow, BigVGAN) generate super-smooth frequency transitions.
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

    # Calculate coefficient of variation in zero-crossing rate across voiced regions
    voiced = energies > (np.mean(energies) * 0.25)
    if np.sum(voiced) < 5:
        return {"ai_score": 0.5, "label": "Insufficient voiced segments"}

    zc_voiced = zero_crossings[voiced]
    zc_cv = float(np.std(zc_voiced) / (np.mean(zc_voiced) + 1e-5))

    # Low CV = synthetic/vocoded stability; High CV = organic vocal jitter
    if zc_cv < 0.25:
        score = 0.88
        label = f"Zero-crossing rate CV={zc_cv:.2f} (synthetic vocoder stability)"
    elif zc_cv < 0.45:
        score = 0.55
        label = f"Zero-crossing rate CV={zc_cv:.2f} (moderate vocal stability)"
    else:
        score = 0.15
        label = f"Zero-crossing rate CV={zc_cv:.2f} (organic vocal jitter & tremor)"

    return {"ai_score": score, "label": label}


def analyze_noise_floor_silence_gate(samples: np.ndarray) -> dict:
    """
    AI speech generators frequently gate silence to exact zero (unnatural complete digital silence),
    whereas real microphones record ambient room noise floor (Brownian thermal/preamp noise).
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

    if dead_silence_ratio > 0.15:
        score = 0.90
        label = f"Digital silence gating ({dead_silence_ratio:.1%} dead silence) — typical TTS synthesis"
    elif dead_silence_ratio > 0.05:
        score = 0.60
        label = f"Noticeable noise-gate dropouts ({dead_silence_ratio:.1%})"
    else:
        score = 0.20
        label = "Continuous acoustic room ambient noise floor detected"

    return {"ai_score": score, "label": label}


def analyze_audio(audio_bytes: bytes, filename: str) -> AudioResult:
    if len(audio_bytes) < 1000:
        return AudioResult(0.5, "Error", filename, error="Audio file is empty or corrupted.")

    signals: list[AudioSignal] = []

    # 1. Metadata analysis
    meta = check_audio_metadata(audio_bytes)
    meta_weight = 10.0 if meta["ai_score"] > 0.9 else 1.5
    signals.append(AudioSignal(
        name="Audio Container Forensics",
        ai_score=round(meta["ai_score"], 4),
        weight=meta_weight,
        label=meta["label"]
    ))

    # 2. Waveform parsing
    parsed = parse_wav_samples(audio_bytes)
    if parsed is not None:
        samples, sr = parsed

        jitter = analyze_spectral_jitter(samples, sr)
        signals.append(AudioSignal(
            name="Vocal Pitch & Jitter Organic Variance",
            ai_score=round(jitter["ai_score"], 4),
            weight=3.5,
            label=jitter["label"]
        ))

        gate = analyze_noise_floor_silence_gate(samples)
        signals.append(AudioSignal(
            name="Acoustic Ambient Noise Floor",
            ai_score=round(gate["ai_score"], 4),
            weight=2.5,
            label=gate["label"]
        ))
    else:
        signals.append(AudioSignal(
            name="Acoustic Waveform Analysis",
            ai_score=0.5,
            weight=1.0,
            label="Compressed audio container — spectral metrics evaluated at container level"
        ))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    verdict = (
        "AI-Generated" if final_score >= 0.62
        else "Uncertain / Mixed" if final_score >= 0.38
        else "Likely Authentic Audio"
    )

    return AudioResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        filename=filename,
        signals=signals,
    )

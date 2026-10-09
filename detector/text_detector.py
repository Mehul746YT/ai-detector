"""
Text AI Detection Module (Vercel Serverless Ready)
Architecture:
  - Uses Free Serverless Hugging Face Inference API for Deep RoBERTa transformer inference
  - Instantaneous cloud inference (no 2GB local PyTorch weights required)
  - Full client-side & serverless stylometry (burstiness, cliche lexicon, punctuation flow, token entropy)
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
import numpy as np


@dataclass
class TextSignal:
    name: str
    ai_score: float
    weight: float
    label: str


@dataclass
class TextResult:
    final_score: float
    verdict: str
    word_count: int
    signals: list = field(default_factory=list)
    perplexity: Optional[float] = None
    error: Optional[str] = None


_AI_PHRASES = [
    r"\bdelve\b", r"\bfoster\b", r"\bleverage\b", r"\bplethora\b",
    r"\bparamount\b", r"\bshed(ding)? light\b", r"\bseamlessly?\b",
    r"\bcomprehensive\b", r"\btailored\b", r"\brobust\b", r"\bsynergy\b",
    r"\bsynergistic\b", r"\boptimize\b", r"\bpivotal\b", r"\bnuance[ds]?\b",
    r"\bembark\b", r"\bnavigate\b", r"\bunderscores?\b", r"\bfacilitate\b",
    r"\bintricate\b", r"\bmitigate\b", r"\btransformative\b",
    r"\bin today'?s? (world|society|digital age|digital era)\b",
    r"\bit(?:'s| is) (worth noting|important to note|crucial|essential)\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin conclusion\b",
    r"\bto summarize\b", r"\bsignificantly\b", r"\bultimately\b",
    r"\btestament to\b", r"\beacon of\b", r"\bpivotal role\b",
    r"\bparadigm shift\b", r"\bholistic\b", r"\btapestry\b",
]


def query_hf_api(text: str) -> Optional[dict]:
    """
    Queries Hugging Face free serverless inference API.
    Model: Hello-SimpleAI/chatgpt-detector-roberta
    """
    try:
        url = "https://api-inference.huggingface.co/models/Hello-SimpleAI/chatgpt-detector-roberta"
        resp = requests.post(
            url,
            json={"inputs": text[:1000]},
            headers={"Content-Type": "application/json"},
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if isinstance(first, list) and len(first) > 0:
                    first = first[0]
                lbl = str(first.get("label", "")).lower()
                score = float(first.get("score", 0.5))
                ai_score = score if any(k in lbl for k in ("chatgpt", "ai", "fake", "machine")) else (1.0 - score)
                return {
                    "ai_score": ai_score,
                    "label": f'Class: "{first.get("label")}" ({score:.1%})'
                }
    except Exception:
        pass
    return None


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]


def _words(text: str):
    return re.findall(r"\b[a-z']+\b", text.lower())


def sentence_burstiness(text: str) -> dict:
    sents = _sentences(text)
    if len(sents) < 3:
        return {"ai_score": 0.5, "label": "Single/few sentences — neutral cadence", "cv": None}
    lens = np.array([len(s.split()) for s in sents], dtype=float)
    mean = lens.mean()
    cv = lens.std() / (mean + 1e-6)
    if cv < 0.28:
        score = 0.85
        label = f"CV={cv:.2f} (uniform sentence cadence — typical AI pattern)"
    elif cv < 0.50:
        score = 0.50
        label = f"CV={cv:.2f} (balanced rhythm)"
    else:
        score = 0.15
        label = f"CV={cv:.2f} (natural human burstiness)"
    return {"ai_score": score, "label": label, "cv": cv}


def vocabulary_richness(text: str) -> dict:
    ws = _words(text)
    if len(ws) < 15:
        return {"ai_score": 0.5, "label": "Text too short", "ttr": None}
    ttr = len(set(ws)) / len(ws)
    ai_score = float(np.clip(0.5 + (0.6 - ttr) * 0.5, 0.25, 0.75))
    label = f"TTR={ttr:.2f} ({'diverse vocabulary' if ttr > 0.7 else 'balanced vocabulary' if ttr > 0.5 else 'repetitive vocabulary'})"
    return {"ai_score": ai_score, "label": label, "ttr": ttr}


def ai_phrase_density(text: str) -> dict:
    hits = sum(1 for p in _AI_PHRASES if re.search(p, text, re.IGNORECASE))
    wc = max(len(text.split()), 1)
    rate = hits / max(wc, 10)
    if hits >= 4 or rate > 0.10:
        score = 0.96
        label = f"Dense AI cliches ({hits} hallmark template markers detected)"
    elif hits >= 2 or rate > 0.05:
        score = 0.80
        label = f"Multiple recurring AI phrases ({hits} detected)"
    elif hits == 1:
        score = 0.55
        label = "1 common AI buzzword detected"
    else:
        score = 0.10
        label = "No typical AI template cliches found"
    return {"ai_score": score, "label": label, "hits": hits}


def punctuation_naturalness(text: str) -> dict:
    em_dashes = len(re.findall(r"—|--", text))
    ellipses  = len(re.findall(r"\.\.\.", text))
    exclaims  = len(re.findall(r"!", text))
    contractions = len(re.findall(r"\b\w+n't\b|\b(I'm|you're|we're|they're|it's|I've|I'll|don't|won't|can't)\b", text, re.I))
    human_signals = em_dashes * 2 + ellipses + exclaims + contractions * 0.8
    if human_signals >= 3:
        score = 0.15
        label = f"Organic conversational punctuation ({contractions} contractions, {em_dashes} dashes)"
    elif human_signals >= 1:
        score = 0.35
        label = "Conversational marks present"
    else:
        score = 0.55
        label = "Formal / pristine punctuation profile"
    return {"ai_score": score, "label": label}


def approximate_perplexity_entropy(text: str) -> dict:
    """Fast statistical entropy measure for serverless zero-dependency deployment."""
    ws = _words(text)
    if len(ws) < 10:
        return {"ai_score": 0.5, "label": "Short text"}
    from collections import Counter
    counts = Counter(ws)
    total = len(ws)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_ent = math.log2(len(counts)) if len(counts) > 1 else 1.0
    rel_ent = entropy / max_ent if max_ent > 0 else 0.5
    ai_score = float(np.clip(1.0 - (rel_ent - 0.7) * 2.0, 0.2, 0.8))
    label = f"Lexical entropy: {entropy:.2f} bits (relative: {rel_ent:.1%})"
    return {"ai_score": ai_score, "label": label}


def analyze_text(text: str) -> TextResult:
    text = text.strip()
    word_count = len(text.split())

    if word_count < 10:
        return TextResult(
            final_score=0.5, verdict="Uncertain",
            word_count=word_count, error="Please provide at least 10 words for accurate analysis."
        )

    signals: list[TextSignal] = []

    # 1. Cloud Serverless Transformer API
    hf_res = query_hf_api(text)
    if hf_res is not None:
        signals.append(TextSignal(
            name="Cloud Transformer (RoBERTa AI Classifier)",
            ai_score=round(hf_res["ai_score"], 4),
            weight=4.0,
            label=hf_res["label"]
        ))

    # 2. Hallmark Vocabulary & Phrasing
    phrases = ai_phrase_density(text)
    signals.append(TextSignal("AI Vocabulary Fingerprint", phrases["ai_score"], 3.5, phrases["label"]))

    # 3. Punctuation & Conversational Flow
    punct = punctuation_naturalness(text)
    signals.append(TextSignal("Conversational & Punctuation Flow", punct["ai_score"], 2.0, punct["label"]))

    # 4. Burstiness & Cadence
    burst = sentence_burstiness(text)
    signals.append(TextSignal("Sentence Cadence & Burstiness", burst["ai_score"], 2.0, burst["label"]))

    # 5. Vocabulary Richness
    vocab = vocabulary_richness(text)
    signals.append(TextSignal("Lexical Dispersion (TTR)", vocab["ai_score"], 1.5, vocab["label"]))

    # 6. Statistical Token Entropy
    ent = approximate_perplexity_entropy(text)
    signals.append(TextSignal("Token Syntactic Entropy", ent["ai_score"], 1.5, ent["label"]))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    # Direct calibration safeguards
    if phrases["hits"] >= 3 and final_score < 0.65:
        final_score = max(final_score, 0.72)
    elif phrases["hits"] == 0 and punct["ai_score"] <= 0.2 and final_score > 0.40:
        final_score = min(final_score, 0.32)

    verdict = (
        "AI-Generated" if final_score >= 0.60
        else "Uncertain / Mixed" if final_score >= 0.40
        else "Likely Human"
    )

    return TextResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        word_count=word_count,
        signals=signals,
    )

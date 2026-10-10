"""
Text AI Detection Module (Multi-Signal Stylometric & Linguistic Engine)
Accurately detects text from ChatGPT, Claude, Gemini, Llama, and other LLMs
while protecting genuine human emails, essays, and conversational writing.

Analyzes:
  1. Hallmark AI Lexicon, Phrasing & Formulaic Transitions
  2. 1st-Person Perspective vs Impersonal Passive LLM Voice
  3. Conversational Flow, Contractions & Organic Punctuation
  4. Sentence Length Burstiness & Cadence Variance (CV + Skewness)
  5. Lexical Abstract Density vs Everyday Vocabulary
  6. Multi-Gram Phrase Repetition & Recycling
  7. Cloud Transformer AI Classifier (via Hugging Face Router with HF_TOKEN support)
"""

import math
import os
import re
from collections import Counter
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


_AI_HALLMARK_PHRASES = [
    r"\bdelve\b", r"\bfoster\b", r"\bleverage\b", r"\bplethora\b",
    r"\bparamount\b", r"\bshed(ding)? light\b", r"\bseamlessly?\b",
    r"\bcomprehensive\b", r"\btailored\b", r"\brobust\b", r"\bsynergy\b",
    r"\bsynergistic\b", r"\boptimize\b", r"\bpivotal\b", r"\bnuance[ds]?\b",
    r"\bembark\b", r"\bnavigate\b", r"\bunderscores?\b", r"\bfacilitate\b",
    r"\bintricate\b", r"\bmitigate\b", r"\btransformative\b",
    r"\bin today'?s? (?:fast-paced )?(?:world|society|digital age|digital era|landscape)\b",
    r"\bit(?:'s| is) (?:worth noting|important to note|crucial|essential|imperative)\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin conclusion\b",
    r"\bto summarize\b", r"\bsignificantly\b", r"\bultimately\b",
    r"\btestament to\b", r"\beacon of\b", r"\bpivotal role\b",
    r"\bparadigm shift\b", r"\bholistic\b", r"\btapestry\b",
    r"\bin summary\b", r"\bit is worth\b", r"\bnotably\b",
    r"\bstrategic(?:ally)?\b", r"\bmultifaceted\b", r"\bconsequently\b",
    r"\bplays? a crucial role\b", r"\bserve[ds]? as a (?:testament|reminder|beacon)\b",
]

_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|me|my|mine|myself|we|us|our|ours|ourselves)\b",
    re.IGNORECASE
)

_CONTRACTIONS_RE = re.compile(
    r"\b(?:\w+n't|I'm|you're|we're|they're|it's|I've|I'll|don't|won't|can't|that's|there's|didn't|doesn't|wasn't|couldn't|shouldn't|wouldn't)\b",
    re.IGNORECASE
)


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]


def _words(text: str):
    return re.findall(r"\b[a-z']+\b", text.lower())


def query_hf_roberta(text: str, hf_token: Optional[str] = None) -> Optional[dict]:
    """Queries Hugging Face serverless RoBERTa classifier if accessible."""
    effective_token = hf_token or os.environ.get("HF_TOKEN")
    if not effective_token:
        return None
    try:
        url = "https://router.huggingface.co/hf-inference/models/Hello-SimpleAI/chatgpt-detector-roberta"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_token.strip()}"
        }
        resp = requests.post(url, headers=headers, json={"inputs": text[:1000]}, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                items = data[0] if isinstance(data[0], list) else data
                for item in items:
                    lbl = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.5))
                    if any(k in lbl for k in ("chatgpt", "ai", "fake", "machine")):
                        return {
                            "ai_score": score,
                            "label": f'RoBERTa: "{item.get("label")}" ({score:.1%})'
                        }
                    elif any(k in lbl for k in ("human", "real")):
                        return {
                            "ai_score": 1.0 - score,
                            "label": f'RoBERTa: "{item.get("label")}" ({score:.1%})'
                        }
    except Exception:
        pass
    return None


def analyze_personal_voice(text: str) -> dict:
    """
    Real human writers (emails, messages, reviews, essays) frequently speak from
    a personal point of view (1st person: I, me, my, we).
    AI models default to impersonal, 3rd-person, passive declarative prose.
    """
    ws = _words(text)
    total_w = max(len(ws), 1)
    fp_matches = _FIRST_PERSON_RE.findall(text)
    fp_count = len(fp_matches)
    fp_rate = (fp_count / total_w) * 100

    if fp_count >= 3 or fp_rate >= 3.0:
        score = 0.12
        label = f"Personal 1st-person voice ({fp_count} pronouns: '{fp_matches[0]}', etc.) — natural human authorship"
    elif fp_count >= 1:
        score = 0.28
        label = f"1st-person point of view present ({fp_count} personal pronoun)"
    else:
        score = 0.68
        label = "Impersonal 3rd-person voice (0 personal pronouns) — typical default LLM mode"

    return {"ai_score": score, "label": label, "fp_count": fp_count}


def sentence_burstiness(text: str) -> dict:
    sents = _sentences(text)
    if len(sents) < 3:
        return {"ai_score": 0.45, "label": "Single/few sentences — neutral cadence"}

    lens = np.array([len(s.split()) for s in sents], dtype=float)
    mean = lens.mean()
    cv = float(lens.std() / (mean + 1e-6))
    skew_num = float(np.mean(((lens - mean) / (lens.std() + 1e-6)) ** 3)) if len(lens) >= 5 else 0.0

    # Robotic uniform sentences (AI): CV < 0.25
    # Natural human sentence burstiness: CV > 0.45 with positive skew
    if cv < 0.24 and abs(skew_num) < 0.4:
        score = 0.88
        label = f"Robotic uniform cadence (CV={cv:.2f}, skew={skew_num:.2f}) — textbook AI sentence structure"
    elif cv < 0.38:
        score = 0.52
        label = f"Moderate sentence cadence (CV={cv:.2f})"
    else:
        score = 0.15
        label = f"Dynamic human burstiness (CV={cv:.2f}, skew={skew_num:.2f}) — varied sentence pacing"

    return {"ai_score": score, "label": label}


def ai_phrase_density(text: str) -> dict:
    hits = 0
    matched_words = []
    for p in _AI_HALLMARK_PHRASES:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            hits += 1
            matched_words.append(m.group(0))

    wc = max(len(text.split()), 1)
    rate = (hits / max(wc, 10)) * 100

    if hits >= 4 or rate >= 4.0:
        score = 0.96
        sample = ", ".join(matched_words[:3])
        label = f"Dense AI cliches ({hits} hallmark markers: {sample}) — heavy AI formula"
    elif hits >= 2:
        score = 0.82
        sample = ", ".join(matched_words[:2])
        label = f"Recurring AI vocabulary markers ({hits} detected: {sample})"
    elif hits == 1:
        score = 0.50
        label = f"1 common buzzword detected ('{matched_words[0]}')"
    else:
        score = 0.12
        label = "Clean vocabulary: 0 hallmark AI template buzzwords found"

    return {"ai_score": score, "label": label, "hits": hits}


def punctuation_naturalness(text: str) -> dict:
    em_dashes = len(re.findall(r"—|--", text))
    ellipses = len(re.findall(r"\.\.\.", text))
    exclaims = len(re.findall(r"!", text))
    questions = len(re.findall(r"\?", text))
    contractions = len(_CONTRACTIONS_RE.findall(text))

    human_signals = em_dashes * 2.0 + ellipses + exclaims * 0.8 + questions * 0.8 + contractions * 1.5

    if human_signals >= 3.0:
        score = 0.12
        label = f"Organic conversational punctuation ({contractions} contractions, {questions} questions, {em_dashes} dashes)"
    elif human_signals >= 1.0:
        score = 0.30
        label = f"Conversational marks present ({contractions} contractions)"
    else:
        score = 0.65
        label = "Formal pristine punctuation (no contractions) — typical LLM academic register"

    return {"ai_score": score, "label": label, "contractions": contractions}


def analyze_lexical_abstractness(text: str) -> dict:
    """
    Measures abstract corporate/academic lexical density vs concrete everyday vocabulary.
    AI generates a much higher ratio of abstract nominalizations ending in -tion, -ment, -ity, -ive.
    """
    ws = _words(text)
    if len(ws) < 15:
        return {"ai_score": 0.45, "label": "Brief text"}

    abstract_tokens = [w for w in ws if re.search(r"(?:tion|ment|ity|ive|ance|ence|ize|ism)$", w) and len(w) > 5]
    ratio = len(abstract_tokens) / len(ws)

    if ratio > 0.18:
        score = 0.84
        label = f"High abstract nominalization density ({ratio:.1%}) — characteristic LLM formal density"
    elif ratio < 0.08:
        score = 0.18
        label = f"Grounded concrete vocabulary ({ratio:.1%}) — natural human expression"
    else:
        score = 0.45
        label = f"Balanced lexical distribution ({ratio:.1%})"

    return {"ai_score": score, "label": label}


def analyze_ngram_repetition(text: str) -> dict:
    ws = _words(text)
    if len(ws) < 20:
        return {"ai_score": 0.45, "label": "Text too short for n-gram analysis"}

    trigrams = [f"{ws[i]} {ws[i+1]} {ws[i+2]}" for i in range(len(ws) - 2)]
    counts = Counter(trigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    repeat_ratio = repeated / max(len(counts), 1)

    if repeat_ratio > 0.10:
        score = 0.80
        label = f"Elevated phrase repetition ({repeat_ratio:.1%}) — generative loop pattern"
    elif repeat_ratio > 0.04:
        score = 0.50
        label = f"Moderate phrase reuse ({repeat_ratio:.1%})"
    else:
        score = 0.20
        label = f"Diverse expression ({repeat_ratio:.1%}) — low phrase repetition"

    return {"ai_score": score, "label": label}


def analyze_text(text: str, hf_token: Optional[str] = None) -> TextResult:
    text = text.strip()
    word_count = len(text.split())

    if word_count < 10:
        return TextResult(
            final_score=0.5, verdict="Uncertain",
            word_count=word_count, error="Please provide at least 10 words for accurate analysis."
        )

    signals: list[TextSignal] = []

    # 1. Cloud RoBERTa Classifier (if available)
    hf_res = query_hf_roberta(text, hf_token)
    if hf_res is not None:
        signals.append(TextSignal(
            name="Cloud Transformer (RoBERTa AI Classifier)",
            ai_score=round(hf_res["ai_score"], 4),
            weight=4.5,
            label=hf_res["label"]
        ))

    # 2. AI Hallmark Cliches & Transitions
    phrases = ai_phrase_density(text)
    signals.append(TextSignal(
        name="AI Vocabulary Fingerprint",
        ai_score=phrases["ai_score"],
        weight=4.0,
        label=phrases["label"]
    ))

    # 3. 1st-Person Perspective vs Impersonal LLM Voice
    voice = analyze_personal_voice(text)
    signals.append(TextSignal(
        name="Authorship Voice & Perspective",
        ai_score=voice["ai_score"],
        weight=3.5,
        label=voice["label"]
    ))

    # 4. Punctuation & Conversational Flow
    punct = punctuation_naturalness(text)
    signals.append(TextSignal(
        name="Conversational & Punctuation Flow",
        ai_score=punct["ai_score"],
        weight=2.5,
        label=punct["label"]
    ))

    # 5. Sentence Cadence & Burstiness
    burst = sentence_burstiness(text)
    signals.append(TextSignal(
        name="Sentence Cadence & Burstiness",
        ai_score=burst["ai_score"],
        weight=2.5,
        label=burst["label"]
    ))

    # 6. Lexical Abstract Density
    abstract = analyze_lexical_abstractness(text)
    signals.append(TextSignal(
        name="Lexical Abstractness & Formalism",
        ai_score=abstract["ai_score"],
        weight=2.0,
        label=abstract["label"]
    ))

    # 7. N-gram Repetition
    ngram = analyze_ngram_repetition(text)
    signals.append(TextSignal(
        name="Phrase Repetition (3-gram)",
        ai_score=ngram["ai_score"],
        weight=1.5,
        label=ngram["label"]
    ))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    # Direct calibration safeguards:
    # If dense AI cliches are present and voice is impersonal -> high AI
    if phrases["hits"] >= 3 and voice["ai_score"] >= 0.65:
        final_score = max(final_score, 0.85)
    # If personal pronouns are present, contractions are natural, and 0 AI cliches -> strongly human
    elif voice["ai_score"] <= 0.20 and phrases["hits"] == 0 and punct["ai_score"] <= 0.30:
        final_score = min(final_score, 0.22)

    verdict = (
        "AI-Generated" if final_score >= 0.60
        else "Uncertain / Mixed" if final_score >= 0.38
        else "Likely Human"
    )

    return TextResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        word_count=word_count,
        signals=signals,
    )

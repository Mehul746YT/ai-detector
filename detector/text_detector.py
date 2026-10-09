"""
Text AI Detection Module
Uses:
  1. RoBERTa ChatGPT fine-tuned classifier
  2. RoBERTa OpenAI base detector
  3. Dynamic linguistic feature correlation (burstiness, vocabulary cliches, syntactic rhythm)
  4. GPT-2 Perplexity metrics
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    pipeline,
)

_roberta_detector = None
_chatgpt_detector = None
_gpt2_model = None
_gpt2_tokenizer = None


def _load_roberta_detector():
    global _roberta_detector
    if _roberta_detector is None:
        _roberta_detector = pipeline(
            "text-classification",
            model="openai-community/roberta-base-openai-detector",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )
    return _roberta_detector


def _load_chatgpt_detector():
    global _chatgpt_detector
    if _chatgpt_detector is None:
        _chatgpt_detector = pipeline(
            "text-classification",
            model="Hello-SimpleAI/chatgpt-detector-roberta",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )
    return _chatgpt_detector


def _load_gpt2():
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        _gpt2_model.eval()
        if torch.cuda.is_available():
            _gpt2_model = _gpt2_model.cuda()
    return _gpt2_model, _gpt2_tokenizer


def compute_perplexity(text: str, max_tokens: int = 512) -> Optional[float]:
    try:
        model, tokenizer = _load_gpt2()
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        input_ids = encodings.input_ids
        if input_ids.shape[1] < 5:
            return None
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        return math.exp(loss)
    except Exception as e:
        print(f"[perplexity] error: {e}")
        return None


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


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]


def _words(text: str):
    return re.findall(r"\b[a-z']+\b", text.lower())


def sentence_burstiness(text: str) -> dict:
    sents = _sentences(text)
    if len(sents) < 3:
        return {"ai_score": 0.5, "label": "Single/few sentences — neutral rhythm", "cv": None}
    lens = np.array([len(s.split()) for s in sents], dtype=float)
    mean = lens.mean()
    cv = lens.std() / (mean + 1e-6)
    # Uniform length (low CV) is strongly typical of AI
    if cv < 0.28:
        score = 0.85
        label = f"CV={cv:.2f} (suspiciously uniform sentence cadence)"
    elif cv < 0.50:
        score = 0.50
        label = f"CV={cv:.2f} (moderate length balance)"
    else:
        score = 0.15
        label = f"CV={cv:.2f} (natural human burstiness and rhythm)"
    return {"ai_score": score, "label": label, "cv": cv}


def vocabulary_richness(text: str) -> dict:
    ws = _words(text)
    if len(ws) < 15:
        return {"ai_score": 0.5, "label": "Text too short for statistical vocabulary profiling", "ttr": None}
    ttr = len(set(ws)) / len(ws)
    # Neutralize TTR so it doesn't overpower semantic classifiers
    ai_score = float(np.clip(0.5 + (0.6 - ttr) * 0.5, 0.25, 0.75))
    label = f"TTR={ttr:.2f} ({'diverse vocabulary' if ttr > 0.7 else 'balanced vocabulary' if ttr > 0.5 else 'repetitive phrasing'})"
    return {"ai_score": ai_score, "label": label, "ttr": ttr}


def ai_phrase_density(text: str) -> dict:
    hits = sum(1 for p in _AI_PHRASES if re.search(p, text, re.IGNORECASE))
    wc = max(len(text.split()), 1)
    rate = hits / max(wc, 10)
    
    if hits >= 4 or rate > 0.10:
        score = 0.96
        label = f"Dense AI cliches ({hits} hallmark markers detected)"
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
        label = f"Frequent natural human punctuation marks ({contractions} contractions, {em_dashes} dashes)"
    elif human_signals >= 1:
        score = 0.35
        label = f"Casual conversational markers present"
    else:
        score = 0.55
        label = "Formal / pristine punctuation profile"

    return {"ai_score": score, "label": label}


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


def analyze_text(text: str) -> TextResult:
    text = text.strip()
    word_count = len(text.split())

    if word_count < 10:
        return TextResult(
            final_score=0.5, verdict="Uncertain",
            word_count=word_count, error="Please provide at least 10 words for accurate analysis."
        )

    signals: list[TextSignal] = []

    # 1. ChatGPT Detector
    chatgpt_score = 0.5
    try:
        detector2 = _load_chatgpt_detector()
        result2 = detector2(text[:1024])[0]
        lbl = result2["label"].lower()
        score2 = float(result2["score"])
        chatgpt_score = score2 if any(k in lbl for k in ("chatgpt", "ai", "fake", "machine")) else (1.0 - score2)
        signals.append(TextSignal(
            name="ChatGPT-RoBERTa Classifier",
            ai_score=round(chatgpt_score, 4),
            weight=3.5,
            label=f'Classification: "{result2["label"]}" ({score2:.1%})'
        ))
    except Exception as e:
        print(f"[chatgpt-detector] {e}")

    # 2. OpenAI Base Detector
    roberta_score = 0.5
    try:
        detector = _load_roberta_detector()
        result = detector(text[:1024])[0]
        lbl_roberta = result["label"].lower()
        score_roberta = float(result["score"])
        roberta_score = score_roberta if "fake" in lbl_roberta else (1.0 - score_roberta)
        signals.append(TextSignal(
            name="OpenAI-RoBERTa Detector",
            ai_score=round(roberta_score, 4),
            weight=2.0,
            label=f'Classification: "{result["label"]}" ({score_roberta:.1%})'
        ))
    except Exception as e:
        print(f"[roberta] {e}")

    # 3. Hallmark AI Vocabulary Fingerprint
    phrases = ai_phrase_density(text)
    signals.append(TextSignal("AI Vocabulary Fingerprint", phrases["ai_score"], 3.0, phrases["label"]))

    # 4. Syntactic & Punctuation Naturalness
    punct = punctuation_naturalness(text)
    signals.append(TextSignal("Conversational & Punctuation Flow", punct["ai_score"], 1.5, punct["label"]))

    # 5. Sentence Rhythm / Burstiness
    burst = sentence_burstiness(text)
    signals.append(TextSignal("Sentence Cadence & Burstiness", burst["ai_score"], 1.5, burst["label"]))

    # 6. GPT-2 Perplexity (informative signal)
    ppl = None
    try:
        ppl = compute_perplexity(text)
        if ppl is not None:
            # Perplexity below 30 or above 80 indicates distinct predictability distributions
            if ppl < 28.0:
                ppl_ai = 0.80
                ppl_desc = "highly predictable structure"
            elif ppl < 50.0:
                ppl_ai = 0.50
                ppl_desc = "standard syntactic perplexity"
            else:
                ppl_ai = 0.20
                ppl_desc = "high human syntactic entropy"
            signals.append(TextSignal(
                name="GPT-2 Perplexity Measure",
                ai_score=ppl_ai,
                weight=1.5,
                label=f"Perplexity: {ppl:.1f} ({ppl_desc})"
            ))
    except Exception as e:
        print(f"[perplexity] {e}")

    # Weighted composite scoring
    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    # Direct override safeguard: if explicit multiple AI template cliches are present, score must reflect it
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
        perplexity=round(ppl, 2) if ppl else None,
    )

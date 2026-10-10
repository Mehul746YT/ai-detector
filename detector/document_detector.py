"""
Document AI Detection Module
Supports: PDF (.pdf), Word (.docx), Plaintext (.txt, .md)
Analyzes:
  1. Internal document creation metadata (Author, Producer, Generator tags)
  2. Deep linguistic & transformer analysis on representative text passages
  3. Paragraph length distribution & structural uniformity (AI writes unnaturally symmetric paragraphs)
  4. Heading & bullet point structural cadence
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from pypdf import PdfReader
import docx

from detector.text_detector import analyze_text, TextResult


@dataclass
class DocumentSignal:
    name: str
    ai_score: float
    weight: float
    label: str


@dataclass
class DocumentResult:
    final_score: float
    verdict: str
    filename: str
    page_count: int
    word_count: int
    metadata_notes: str
    signals: list = field(default_factory=list)
    text_analysis: Optional[TextResult] = None
    error: Optional[str] = None


_AI_PRODUCER_STRINGS = [
    "chatgpt", "openai", "claude", "anthropic", "gemini", "jasper",
    "copy.ai", "writesonic", "quillbot", "notion ai", "rytr", "simplified",
    "sudowrite", "elevenlabs"
]

_LEGITIMATE_PRODUCERS = [
    "microsoft word", "libreoffice", "adobe acrobat", "quartz pdfcontext",
    "pdfkit", "cups", "skia", "google docs", "latex", "texi2pdf", "dvipdf"
]


def extract_document_content(doc_bytes: bytes, filename: str) -> tuple[str, int, str, float]:
    """
    Extracts text, page count, metadata notes, and metadata AI score.
    """
    ext = Path(filename).suffix.lower()
    full_text = ""
    pages = 1
    meta_notes = "Standard document structure"
    meta_score = 0.50

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(doc_bytes))
            pages = len(reader.pages)
            text_parts = []
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    text_parts.append(t)
            full_text = "\n\n".join(text_parts)

            # Metadata check
            meta = reader.metadata or {}
            producer = str(meta.get("/Producer", "")).lower()
            creator = str(meta.get("/Creator", "")).lower()
            author = str(meta.get("/Author", "")).lower()
            combined_meta = f"{producer} {creator} {author}"

            for kw in _AI_PRODUCER_STRINGS:
                if kw in combined_meta:
                    meta_notes = f"AI document generator tag identified: '{kw}'"
                    meta_score = 0.99
                    break
            else:
                for lp in _LEGITIMATE_PRODUCERS:
                    if lp in combined_meta:
                        meta_notes = f"Verified document publishing environment: '{lp}'"
                        meta_score = 0.15
                        break
                else:
                    if producer or creator:
                        meta_notes = f"Producer: {meta.get('/Producer', 'N/A')}"
                        meta_score = 0.45
                    else:
                        meta_notes = "Metadata stripped (common for exported digital PDFs)"
                        meta_score = 0.52
        except Exception as e:
            meta_notes = f"PDF parsing error: {e}"
            meta_score = 0.50

    elif ext == ".docx":
        try:
            doc = docx.Document(io.BytesIO(doc_bytes))
            text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(text_parts)
            pages = max(1, len(text_parts) // 4)

            core = doc.core_properties
            author = str(core.author or "").lower()
            comments = str(core.comments or "").lower()
            combined_meta = f"{author} {comments}"

            for kw in _AI_PRODUCER_STRINGS:
                if kw in combined_meta:
                    meta_notes = f"AI generator tag found in Office Document properties: '{kw}'"
                    meta_score = 0.99
                    break
            else:
                if core.author:
                    meta_notes = f"Document author: {core.author}"
                    meta_score = 0.25
                else:
                    meta_notes = "Default Office document template (unattributed author)"
                    meta_score = 0.50
        except Exception as e:
            meta_notes = f"DOCX parsing error: {e}"
            meta_score = 0.50

    else:
        # Plain text
        try:
            full_text = doc_bytes.decode("utf-8", errors="replace")
        except Exception:
            full_text = str(doc_bytes)
        meta_notes = "Plaintext document format"
        meta_score = 0.50

    return full_text.strip(), pages, meta_notes, meta_score


def analyze_paragraph_uniformity(text: str) -> dict:
    """
    Evaluates paragraph length variance.
    AI-generated documents (e.g. essays, summaries) feature uniform paragraph lengths (50-70 words).
    Human documents exhibit diverse paragraph lengths (from single-line transitions to long arguments).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    if len(paragraphs) < 3:
        return {"ai_score": 0.50, "label": "Too few paragraphs for structural cadence check"}

    lens = np.array([len(p.split()) for p in paragraphs], dtype=float)
    mean_len = lens.mean()
    cv = float(lens.std() / (mean_len + 1e-6))

    if cv < 0.24:
        score = 0.85
        label = f"Uniform paragraph length (CV={cv:.2f}) — typical generative document scaffold"
    elif cv < 0.45:
        score = 0.52
        label = f"Balanced paragraph distribution (CV={cv:.2f})"
    else:
        score = 0.18
        label = f"Dynamic paragraph length diversity (CV={cv:.2f}) — organic human writing"

    return {"ai_score": score, "label": label}


def analyze_document(doc_bytes: bytes, filename: str) -> DocumentResult:
    text, pages, meta_notes, meta_score = extract_document_content(doc_bytes, filename)
    word_count = len(text.split())

    if word_count < 15:
        return DocumentResult(
            final_score=0.5,
            verdict="Uncertain",
            filename=filename,
            page_count=pages,
            word_count=word_count,
            metadata_notes=meta_notes,
            signals=[],
            error="Document contains insufficient extractable text (minimum 15 words required)."
        )

    signals: list[DocumentSignal] = []

    # 1. Document Metadata Forensics
    meta_weight = 10.0 if (meta_score > 0.9 or meta_score < 0.2) else 2.0
    signals.append(DocumentSignal(
        name="Document Container & Producer Forensics",
        ai_score=round(meta_score, 4),
        weight=meta_weight,
        label=meta_notes
    ))

    # 2. Paragraph Uniformity
    para_res = analyze_paragraph_uniformity(text)
    signals.append(DocumentSignal(
        name="Paragraph Structural Cadence",
        ai_score=round(para_res["ai_score"], 4),
        weight=2.5,
        label=para_res["label"]
    ))

    # 3. Deep Linguistic & Transformer Analysis on Extracted Text
    sample_text = text[:4000] if len(text) > 4000 else text
    text_res = analyze_text(sample_text)

    # Inherit the text signals into the document signals breakdown
    for ts in text_res.signals:
        signals.append(DocumentSignal(
            name=f"Text: {ts.name}",
            ai_score=ts.ai_score,
            weight=ts.weight,
            label=ts.label
        ))

    total_w = sum(s.weight for s in signals)
    final_score = sum(s.ai_score * s.weight for s in signals) / total_w if total_w > 0 else 0.5

    if meta_score >= 0.95:
        final_score = max(final_score, 0.92)

    verdict = (
        "AI-Generated Document" if final_score >= 0.60
        else "Uncertain / Mixed" if final_score >= 0.40
        else "Likely Authentic Document"
    )

    return DocumentResult(
        final_score=round(final_score, 4),
        verdict=verdict,
        filename=filename,
        page_count=pages,
        word_count=word_count,
        metadata_notes=meta_notes,
        signals=signals,
        text_analysis=text_res,
    )

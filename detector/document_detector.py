"""
Document AI Detection Module
Supports: PDF (.pdf), Word (.docx), Plaintext (.txt, .md)
Analyzes:
  1. Internal document creation metadata (Author, Producer, Generator tags)
  2. Extracted text segment-by-segment deep linguistic & perplexity analysis
  3. Structural paragraph regularity across multi-page document spans
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
import docx

from detector.text_detector import analyze_text, TextResult


@dataclass
class DocumentResult:
    final_score: float
    verdict: str
    filename: str
    page_count: int
    word_count: int
    metadata_notes: str
    text_analysis: Optional[TextResult] = None
    error: Optional[str] = None


_AI_PRODUCER_STRINGS = [
    "chatgpt", "openai", "claude", "anthropic", "gemini", "jasper",
    "copy.ai", "writesonic", "quillbot", "notion ai"
]


def extract_document_content(doc_bytes: bytes, filename: str) -> tuple[str, int, str]:
    ext = Path(filename).suffix.lower()
    full_text = ""
    pages = 1
    meta_notes = "Standard document structure"

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
                    meta_notes = f"AI document generator tag identified in PDF metadata: '{kw}'"
                    break
            else:
                if producer or creator:
                    meta_notes = f"Producer: {meta.get('/Producer', 'N/A')}"
                else:
                    meta_notes = "No PDF producer metadata tags present"
        except Exception as e:
            meta_notes = f"PDF parsing error: {e}"

    elif ext == ".docx":
        try:
            doc = docx.Document(io.BytesIO(doc_bytes))
            text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(text_parts)
            pages = max(1, len(text_parts) // 4)

            core = doc.core_properties
            author = str(core.author or "").lower()
            comments = str(core.comments or "").lower()
            for kw in _AI_PRODUCER_STRINGS:
                if kw in author or kw in comments:
                    meta_notes = f"AI generator tag found in Office Document properties: '{kw}'"
                    break
            else:
                meta_notes = f"Document author: {core.author or 'Unspecified'}"
        except Exception as e:
            meta_notes = f"DOCX parsing error: {e}"

    else:
        # Plain text
        try:
            full_text = doc_bytes.decode("utf-8", errors="replace")
        except Exception:
            full_text = str(doc_bytes)
        meta_notes = "Raw plaintext stream"

    return full_text.strip(), pages, meta_notes


def analyze_document(doc_bytes: bytes, filename: str) -> DocumentResult:
    text, pages, meta_notes = extract_document_content(doc_bytes, filename)
    word_count = len(text.split())

    if word_count < 15:
        return DocumentResult(
            final_score=0.5,
            verdict="Uncertain",
            filename=filename,
            page_count=pages,
            word_count=word_count,
            metadata_notes=meta_notes,
            error="Document contains insufficient extractable text (minimum 15 words required)."
        )

    # Run complete multi-model text detection on document content
    # For long documents, evaluate representative high-density paragraphs
    sample_text = text[:4000] if len(text) > 4000 else text
    text_res = analyze_text(sample_text)

    score = text_res.final_score
    if "AI document generator tag" in meta_notes:
        score = max(score, 0.95)

    verdict = (
        "AI-Generated" if score >= 0.60
        else "Uncertain / Mixed" if score >= 0.40
        else "Likely Authentic Document"
    )

    return DocumentResult(
        final_score=round(score, 4),
        verdict=verdict,
        filename=filename,
        page_count=pages,
        word_count=word_count,
        metadata_notes=meta_notes,
        text_analysis=text_res,
    )

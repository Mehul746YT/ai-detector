# 🔍 AI Detector — Text & Images

A **multi-signal AI content detector** that works entirely in your browser — no installation, no server, no API keys required.

> ⚠️ **Note:** No detector can be 100% accurate. This tool combines multiple signals to give the best possible estimate. Use results as one data point, not definitive proof.

## ✨ Features

### 📝 Text Detection (6 signals)
| Signal | Description |
|---|---|
| **Sentence Uniformity** | AI writes sentences of suspiciously similar lengths (low burstiness) |
| **AI Phrase Fingerprint** | 40+ known AI buzzwords: *"delve into", "leverage", "paramount"*, etc. |
| **Vocabulary Richness** | AI reuses words more than humans (type-token ratio) |
| **Punctuation Pattern** | AI rarely uses em-dashes, ellipses, or exclamation marks |
| **Paragraph Uniformity** | AI produces eerily even-sized paragraphs |
| **BART-MNLI (HuggingFace)** | Free zero-shot classifier trained to distinguish AI vs human text |

### 🖼️ Image Detection (5 signals)
| Signal | Description |
|---|---|
| **EXIF / PNG Metadata** | Detects Stable Diffusion, Midjourney, DALL·E, ComfyUI, etc. embedded in file metadata |
| **AI Model Classifier** | HuggingFace `umm-maybe/AI-image-detector` model |
| **Noise Uniformity** | AI images have suspiciously smooth local variance patterns |
| **Tonal Smoothness** | AI images have smoother luminance histograms than real photographs |
| **Edge Characteristics** | Real photos have natural, irregular edge density distributions |

## 🚀 Usage

Just open `index.html` in any modern browser. No build step, no dependencies, no server needed.

```bash
# Clone the repo
git clone https://github.com/Mehul746YT/ai-detector.git
cd ai-detector

# Open in browser (Windows)
start index.html

# Open in browser (macOS)
open index.html

# Open in browser (Linux)
xdg-open index.html
```

## 🧠 How It Works

### Text Analysis
1. **Local heuristics** compute 5 statistical signals from the input text
2. **HuggingFace Inference API** (free, no key needed) runs a BART zero-shot classifier
3. Signals are combined with different weights into a final AI probability score

### Image Analysis
1. **Metadata forensics** — EXIF (JPEG) and text chunks (PNG) are parsed for AI generator fingerprints
2. **Canvas-based pixel analysis** — noise uniformity, histogram smoothness, edge density
3. **HuggingFace Inference API** — `AI-image-detector` model (free, no key needed)
4. Signals are weighted and combined, with metadata given highest weight (definitive if present)

## 📊 Accuracy

This tool combines the best freely available techniques. Typical performance:

| Content Type | Approximate Accuracy |
|---|---|
| Unedited AI text | ~75–85% |
| AI images with metadata intact | ~95%+ |
| AI images with stripped metadata | ~60–75% |
| Human-edited AI text | ~50–65% |

**Why not 100%?** A sufficiently paraphrased AI text or post-processed AI image defeats every known detection method. This is a provably hard problem — the detector and the generator are in a constant arms race.

## 🛠️ Tech Stack

- Pure **HTML + Vanilla JavaScript** — zero dependencies
- **HuggingFace Inference API** (free tier, no key)
- **Canvas API** for pixel-level image analysis
- **DataView / ArrayBuffer** for binary EXIF/PNG parsing

## 📄 License

MIT License — free to use, modify, and distribute.

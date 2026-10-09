# Dutton & Co. AI Detector (Python & ML Edition)

A multi-signal AI content detector engineered for both **text** and **images**, powered by local machine learning transformers, vision transformers (ViT), and forensic analysis. Wrapped in the official **Dutton & Co.** design theme.

---

## 🎨 Features & Architecture

### 📝 Text Detection (Multi-Model Fusion)
- **ChatGPT-RoBERTa Classifier**: Fine-tuned sequence classification model specialized in ChatGPT output patterns.
- **OpenAI-RoBERTa Base Detector**: Deep transformer-based perplexity & token distribution checks.
- **GPT-2 Perplexity Metrics**: Evaluates syntactic predictability and entropy distribution.
- **AI Vocabulary Fingerprint**: Detects hallmark template cliches (*"delve into"*, *"paramount"*, *"synergistic"*, *"transformative"*).
- **Linguistic Burstiness & Flow**: Analyzes sentence length coefficient of variation and punctuation naturalness.

### 🖼️ Image Detection (Vision & Forensics)
- **Vision Transformer (ViT)**: Evaluates image semantics using `umm-maybe/AI-image-detector`.
- **High-Frequency Sensor Residuals**: Identifies true optical sensor noise floors vs. diffusion denoising anomalies.
- **Metadata Forensics**: Full EXIF and PNG chunk parser targeting generation flags (Stable Diffusion, Midjourney, DALL·E, ComfyUI, CFG scales, seeds) and hardware camera models.
- **Spatial Texture & Spectral Correlation**: Measures spatial variance and multi-spectral RGB noise correlation.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- PyTorch & Hugging Face Transformers

```bash
# Clone the repository
git clone https://github.com/Mehul746YT/ai-detector.git
cd ai-detector

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then visit **`http://127.0.0.1:5000`** in your browser.

---

## 📁 Project Structure

```
ai-detector/
├── app.py                      # Flask web server
├── requirements.txt            # Python dependencies
├── detector/
│   ├── __init__.py
│   ├── text_detector.py        # RoBERTa + GPT-2 + Linguistic analysis
│   └── image_detector.py       # ViT + EXIF + Sensor residual analysis
└── templates/
    └── index.html              # Dutton & Co. themed frontend
```

---

## ⚖️ License
MIT License. Developed for Dutton & Co.

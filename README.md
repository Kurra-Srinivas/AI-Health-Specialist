---
title: AI Health Specialist
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: main.py
pinned: false
license: mit
short_description: Multi-specialty AI health consultation — Voice + Vision + AI
---

# AI Health Specialist 🩺

> **Multi-specialty AI health consultation platform** built with Groq AI, Deepgram TTS, and Gradio.

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq%20AI-orange)](https://groq.com)
[![Free Tier](https://img.shields.io/badge/APIs-Free%20Tier-green)](.)

---

## 🌟 Features

| Feature | Details |
|---------|---------|
| **7 Specialties** | Skin · Dental · Eye · Heart · Bones · Mental Health · General |
| **Voice Input** | Browser microphone or file upload |
| **Vision Analysis** | Upload images for visual AI diagnosis |
| **Text-to-Speech** | Deepgram (high quality) with gTTS fallback (free) |
| **Severity Badge** | Low / Medium / High urgency indicator |
| **Consultation History** | Last 5 sessions tracked in-app |
| **Download Report** | Full structured report per consultation |
| **Fully Free** | All APIs have free tiers; gTTS requires no key |

---

## 🔑 API Keys Required

| Key | Where to Get | Cost |
|-----|-------------|------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | **Free** |
| `DEEPGRAM_API_KEY` | [console.deepgram.com](https://console.deepgram.com) | **$200 free credits** (optional — app uses gTTS fallback) |

---

## 🚀 Deployment Options

### 1. Hugging Face Spaces (Recommended — Free)
```bash
# 1. Create a new Space at huggingface.co/new-space
# 2. Choose Gradio SDK
# 3. Push this repo
# 4. Add secrets: GROQ_API_KEY and DEEPGRAM_API_KEY
git push huggingface main
```

### 2. Render.com (Free Tier)
```bash
# 1. Create new Web Service on render.com
# 2. Connect GitHub repo
# 3. Build: pip install -r requirements.txt
# 4. Start: python main.py
# 5. Add env vars: GROQ_API_KEY, DEEPGRAM_API_KEY
```

### 3. Local Development
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Fill in .env file
cp sample.env .env
# Edit .env and add your GROQ_API_KEY

# Run
python main.py
# Open: http://localhost:7860
```

---

## 🏗️ Architecture

```
User (Browser)
    │
    ├── 🎤 Voice → Gradio Audio Component → audio.wav/mp3
    │                                          │
    │                                 Groq Whisper API
    │                                          │
    │                              Patient Text (transcribed)
    │                                          │
    ├── 🖼️ Image → Gradio Image Component → image.jpg
    │                                          │
    │                              Groq Vision LLM (Llama 4)
    │                         (text + image → structured response)
    │                                          │
    │                              Parse: Assessment, Severity,
    │                              Confidence, Recommendation
    │                                          │
    │                         Deepgram TTS / gTTS fallback
    │                                          │
    └── 🔊 Doctor Audio + Text Response ◄──────┘
```

---

## ⚠️ Medical Disclaimer

This tool provides **general informational guidance only**. It is **NOT a medical diagnosis** and cannot replace evaluation by a licensed healthcare professional. For emergencies, call **112** (India) or **911** (US) immediately.

---

## 📦 Tech Stack

- **Groq** — Whisper STT + LLaMA 4 Vision LLM (free tier)
- **Deepgram** — Neural TTS (free $200 credit)
- **gTTS** — Free Google TTS fallback
- **Gradio** — Web UI framework
- **Pillow** — Image preprocessing
- **Python-dotenv** — Environment management

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

> **Multi-specialty AI health consultation platform** powered by Google Gemini 2.5 Flash, Groq Whisper STT, and Deepgram TTS.

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces)
[![Gemini](https://img.shields.io/badge/VLM-Gemini%202.5%20Flash-4285F4)](https://aistudio.google.com)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq%20AI-orange)](https://groq.com)
[![Free Tier](https://img.shields.io/badge/APIs-Free%20Tier-green)](.)

---

## 🌟 Features

| Feature | Details |
|---------|---------|
| **7 Specialties** | Skin · Dental · Eye · Heart · Bones · Mental Health · General |
| **Voice Input & STT** | Browser microphone / file upload via **Groq Whisper Large v3** |
| **Multimodal Vision & Video** | **Google Gemini 2.5 Flash** for image & video diagnosis |
| **Text-to-Speech** | **Deepgram Aura-2** (neural voice) with **gTTS** fallback (free) |
| **Severity Badge** | Color-coded Low / Medium / High urgency indicator |
| **Consultation History** | Last 5 sessions tracked with timestamps |
| **Download Report** | Export structured medical consultation summaries |
| **Fully Free Tier Compatible** | All core APIs run seamlessly on generous free tiers |

---

## 🔑 API Keys

| Key | Purpose | Where to Get | Cost |
|-----|---------|-------------|------|
| `GEMINI_API_KEY` | **Primary VLM** (Text, Image & Video AI Doctor) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | **Free** |
| `GROQ_API_KEY` | **Speech-to-Text** (Whisper) & Free Fallback LLM | [console.groq.com](https://console.groq.com) | **Free** |
| `DEEPGRAM_API_KEY` | **Text-to-Speech** (Neural Doctor Voice) | [console.deepgram.com](https://console.deepgram.com) | **$200 Free Credit** (gTTS fallback if blank) |

---

## 🚀 Local Setup & Running

```bash
# 1. Clone or navigate to folder
cd ai-skin-specialist

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fill API keys in .env
# GEMINI_API_KEY=...
# GROQ_API_KEY=...
# DEEPGRAM_API_KEY=...

# 5. Start app
python main.py
```

Open browser at: **http://localhost:7860**

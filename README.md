# 🩺 AI Multimodal Health Specialist & Clinical Triage Platform

> An intelligent, multi-specialty clinical triage platform engineered for low-latency assessment across **Voice (Speech-to-Text)**, **High-Resolution Computer Vision (VLM)**, **Dynamic Video Keyframe Analysis**, **Interactive Follow-Up Chat with Session Memory**, and **Neural Text-to-Speech (TTS)**.

[![Gradio](https://img.shields.io/badge/UI-Gradio%20v6-FF7C00?logo=gradio&logoColor=white)](https://gradio.app)
[![Gemini](https://img.shields.io/badge/VLM-Google%20Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white)](https://aistudio.google.com)
[![Groq](https://img.shields.io/badge/STT%20%26%20LLM-Groq%20LPU-F55036?logo=groq&logoColor=white)](https://groq.com)
[![Deepgram](https://img.shields.io/badge/TTS-Deepgram%20Aura--2-13EF93?logo=deepgram&logoColor=black)](https://deepgram.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Free Tier Compatible](https://img.shields.io/badge/APIs-100%25%20Free%20Tier-success)](.)

---

## 🌟 Key Features

| Feature | Technical Implementation | Highlights |
|---|---|---|
| **🎙️ Voice Speech-to-Text** | **Groq Whisper Large v3** | Sub-second audio transcription (~0.8s) with silent hallucination filtering. |
| **📸 Vision Photo Diagnosis** | **Google Gemini 2.5 Flash** | Clinical lesion, rash, and swelling inspection via high-resolution image analysis. |
| **🎬 Video Keyframe Analysis** | **OpenCV + Gemini VLM** | Samples 4–6 temporal frames across video footage for zero-delay visual movement evaluation. |
| **💬 Interactive Follow-Up Chat** | **Multi-Turn Session Memory** | In-memory conversation state (`gr.State`) preserving case history across Q&A turns. |
| **🔊 Neural Doctor Speech** | **Deepgram Aura-2** | Natural spoken audio responses with automatic **gTTS** free fallback. |
| **📄 1-Click Clinical Report** | **Automated Report Builder** | Exports comprehensive clinical consultation documentation to `.txt` with Reference ID. |
| **🛡️ Resilient AI Fallback** | **Dual-Provider Architecture** | Gemini 2.5 Flash automatically falls back to **Groq LLaMA 3.3 70B & 3.2 Vision**. |

---

## 🩺 7 Supported Medical Specialties

The platform features specialized AI doctor personas with dedicated clinical prompts:

1. **🩺 Dermatology** — Skin lesions, rashes, acne, burns, and mole evaluations.
2. **🦷 Dentistry** — Toothaches, gum bleeding, cavities, and oral hygiene.
3. **👁️ Ophthalmology** — Eye redness, blurriness, floaters, and visual strain.
4. **🫀 Cardiology** — Palpitations, shortness of breath, chest discomfort triage.
5. **🦴 Orthopedics** — Joint pain, swelling, sprains, and mobility recovery.
6. **🧠 Mental Health** — Stress, anxiety, sleep disturbances, and empathetic counseling.
7. **🏥 General Medicine** — Fever, cold/flu, headaches, fatigue, and general health triage.

---

## 🏗️ System Architecture

```
                                  [ Patient Input ]
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
             🎤 Voice Audio        📸 Clinical Photo     🎬 Dynamic Video
                    │                     │                     │
                    ▼                     ▼                     ▼
          Groq Whisper Large v3     PIL Resizer (≤1024)   OpenCV Keyframe Sampler
           (Sub-second STT)         (Inline Image Part)   (4-6 Temporal Frames)
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ▼
                         [ Google Gemini 2.5 Flash VLM ]
                         (Primary Multimodal Diagnostic)
                                          │
                               (If Rate-Limited / Quota)
                                          ▼
                         [ Groq LLaMA 3.3 70B / 3.2 Vision ]
                                 (Free Fallback LLM)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
          🔬 Clinical Assessment    🔊 Deepgram Aura-2    📄 Official Report Export
          & Triage Severity Badge    (Doctor Speech TTS)   (1-Click .txt Download)
                    │
                    ▼
          [ 💬 Interactive Follow-Up Chat ]
          (Multi-Turn In-Memory Session State)
```

> 📖 **Detailed Architecture Docs**: For complete component breakdown, data schemas, and pipeline diagrams, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 🚀 Quickstart & Local Setup

### 1. Clone Repository & Setup Virtual Environment
```powershell
git clone https://github.com/your-username/ai-health-specialist.git
cd ai-health-specialist

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On macOS/Linux
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root folder:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=llama-3.2-11b-vision-preview
WHISPER_MODEL=whisper-large-v3

DEEPGRAM_API_KEY=your_deepgram_api_key_here
DEEPGRAM_TTS_MODEL=aura-2-thalia-en
```

### 4. Run Application
```powershell
python main.py
```
Open **`http://localhost:7860`** in your browser!

---

## 🌐 Free Deployment Guide

### Option 1: Hugging Face Spaces (Recommended — 100% Free & Native)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name: `ai-health-specialist`.
3. Choose **Gradio** SDK.
4. Clone the Space repo or push your local repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-health-specialist
   git push space main
   ```
5. In your Space **Settings → Variables and Secrets**, add:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `DEEPGRAM_API_KEY` (optional)
6. Your app is live with a permanent public link!

---

### Option 2: Render.com (Free Web Service)

1. Push your repository to **GitHub**.
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New Web Service**.
3. Connect your GitHub repository.
4. Set:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
5. In **Environment Variables**, add `GEMINI_API_KEY`, `GROQ_API_KEY`, `DEEPGRAM_API_KEY`.
6. Deploy!

---

## 📁 Repository Structure

```
ai-skin-specialist/
├── main.py                     # Gradio UI, application orchestration & report pipeline
├── brain_of_the_doctor_groq.py # Multimodal VLM engine, video frame sampler & follow-up chat
├── voice_of_the_patient.py     # Groq Whisper Large v3 STT & silent artifact filtering
├── voice_of_the_doctor.py      # Deepgram Aura-2 neural TTS with gTTS fallback
├── requirements.txt            # Python dependencies (Gradio, Gemini, Groq, Deepgram, OpenCV)
├── Procfile                    # Render / cloud deployment process definition
├── ARCHITECTURE.md             # System architecture, Mermaid diagrams & data flow specs
├── README.md                   # Project documentation & deployment guides
├── .gitignore                  # Excludes .env, local audio, reports, and temp caches
└── .env                        # Local API secrets (NEVER committed to git)
```

---

## ⚠️ Medical & Legal Disclaimer

> **IMPORTANT**: This application provides AI-generated informational triage guidance based on self-reported symptoms and user-provided media. It is **NOT** a certified medical diagnosis and cannot replace evaluation by a licensed healthcare professional. If you or someone around you is experiencing life-threatening emergencies (e.g., chest pain, difficulty breathing, severe bleeding, or stroke symptoms), contact emergency services immediately (**112 / 911**).

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

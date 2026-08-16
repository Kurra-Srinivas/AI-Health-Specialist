"""
main.py — AI Health Specialist
================================
Multi-specialty AI health consultation platform.
Supports: Skin · Dental · Eye · Heart · Bones · Mental Health · General Medicine

Deployment targets:
  • Hugging Face Spaces (primary — free, perfect for Gradio)
  • Render.com (secondary — free tier)
  • Local development
"""

import datetime
import logging
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from brain_of_the_doctor_groq import (
    SPECIALTY_PROMPTS,
    brain_of_the_doctor,
    parse_doctor_response,
)
from voice_of_the_doctor import convert_text_to_doctor_audio
from voice_of_the_patient import transcribe_patient_voice

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "AI Health Specialist"
MAX_HISTORY = 5

# ============================================================
# CSS — Premium Dark-themed Design
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:wght,FILL@100..700,0..1&display=swap');

:root {
  --bg:           #0a0e1a;
  --bg-surface:   #111827;
  --bg-card:      #1a2234;
  --bg-input:     #1e2a3a;
  --border:       rgba(99,120,180,0.18);
  --border-focus: #4f8ef7;
  --text:         #e8edf8;
  --text-muted:   #7a8aaa;
  --text-label:   #9aabcc;
  --primary:      #4f8ef7;
  --primary-glow: rgba(79,142,247,0.35);
  --primary-dark: #2563eb;
  --accent:       #a78bfa;
  --success:      #34d399;
  --warning:      #fbbf24;
  --danger:       #f87171;
  --radius:       14px;
  --radius-sm:    8px;
  --shadow:       0 8px 32px rgba(0,0,0,0.45);
  --shadow-card:  0 4px 24px rgba(0,0,0,0.35);
}

/* ---- Base ---- */
body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gradio-container { max-width: 100% !important; }

/* ---- App Shell ---- */
.app-shell {
  max-width: 1320px;
  margin: 0 auto;
  padding: 24px 32px 48px;
}

/* ---- Header ---- */
.app-header {
  background: linear-gradient(135deg, #1a2234 0%, #0f1829 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 28px 36px;
  margin-bottom: 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: relative;
  overflow: hidden;
}
.app-header::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(79,142,247,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.header-brand { display: flex; align-items: center; gap: 16px; }
.header-icon {
  width: 52px; height: 52px;
  background: linear-gradient(135deg, #4f8ef7, #a78bfa);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px;
  box-shadow: 0 0 24px rgba(79,142,247,0.4);
}
.header-title { margin: 0; font-size: 26px; font-weight: 800; color: var(--text); letter-spacing: -0.02em; }
.header-subtitle { margin: 4px 0 0; font-size: 12px; color: var(--text-muted); font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }
.header-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.hbadge {
  background: rgba(79,142,247,0.12);
  border: 1px solid rgba(79,142,247,0.25);
  border-radius: 20px;
  padding: 5px 12px;
  font-size: 11px;
  font-weight: 600;
  color: #93b8fd;
  letter-spacing: 0.04em;
}

/* ---- Specialty Selector ---- */
.specialty-section {
  margin-bottom: 28px;
}
.specialty-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 14px;
  display: block;
}
.specialty-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 10px;
}
@media (max-width: 900px) {
  .specialty-grid { grid-template-columns: repeat(4, 1fr); }
  .app-shell { padding: 16px; }
  .main-grid { grid-template-columns: 1fr !important; }
  .header-badges { display: none; }
}
@media (max-width: 560px) {
  .specialty-grid { grid-template-columns: repeat(2, 1fr); }
}

/* ---- Main Grid ---- */
.main-grid {
  display: grid;
  grid-template-columns: minmax(0, 5fr) minmax(0, 7fr);
  gap: 22px;
  align-items: start;
}

/* ---- Cards ---- */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px;
  box-shadow: var(--shadow-card);
}
.card-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}
.card-title h2 {
  margin: 0;
  font-size: 17px;
  font-weight: 700;
  color: var(--text);
}
.card-icon {
  font-family: 'Material Symbols Rounded';
  font-size: 20px;
  color: var(--primary);
  font-variation-settings: 'FILL' 1, 'wght' 400;
}

/* ---- Section Divider ---- */
.section-divider {
  border: 0;
  border-top: 1px solid var(--border);
  margin: 18px 0;
}

/* ---- Severity Badge ---- */
.severity-container { margin-bottom: 18px; }
.severity-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.severity-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 20px;
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.severity-low    { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.severity-medium { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.severity-high   { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.confidence-badge {
  background: rgba(167,139,250,0.1);
  border: 1px solid rgba(167,139,250,0.25);
  border-radius: 20px;
  padding: 5px 12px;
  font-size: 11px;
  font-weight: 600;
  color: #c4b5fd;
  letter-spacing: 0.04em;
}

/* ---- Empty State ---- */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  gap: 14px;
  color: var(--text-muted);
  text-align: center;
  padding: 24px;
}
.empty-icon {
  font-family: 'Material Symbols Rounded';
  font-size: 54px;
  color: rgba(99,120,180,0.25);
  font-variation-settings: 'FILL' 0, 'wght' 300;
}
.empty-state strong { font-size: 16px; font-weight: 600; color: var(--text-label); display: block; margin-bottom: 4px; }
.empty-state p { font-size: 13px; margin: 0; max-width: 220px; line-height: 1.5; }

/* ---- Quick Symptom Chips ---- */
.chips-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 10px;
  display: block;
}

/* ---- Gradio component overrides ---- */
.gradio-container label,
.gradio-container .label-wrap span {
  color: var(--text-label) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: var(--bg-input) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 3px var(--primary-glow) !important;
}
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .form {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
.gradio-container .gradio-audio > div,
.gradio-container .gradio-image > div,
.gradio-container .gradio-video > div,
.gradio-container .gradio-textbox > div {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
/* Upload zones */
.gradio-container .upload-container,
.gradio-container .dropzone {
  background: var(--bg-input) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius-sm) !important;
  min-height: 130px !important;
}
.gradio-container .upload-container *,
.gradio-container .dropzone * {
  color: var(--text-muted) !important;
}
/* Textbox read content */
.gradio-container textarea:disabled,
.gradio-container [aria-disabled="true"] {
  background: var(--bg-input) !important;
  color: var(--text) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--text) !important;
}
/* Primary button */
.gr-button-primary, button[variant="primary"] {
  background: linear-gradient(135deg, #4f8ef7, #2563eb) !important;
  border: 0 !important;
  border-radius: 12px !important;
  color: #fff !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  min-height: 56px !important;
  box-shadow: 0 6px 20px rgba(79,142,247,0.35) !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.02em !important;
}
.gr-button-primary:hover { transform: translateY(-1px) !important; box-shadow: 0 10px 28px rgba(79,142,247,0.5) !important; }
/* Secondary button */
.gr-button-secondary, button[variant="secondary"] {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-label) !important;
  font-weight: 600 !important;
  transition: all 0.15s ease !important;
}
.gr-button-secondary:hover { border-color: var(--primary) !important; color: var(--primary) !important; background: rgba(79,142,247,0.06) !important; }

/* Block label overrides */
.gradio-container [data-testid="block-label"],
.gradio-container div[class*="block-label"] {
  background: var(--bg-card) !important;
  border-color: var(--border) !important;
  color: var(--text-label) !important;
}

/* Radio group for specialty selector */
.gradio-container .gradio-radio label {
  background: var(--bg-input) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 14px !important;
  transition: all 0.15s ease !important;
  cursor: pointer !important;
  color: var(--text-label) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  text-align: center !important;
  min-height: 52px !important;
}
.gradio-container .gradio-radio label:hover {
  border-color: var(--primary) !important;
  background: rgba(79,142,247,0.08) !important;
  color: var(--text) !important;
}
.gradio-container .gradio-radio input[type="radio"]:checked + label,
.gradio-container .gradio-radio label[data-checked="true"] {
  border-color: var(--primary) !important;
  background: linear-gradient(135deg, rgba(79,142,247,0.18), rgba(167,139,250,0.1)) !important;
  color: #93b8fd !important;
}
.gradio-container .gradio-radio input[type="radio"] { display: none !important; }

/* Accordion */
.gradio-container .gradio-accordion {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
.gradio-container .gradio-accordion > button {
  background: var(--bg-card) !important;
  color: var(--text-label) !important;
  padding: 16px 20px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
}

/* ---- Disclaimer ---- */
.disclaimer {
  margin-top: 28px;
  background: rgba(248,113,113,0.06);
  border: 1px solid rgba(248,113,113,0.2);
  border-radius: var(--radius);
  padding: 14px 20px;
  font-size: 12px;
  color: #fca5a5;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  line-height: 1.5;
}
.disclaimer .disc-icon {
  font-family: 'Material Symbols Rounded';
  font-size: 18px;
  flex-shrink: 0;
  margin-top: 1px;
  font-variation-settings: 'FILL' 1, 'wght' 500;
}

/* ---- Footer ---- */
.app-footer {
  margin-top: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
  color: var(--text-muted);
  padding: 14px 0;
  border-top: 1px solid var(--border);
}

/* ---- History ---- */
.history-entry {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 16px;
  margin-bottom: 10px;
  font-size: 12px;
  line-height: 1.5;
}
.history-entry .h-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.history-entry .h-spec { font-weight: 700; color: #93b8fd; font-size: 12px; }
.history-entry .h-time { color: var(--text-muted); font-size: 11px; }
.history-entry .h-excerpt { color: var(--text-label); }

/* ---- Info Note ---- */
.info-note {
  background: rgba(79,142,247,0.08);
  border: 1px solid rgba(79,142,247,0.2);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  font-size: 12px;
  color: #93b8fd;
  line-height: 1.5;
  display: flex;
  gap: 8px;
  align-items: flex-start;
}
.info-note .ni-icon {
  font-family: 'Material Symbols Rounded';
  font-size: 16px;
  flex-shrink: 0;
  font-variation-settings: 'FILL' 1, 'wght' 500;
  margin-top: 1px;
}

/* Gradio scrollbars */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
"""

# ============================================================
# Helper Renderers
# ============================================================

SEVERITY_CLASS = {"Low": "severity-low", "Medium": "severity-medium", "High": "severity-high"}
SEVERITY_DOT = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


def render_severity_badges(severity: str, confidence: str) -> str:
    sev_cls = SEVERITY_CLASS.get(severity, "severity-medium")
    dot = SEVERITY_DOT.get(severity, "🟡")
    return f"""
    <div class="severity-container">
      <div class="severity-row">
        <span class="severity-badge {sev_cls}">{dot} Severity: {severity}</span>
        <span class="confidence-badge">🎯 Confidence: {confidence}</span>
      </div>
    </div>
    """


def render_history_html(history: list) -> str:
    if not history:
        return """
        <div class="empty-state" style="min-height:100px">
          <span style="color:var(--text-muted);font-size:13px">No consultations yet.</span>
        </div>
        """
    html = ""
    for entry in reversed(history[-MAX_HISTORY:]):
        spec_label = SPECIALTY_PROMPTS.get(entry["specialty"], {}).get("label", "🏥")
        html += f"""
        <div class="history-entry">
          <div class="h-header">
            <span class="h-spec">{spec_label} {entry['specialty_name']}</span>
            <span class="h-time">{entry['timestamp']}</span>
          </div>
          <div class="h-excerpt">{entry['excerpt']}</div>
          <div style="margin-top:6px">
            {render_severity_badges(entry['severity'], entry['confidence'])}
          </div>
        </div>
        """
    return html


def generate_report(
    specialty: str,
    patient_text: str,
    assessment: str,
    severity: str,
    confidence: str,
    recommendation: str,
    full_text: str,
) -> str:
    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"╔══════════════════════════════════════════╗\n"
        f"       AI HEALTH SPECIALIST — REPORT\n"
        f"╚══════════════════════════════════════════╝\n\n"
        f"Date & Time   : {now}\n"
        f"Specialty     : {spec['label']} {spec['name']}\n"
        f"Severity      : {severity}\n"
        f"AI Confidence : {confidence}\n\n"
        f"──────────────────────────────────────────\n"
        f"PATIENT DESCRIPTION\n"
        f"──────────────────────────────────────────\n"
        f"{patient_text}\n\n"
        f"──────────────────────────────────────────\n"
        f"ASSESSMENT\n"
        f"──────────────────────────────────────────\n"
        f"{assessment}\n\n"
        f"──────────────────────────────────────────\n"
        f"RECOMMENDATION\n"
        f"──────────────────────────────────────────\n"
        f"{recommendation}\n\n"
        f"══════════════════════════════════════════\n"
        f"⚠  DISCLAIMER: This is AI-generated triage\n"
        f"   guidance, NOT a medical diagnosis.\n"
        f"   Always consult a licensed clinician.\n"
        f"══════════════════════════════════════════\n"
    )


# ============================================================
# Core Processing Function
# ============================================================

def process_consultation(
    audio_filepath,
    text_desc: str,
    image_filepath,
    video_filepath,
    specialty: str,
    history: list,
):
    """
    Full consultation pipeline:
      1. Transcribe audio  → Groq Whisper STT (if voice provided)
      2. Combine voice transcription + typed symptom text
      3. AI Analysis:
           • Image/Video/Text → Google Gemini 2.5 Flash (Multimodal VLM)
           • Fallback         → Groq LLaMA (LLaMA 3.3 70B text / 3.2 11B Vision)
      4. Generate TTS     → Deepgram neural / gTTS fallback
      5. Parse structured response and render UI
      6. Append to session history
    """
    patient_text = ""

    # Step 1: Transcribe audio if provided
    if audio_filepath:
        try:
            patient_text = transcribe_patient_voice(audio_filepath)
            if patient_text:
                logger.info(f"Transcribed: {patient_text[:60]}...")
        except Exception as exc:
            logger.warning(f"Voice transcription error: {exc}")

    # Step 2: Merge or fallback to typed symptom text
    typed_text = (text_desc or "").strip()
    if patient_text and typed_text and typed_text.lower() not in patient_text.lower():
        patient_text = f"{patient_text}. Additional symptoms: {typed_text}"
    elif not patient_text and typed_text:
        patient_text = typed_text

    if not patient_text:
        raise gr.Error("Please speak into the microphone or type your symptoms in the text box.")

    try:
        # Step 3: AI Analysis
        raw_response = brain_of_the_doctor(
            patient_text=patient_text,
            image_filepath=image_filepath,
            video_filepath=video_filepath,
            specialty=specialty,
        )
        parsed = parse_doctor_response(raw_response)
    except Exception as exc:
        raise gr.Error(f"AI analysis failed: {exc}")

    try:
        # Step 3: TTS
        doctor_audio = convert_text_to_doctor_audio(parsed["audio_response"])
    except Exception as exc:
        logger.warning(f"TTS failed: {exc}")
        doctor_audio = None

    # Step 4: Build UI outputs
    severity_html = render_severity_badges(parsed["severity"], parsed["confidence"])

    assessment_text = parsed["assessment"]
    recommendation_text = parsed["recommendation"]

    # Fallback if structured parsing failed
    if not assessment_text and not recommendation_text:
        assessment_text = raw_response
        recommendation_text = "Please consult a licensed clinician for a formal evaluation."

    # Report text
    report_text = generate_report(
        specialty=specialty,
        patient_text=patient_text,
        assessment=assessment_text,
        severity=parsed["severity"],
        confidence=parsed["confidence"],
        recommendation=recommendation_text,
        full_text=raw_response,
    )

    # Step 5: Update history
    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    history = list(history or [])
    history.append({
        "specialty": specialty,
        "specialty_name": spec["name"],
        "timestamp": datetime.datetime.now().strftime("%H:%M, %b %d"),
        "excerpt": patient_text[:80] + ("..." if len(patient_text) > 80 else ""),
        "severity": parsed["severity"],
        "confidence": parsed["confidence"],
    })
    history = history[-MAX_HISTORY:]
    history_html = render_history_html(history)

    audio_path = str(doctor_audio) if doctor_audio else None

    return (
        patient_text,         # transcript_box
        assessment_text,      # assessment_box
        recommendation_text,  # recommendation_box
        severity_html,        # severity_html_output
        audio_path,           # audio_output
        report_text,          # report_box
        history,              # history_state
        history_html,         # history_html_output
    )


def inject_symptom(symptom: str) -> str:
    return symptom


# ============================================================
# Gradio UI
# ============================================================

specialty_labels = [v["label"] for v in SPECIALTY_PROMPTS.values()]
specialty_keys = list(SPECIALTY_PROMPTS.keys())
specialty_label_to_key = {v["label"]: k for k, v in SPECIALTY_PROMPTS.items()}

with gr.Blocks(title=APP_TITLE) as demo:
    history_state = gr.State([])

    with gr.Column(elem_classes="app-shell"):
        # ── Header ──────────────────────────────────────────
        gr.HTML("""
        <header class="app-header">
          <div class="header-brand">
            <div class="header-icon">🩺</div>
            <div>
              <h1 class="header-title">AI Health Specialist</h1>
              <p class="header-subtitle">Multi-Specialty · Voice · Vision · AI</p>
            </div>
          </div>
          <div class="header-badges">
            <span class="hbadge">✨ Gemini 2.5 Flash</span>
            <span class="hbadge">⚡ Groq AI</span>
            <span class="hbadge">🎙️ Whisper STT</span>
            <span class="hbadge">🔊 Deepgram TTS</span>
          </div>
        </header>
        """)

        # ── Specialty Selector ───────────────────────────────
        gr.HTML('<span class="specialty-label">Choose Your Specialist</span>')
        with gr.Row(elem_classes="specialty-grid"):
            specialty_radio = gr.Radio(
                choices=specialty_labels,
                value=specialty_labels[0],
                label="",
                show_label=False,
                elem_id="specialty-selector",
            )

        # ── Main Two-Column Layout ───────────────────────────
        with gr.Row(elem_classes="main-grid"):

            # ── LEFT: Patient Input ──────────────────────────
            with gr.Column(scale=5):
                with gr.Column(elem_classes="card"):
                    gr.HTML("""
                    <div class="card-title">
                      <span class="card-icon">mic</span>
                      <h2>Patient Input</h2>
                    </div>
                    """)

                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Your Voice Description",
                        show_label=True,
                    )

                    with gr.Row():
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload Image (optional)",
                            height=200,
                        )
                        video_input = gr.Video(
                            label="Upload Video (optional)",
                            height=200,
                        )

                    # ── Quick Symptoms ──────────────────────
                    gr.HTML('<hr class="section-divider"><span class="chips-label">Quick Symptom Chips — click to use as voice prompt hint</span>')

                    with gr.Row():
                        chip1 = gr.Button("🔴 Rash / Redness", variant="secondary", size="sm")
                        chip2 = gr.Button("😣 Pain / Ache",     variant="secondary", size="sm")
                        chip3 = gr.Button("😰 Fatigue / Tired", variant="secondary", size="sm")

                    with gr.Row():
                        chip4 = gr.Button("🌡️ Fever / Chills",   variant="secondary", size="sm")
                        chip5 = gr.Button("😮 Swelling",          variant="secondary", size="sm")
                        chip6 = gr.Button("😟 Anxiety / Stress",  variant="secondary", size="sm")

                    chip_hint = gr.Textbox(
                        label="Symptom Description / Spoken Notes",
                        placeholder="Click a chip above, type your symptoms, or speak into the mic...",
                        interactive=True,
                        lines=2,
                    )

                    gr.HTML('<hr class="section-divider">')

                    analyze_btn = gr.Button(
                        "🔍  Analyze Concern",
                        variant="primary",
                        size="lg",
                    )

                    gr.HTML("""
                    <div class="info-note" style="margin-top:14px">
                      <span class="ni-icon">info</span>
                      <span>Record your voice describing your concern. Upload an image or video for visual analysis (recommended for Skin, Dental, Eye, Bones). Your data is processed securely and not stored.</span>
                    </div>
                    """)

            # ── RIGHT: Doctor Response ───────────────────────
            with gr.Column(scale=7):
                with gr.Column(elem_classes="card"):
                    gr.HTML("""
                    <div class="card-title">
                      <span class="card-icon">smart_toy</span>
                      <h2>Doctor's Response</h2>
                    </div>
                    """)

                    severity_html_output = gr.HTML("""
                    <div class="empty-state">
                      <span class="empty-icon">pending_actions</span>
                      <div>
                        <strong>Ready for Analysis</strong>
                        <p>Record your voice description and click Analyze to begin your consultation.</p>
                      </div>
                    </div>
                    """)

                    transcript_box = gr.Textbox(
                        label="📝 Your Transcribed Description",
                        lines=3,
                        interactive=False,
                        placeholder="Your spoken description will appear here...",
                    )

                    assessment_box = gr.Textbox(
                        label="🔬 Clinical Assessment",
                        lines=4,
                        interactive=False,
                        placeholder="AI assessment will appear here...",
                    )

                    recommendation_box = gr.Textbox(
                        label="💊 Recommendation",
                        lines=3,
                        interactive=False,
                        placeholder="Doctor's recommendation will appear here...",
                    )

                    audio_output = gr.Audio(
                        label="🔊 Doctor Voice Response",
                        type="filepath",
                        autoplay=True,
                    )

                    with gr.Accordion("📄 Download Full Report", open=False):
                        report_box = gr.Textbox(
                            label="Consultation Report",
                            lines=18,
                            interactive=False,
                            placeholder="Full structured report will appear here after analysis...",
                        )
                        gr.HTML("<p style='font-size:11px;color:var(--text-muted);margin-top:8px'>Copy the text above and save as a .txt file to keep a record of your consultation.</p>")

        # ── Consultation History ─────────────────────────────
        with gr.Accordion("🕐 Consultation History (Last 5 Sessions)", open=False, elem_classes="card"):
            history_html_output = gr.HTML("""
            <div class="empty-state" style="min-height:80px">
              <span style="color:var(--text-muted);font-size:13px">No consultations yet this session.</span>
            </div>
            """)

        # ── Disclaimer ───────────────────────────────────────
        gr.HTML("""
        <div class="disclaimer">
          <span class="disc-icon">warning</span>
          <span><strong>Medical Disclaimer:</strong> This tool provides general informational guidance only. It is NOT a diagnosis and cannot replace evaluation by a licensed healthcare professional. For emergencies — chest pain, difficulty breathing, severe bleeding, stroke symptoms — call emergency services immediately (112 / 911).</span>
        </div>
        """)

        # ── Footer ───────────────────────────────────────────
        gr.HTML("""
        <footer class="app-footer">
          <span>🩺 AI Health Specialist </span>
          <span>Powered by Google Gemini 2.5 Flash · Groq Whisper · Deepgram TTS · gTTS · Gradio</span>
        </footer>
        """)

    # ── Event Wiring ─────────────────────────────────────────

    def get_specialty_key(label: str) -> str:
        return specialty_label_to_key.get(label, "skin")

    analyze_btn.click(
        fn=lambda audio, txt, img, vid, spec_label, hist: process_consultation(
            audio, txt, img, vid, get_specialty_key(spec_label), hist
        ),
        inputs=[audio_input, chip_hint, image_input, video_input, specialty_radio, history_state],
        outputs=[
            transcript_box,
            assessment_box,
            recommendation_box,
            severity_html_output,
            audio_output,
            report_box,
            history_state,
            history_html_output,
        ],
        show_progress="full",
        concurrency_limit=4,
    )

    # Chip click handlers
    for chip, symptom in [
        (chip1, "I have a rash with redness on my skin"),
        (chip2, "I am experiencing pain or ache"),
        (chip3, "I feel very fatigued and tired"),
        (chip4, "I have a fever and chills"),
        (chip5, "I notice swelling in the affected area"),
        (chip6, "I am feeling anxious and stressed"),
    ]:
        chip.click(fn=lambda s=symptom: s, outputs=[chip_hint])


# ============================================================
# Launch Configuration
# ============================================================

# Enable queue BEFORE launch — critical for non-blocking API calls
# Without this, Groq STT + LLM + TTS (~15s) freezes the entire UI
demo.queue(max_size=10)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")

    logger.info(f"Starting AI Health Specialist on {server_name}:{port}")
    demo.launch(
        server_name=server_name,
        server_port=port,
        share=False,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(),
    )

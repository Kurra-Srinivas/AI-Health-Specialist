"""
brain_of_the_doctor_groq.py
----------------------------
Multi-specialty AI health analysis.

Architecture:
  PRIMARY VLM  : Google Gemini 2.5 Flash  — text + image + video
                 (gemini-2.5-flash-lite by default, free via Google AI Studio)
  FALLBACK LLM : Groq LLaMA               — free, text-only when Gemini fails
  STT          : Groq Whisper             — always (voice_of_the_patient.py)
  TTS          : Deepgram / gTTS          — always (voice_of_the_doctor.py)

Get your free Gemini API key at: https://aistudio.google.com/apikey
Get your free Groq API key at:   https://console.groq.com
"""

import base64
import logging
import mimetypes
import os
import re
import time
from io import BytesIO
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Specialty Configuration — 7 Doctor Personas
# ---------------------------------------------------------------------------

SPECIALTY_PROMPTS: dict[str, dict] = {
    "skin": {
        "name": "Dermatology",
        "label": "🩺 Skin",
        "system": (
            "You are Dr. Anika, a board-certified dermatologist with 15 years of experience. "
            "Analyze skin conditions from images and patient descriptions with clinical precision. "
            "Provide compassionate, clear guidance. Always recommend professional consultation for serious concerns."
        ),
        "needs_image": True,
        "example_symptoms": [
            "rash on my arm", "acne breakout", "dry flaky skin",
            "mole that changed color", "itchy red patches", "sunburn",
        ],
    },
    "dental": {
        "name": "Dentistry",
        "label": "🦷 Dental",
        "system": (
            "You are Dr. Priya, a licensed dentist with expertise in general and cosmetic dentistry. "
            "Analyze dental and oral health concerns from images and descriptions. "
            "Provide practical oral health guidance and refer for urgent cases."
        ),
        "needs_image": False,
        "example_symptoms": [
            "toothache", "bleeding gums", "tooth sensitivity",
            "cavity", "bad breath", "wisdom tooth pain",
        ],
    },
    "eye": {
        "name": "Ophthalmology",
        "label": "👁️ Eye",
        "system": (
            "You are Dr. Vikram, an expert ophthalmologist specializing in vision health. "
            "Analyze eye conditions from images and patient descriptions. "
            "Be precise about symptoms that require urgent emergency care."
        ),
        "needs_image": False,
        "example_symptoms": [
            "red eye", "blurry vision", "eye pain",
            "discharge from eye", "sensitivity to light", "floaters",
        ],
    },
    "cardiology": {
        "name": "Cardiology",
        "label": "🫀 Heart",
        "system": (
            "You are Dr. Suresh, a senior cardiologist. "
            "Assess cardiovascular symptoms from patient descriptions with careful clinical judgment. "
            "Always err on the side of caution — chest pain and palpitations may be emergencies. "
            "Immediately recommend emergency services if symptoms are severe."
        ),
        "needs_image": False,
        "example_symptoms": [
            "chest pain", "heart palpitations", "shortness of breath",
            "irregular heartbeat", "dizziness", "swollen ankles",
        ],
    },
    "orthopedics": {
        "name": "Orthopedics",
        "label": "🦴 Bones",
        "system": (
            "You are Dr. Rajiv, an orthopedic surgeon specializing in bone, joint, and muscle conditions. "
            "Analyze musculoskeletal concerns from patient descriptions and X-ray images if provided. "
            "Provide recovery guidance and referral recommendations."
        ),
        "needs_image": False,
        "example_symptoms": [
            "knee pain", "back pain", "joint swelling",
            "fracture concern", "sports injury", "arthritis pain",
        ],
    },
    "mental_health": {
        "name": "Mental Health",
        "label": "🧠 Mental",
        "system": (
            "You are Dr. Meera, a compassionate psychiatrist and mental health counselor. "
            "Assess mental wellness from patient descriptions with empathy and without judgment. "
            "Provide evidence-based coping strategies and refer to professional support when needed. "
            "Always check for crisis situations and provide helpline info if needed."
        ),
        "needs_image": False,
        "example_symptoms": [
            "feeling anxious", "trouble sleeping", "feeling depressed",
            "panic attacks", "stress at work", "mood swings",
        ],
    },
    "general": {
        "name": "General Medicine",
        "label": "🏥 General",
        "system": (
            "You are Dr. Kumar, a general practitioner with broad medical expertise. "
            "Assess general health concerns from patient descriptions and any images provided. "
            "Provide initial guidance and route to appropriate specialists."
        ),
        "needs_image": False,
        "example_symptoms": [
            "fever", "headache", "fatigue",
            "nausea", "cold symptoms", "body ache",
        ],
    },
}


# ---------------------------------------------------------------------------
# Structured Prompt Builder (shared by Gemini and Groq)
# ---------------------------------------------------------------------------

def _build_prompt(patient_text: str, specialty: str, has_video: bool = False, has_image: bool = False) -> str:
    """Build the comprehensive clinical output prompt for the LLM."""
    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    prompt = (
        f"You are {spec['name']} clinical specialist. "
        "Conduct a comprehensive, compassionate clinical assessment of the patient's presentation. "
        "Respond in the EXACT structured format below — follow the section headers strictly:\n\n"
        "ASSESSMENT: [3-4 detailed sentences: clinical evaluation of symptoms, observed visual/movement findings, and pathological possibilities]\n"
        "SEVERITY: [EXACTLY one of: Low | Medium | High]\n"
        "CONFIDENCE: [EXACTLY one of: Low | Medium | High]\n"
        "DIFFERENTIAL_FACTORS: [2-3 sentences: key potential contributing factors or conditions to consider]\n"
        "RECOMMENDATION: [2-3 sentences: clear actionable guidance, recommended home care/precautions, and suggested medical specialist follow-up]\n"
        "RED_FLAGS: [1-2 sentences: specific urgent warning signs that necessitate immediate emergency medical attention]\n"
        "AUDIO_RESPONSE: [2-3 natural, warm, empathetic spoken sentences for the voice audio player. "
        "NO asterisks, NO markdown, NO bullet points — pure natural spoken text only.]\n\n"
        "Medical context: Patient self-triage consultation. Provide thorough clinical reasoning.\n\n"
        f"Patient Description: {patient_text}"
    )
    if has_image:
        prompt += "\n\nVisual Context: A clinical photo has been provided. Incorporate specific visual observations (color, margins, texture, swelling, morphology) into the assessment."
    elif has_video:
        prompt += "\n\nVisual Context: Video frames are provided above. Analyze dynamic features, physical changes, or movement limitations shown in the footage."
    return prompt


# ---------------------------------------------------------------------------
# Video Frame Extraction (Direct visual input for Gemini & VLMs)
# ---------------------------------------------------------------------------

def extract_video_frames(filepath: str, max_frames: int = 6) -> list[Image.Image]:
    """
    Extract evenly-spaced representative frames from a video file as PIL Images.
    Allows direct, zero-delay visual analysis in Gemini without cloud upload errors.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed. Cannot extract video frames.")
        return []

    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, total_frames // max_frames)
    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((1024, 1024))
            frames.append(img)
        if len(frames) >= max_frames:
            break
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video: {filepath}")
    return frames


# ---------------------------------------------------------------------------
# PRIMARY: Gemini 2.5 Flash — text + image + video (Google AI Studio free tier)
# ---------------------------------------------------------------------------

def _call_gemini(
    patient_text: str,
    image_filepath: str | None,
    video_filepath: str | None,
    specialty: str,
) -> str:
    """
    PRIMARY provider: Google Gemini 2.5 Flash via google-generativeai SDK.

    Supports:
      • Text-only analysis
      • Image analysis (PIL inline — no upload needed)
      • Video analysis (Files API — handles large videos automatically)

    Free tier: https://aistudio.google.com/apikey
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file.\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    genai.configure(api_key=gemini_key)

    spec       = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    has_video  = bool(video_filepath and not image_filepath)
    has_image  = bool(image_filepath)
    prompt     = _build_prompt(patient_text, specialty, has_video=has_video, has_image=has_image)
    primary_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    candidate_models = [primary_model, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    # Deduplicate while preserving order
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    # Build the content parts list
    parts: list = []

    # ── Image: pass as PIL Image (inline, no upload needed) ──────────────
    if image_filepath:
        try:
            img = Image.open(image_filepath)
            img.thumbnail((1024, 1024))
            img = img.convert("RGB")
            parts.append(img)
            logger.info(f"Image attached to Gemini request ({img.size}).")
        except Exception as exc:
            logger.warning(f"Image load failed ({exc}) — sending text only to Gemini.")

    # ── Video: extract frames and pass directly as visual PIL images ─────
    elif video_filepath:
        try:
            frames = extract_video_frames(video_filepath, max_frames=6)
            if frames:
                parts.extend(frames)
                logger.info(f"Attached {len(frames)} video frames to Gemini request.")
            else:
                logger.warning("No video frames could be decoded — proceeding text only.")
        except Exception as exc:
            logger.warning(f"Video frame extraction failed ({exc}) — sending text only to Gemini.")

    # ── Text prompt always appended last ─────────────────────────────────
    parts.append(prompt)

    last_err = None
    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name=m_name,
                system_instruction=spec["system"],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=1024,
                ),
            )
            response = model.generate_content(parts)
            result = response.text
            logger.info(f"Gemini {m_name} response: {len(result)} chars")
            return result
        except Exception as err:
            logger.warning(f"Gemini model {m_name} failed: {err}. Trying next candidate...")
            last_err = err

    raise last_err or RuntimeError("All Gemini model candidates failed.")


# ---------------------------------------------------------------------------
# FALLBACK: Groq LLM — text-only (free tier, no vision)
# ---------------------------------------------------------------------------

def _call_groq_text(patient_text: str, specialty: str) -> str:
    """
    FALLBACK: Groq LLaMA 3.3 70B (text-only, free tier).
    Used when Gemini is unavailable or API key is missing.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Get a free key at: https://console.groq.com"
        )

    spec   = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    prompt = _build_prompt(patient_text, specialty)
    model  = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    client   = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        temperature=0.4,
        messages=[
            {"role": "system", "content": spec["system"]},
            {"role": "user",   "content": prompt},
        ],
    )

    result = response.choices[0].message.content
    logger.info(f"Groq fallback ({model}): {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# FALLBACK: Groq Vision — image + text (free tier)
# ---------------------------------------------------------------------------

def _call_groq_vision(patient_text: str, image_filepath: str, specialty: str) -> str:
    """
    FALLBACK vision: Groq LLaMA 3.2 Vision (free tier).
    Used when Gemini fails and an image was uploaded.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise ValueError("GROQ_API_KEY is missing.")

    spec   = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    prompt = _build_prompt(patient_text, specialty, has_image=True)
    model  = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")

    # Encode image to base64
    img = Image.open(image_filepath)
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ]

    client   = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        temperature=0.4,
        messages=[
            {"role": "system", "content": spec["system"]},
            {"role": "user",   "content": user_content},
        ],
    )

    result = response.choices[0].message.content
    logger.info(f"Groq Vision fallback ({model}): {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Public API: brain_of_the_doctor()
# ---------------------------------------------------------------------------

def brain_of_the_doctor(
    patient_text: str,
    image_filepath: str | None = None,
    video_filepath: str | None = None,
    specialty: str = "skin",
) -> str:
    """
    Analyze patient input using Gemini 2.5 Flash (primary) with Groq as fallback.

    Routing:
      Image input  → Gemini 2.5 Flash (inline PIL)
                     └─ fails → Groq Vision (llama-3.2-11b-vision-preview)
                     └─ fails → Groq text + image note
      Video input  → Gemini 2.5 Flash (Files API upload)
                     └─ fails → Groq text + video note
      Text only    → Gemini 2.5 Flash
                     └─ fails → Groq LLM (llama-3.3-70b-versatile)

    Args:
        patient_text:    Transcribed patient voice description.
        image_filepath:  Optional path to uploaded image.
        video_filepath:  Optional path to uploaded video.
        specialty:       Key from SPECIALTY_PROMPTS.

    Returns:
        Structured doctor response string.
    """
    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])

    # ── IMAGE ───────────────────────────────────────────────────────────────
    if image_filepath:
        logger.info(f"[{spec['name']}] Image → Gemini 2.5 Flash (primary)")
        try:
            return _call_gemini(patient_text, image_filepath, None, specialty)
        except Exception as gem_err:
            logger.warning(f"Gemini failed ({gem_err}). Trying Groq Vision...")
            try:
                return _call_groq_vision(patient_text, image_filepath, specialty)
            except Exception as gv_err:
                logger.warning(f"Groq Vision failed ({gv_err}). Falling back to Groq text.")
                augmented = (
                    f"{patient_text}\n\n"
                    "[Note: Patient uploaded an image that could not be analyzed by the "
                    "vision model. Please assess based on the verbal description only.]"
                )
                return _call_groq_text(augmented, specialty)

    # ── VIDEO ───────────────────────────────────────────────────────────────
    if video_filepath:
        logger.info(f"[{spec['name']}] Video → Gemini 2.5 Flash (Files API)")
        try:
            return _call_gemini(patient_text, None, video_filepath, specialty)
        except Exception as gem_err:
            logger.warning(f"Gemini video failed ({gem_err}). Falling back to Groq text.")
            augmented = (
                f"{patient_text}\n\n"
                "[Note: Patient uploaded a video that could not be analyzed. "
                "Please assess based on the verbal description only.]"
            )
            return _call_groq_text(augmented, specialty)

    # ── TEXT ONLY ───────────────────────────────────────────────────────────
    logger.info(f"[{spec['name']}] Text only → Gemini 2.5 Flash (primary)")
    try:
        return _call_gemini(patient_text, None, None, specialty)
    except Exception as gem_err:
        logger.warning(f"Gemini failed ({gem_err}). Falling back to Groq LLM...")
        try:
            return _call_groq_text(patient_text, specialty)
        except Exception as groq_err:
            raise RuntimeError(
                f"Both Gemini and Groq failed.\n"
                f"  Gemini error : {gem_err}\n"
                f"  Groq error   : {groq_err}\n"
                "Check your API keys."
            ) from groq_err


# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------

def parse_doctor_response(raw: str) -> dict:
    """
    Parse structured LLM response into named fields.

    Returns dict with keys:
        assessment, severity, confidence, differential_factors,
        recommendation, red_flags, audio_response, full_text
    """
    result = {
        "assessment":           "",
        "severity":             "Medium",
        "confidence":           "Medium",
        "differential_factors": "Follow regular clinical precautions.",
        "recommendation":       "",
        "red_flags":            "Sudden severe pain, shortness of breath, or rapidly spreading symptoms require urgent emergency care.",
        "audio_response":       "",
        "full_text":            raw,
    }

    patterns = {
        "assessment":           r"ASSESSMENT:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "severity":             r"SEVERITY:\s*(Low|Medium|High)",
        "confidence":           r"CONFIDENCE:\s*(Low|Medium|High)",
        "differential_factors": r"DIFFERENTIAL_FACTORS:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "recommendation":       r"RECOMMENDATION:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "red_flags":            r"RED_FLAGS:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "audio_response":       r"AUDIO_RESPONSE:\s*(.+?)(?=\n[A-Z_]+:|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # Graceful fallbacks
    if not result["audio_response"]:
        result["audio_response"] = result["assessment"] or raw[:400]
    if not result["assessment"]:
        result["assessment"] = raw

    return result

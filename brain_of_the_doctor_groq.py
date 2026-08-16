"""
brain_of_the_doctor_groq.py
----------------------------
Multi-specialty AI health analysis.

VLM Routing Strategy:
  • Image / Video provided → MiniMax VLM  (MiniMax-M3, Anthropic-compatible API)
  • Text only              → Groq LLM     (LLaMA 4 Scout, fast & free)
  • Groq fails             → MiniMax LLM  (fallback for text-only)

APIs:
  MINIMAX_API_KEY   — MiniMax VLM (image + video + text)
  GROQ_API_KEY      — Groq Whisper STT + LLaMA 4 text analysis
"""

import base64
import logging
import mimetypes
import os
import re
from io import BytesIO
from pathlib import Path

import anthropic
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
# Structured Prompt Template (shared by both Groq and MiniMax)
# ---------------------------------------------------------------------------

def _build_prompt(patient_text: str, specialty: str, has_video: bool = False) -> str:
    """Build the structured output prompt for the LLM."""
    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    prompt = (
        f"You are a {spec['name']} specialist. "
        "Respond in the EXACT structured format below — no deviations:\n\n"
        "ASSESSMENT: [2-3 sentences: clinical assessment of what you observe/hear]\n"
        "SEVERITY: [EXACTLY one of: Low | Medium | High]\n"
        "CONFIDENCE: [EXACTLY one of: Low | Medium | High]\n"
        "RECOMMENDATION: [1-2 sentences: what the patient should do next, including urgency]\n"
        "AUDIO_RESPONSE: [2-3 sentences: natural spoken response for text-to-speech. "
        "NO asterisks, NO bullets, NO markdown, NO special characters — plain text only.]\n\n"
        "Medical context: Patient is self-reporting. This is triage guidance, not a diagnosis.\n\n"
        f"Patient says: {patient_text}"
    )
    if has_video:
        prompt += (
            "\n\nNote: Patient has uploaded a video. Analyze visible details carefully "
            "and note what you can observe from the video footage."
        )
    return prompt


# ---------------------------------------------------------------------------
# Image Encoding
# ---------------------------------------------------------------------------

def _encode_image_for_api(filepath: str, max_size: int = 1024) -> tuple[str, str]:
    """
    Resize image and base64-encode it for API submission.

    Returns:
        (base64_string, media_type)
    """
    image = Image.open(filepath)
    image.thumbnail((max_size, max_size))

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    logger.info(f"Image encoded: {len(encoded):,} chars after resize to ≤{max_size}px")
    return encoded, "image/jpeg"


def _encode_video_for_api(filepath: str, max_mb: int = 20) -> tuple[str, str]:
    """
    Base64-encode a video file for API submission.

    Returns:
        (base64_string, media_type)

    Raises:
        ValueError: If video exceeds max_mb size limit.
    """
    file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    if file_size_mb > max_mb:
        raise ValueError(
            f"Video file is {file_size_mb:.1f} MB — too large (limit: {max_mb} MB). "
            f"Please trim or compress the video first."
        )

    media_type, _ = mimetypes.guess_type(filepath)
    media_type = media_type or "video/mp4"

    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    logger.info(f"Video encoded: {file_size_mb:.1f} MB, type={media_type}")
    return encoded, media_type


# ---------------------------------------------------------------------------
# MiniMax VLM — Handles image + video + text via Anthropic-compatible API
# ---------------------------------------------------------------------------

def _call_minimax_vlm(
    patient_text: str,
    image_filepath: str | None,
    video_filepath: str | None,
    specialty: str,
) -> str:
    """
    Call MiniMax VLM (MiniMax-M3) via Anthropic-compatible API.
    Supports text, image, and video inputs.
    """
    minimax_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not minimax_key:
        raise ValueError(
            "MINIMAX_API_KEY is missing. Please fill it in your .env file.\n"
            "Sign up at: https://www.minimax.io"
        )

    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    has_video = bool(video_filepath and not image_filepath)
    prompt = _build_prompt(patient_text, specialty, has_video=has_video)

    # Build user message content list
    user_content: list[dict] = []

    # Attach image if provided
    if image_filepath:
        try:
            img_data, img_type = _encode_image_for_api(image_filepath)
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img_type,
                    "data": img_data,
                },
            })
            logger.info("Image attached for MiniMax VLM.")
        except Exception as exc:
            logger.warning(f"Could not encode image: {exc}. Proceeding text-only.")

    # Attach video if provided (and no image)
    elif video_filepath:
        try:
            vid_data, vid_type = _encode_video_for_api(video_filepath)
            user_content.append({
                "type": "video",
                "source": {
                    "type": "base64",
                    "media_type": vid_type,
                    "data": vid_data,
                },
            })
            logger.info("Video attached for MiniMax VLM.")
        except Exception as exc:
            logger.warning(f"Could not encode video: {exc}. Proceeding text-only.")

    # Always add the text prompt
    user_content.append({"type": "text", "text": prompt})

    client = anthropic.Anthropic(
        api_key=minimax_key,
        base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"),
    )

    response = client.messages.create(
        model=os.environ.get("MINIMAX_MODEL", "MiniMax-M3"),
        max_tokens=1024,
        temperature=0.4,
        system=spec["system"],
        messages=[{"role": "user", "content": user_content}],
    )

    result = response.content[0].text
    logger.info(f"MiniMax VLM response: {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Groq LLM — Text-only (fast, free tier)
# ---------------------------------------------------------------------------

def _call_groq_text_only(patient_text: str, specialty: str) -> str:
    """
    Call Groq LLaMA 4 for text-only consultations (no image/video).
    Fast and on the free tier.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please fill it in your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    prompt = _build_prompt(patient_text, specialty)

    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        max_completion_tokens=1024,
        temperature=0.4,
        messages=[
            {"role": "system", "content": spec["system"]},
            {"role": "user",   "content": prompt},
        ],
    )

    result = response.choices[0].message.content
    logger.info(f"Groq LLM response: {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Main Router — Public API
# ---------------------------------------------------------------------------

def brain_of_the_doctor(
    patient_text: str,
    image_filepath: str | None = None,
    video_filepath: str | None = None,
    specialty: str = "skin",
) -> str:
    """
    Smart routing:
      - Image or video provided → MiniMax VLM  (full multimodal understanding)
      - Text only               → Groq LLM     (fast, free)
      - Groq fails              → MiniMax LLM  (fallback)

    Args:
        patient_text:    Transcribed patient description.
        image_filepath:  Optional path to uploaded image.
        video_filepath:  Optional path to uploaded video.
        specialty:       One of the keys in SPECIALTY_PROMPTS.

    Returns:
        Structured doctor response string.
    """
    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    has_visual = bool(image_filepath or video_filepath)

    if has_visual:
        # Visual input → always use MiniMax VLM
        logger.info(f"[{spec['name']}] Visual input detected → MiniMax VLM")
        return _call_minimax_vlm(patient_text, image_filepath, video_filepath, specialty)
    else:
        # Text only → try Groq first (faster), fall back to MiniMax
        logger.info(f"[{spec['name']}] Text-only → Groq LLM")
        try:
            return _call_groq_text_only(patient_text, specialty)
        except Exception as groq_err:
            logger.warning(f"Groq failed: {groq_err}. Falling back to MiniMax LLM...")
            return _call_minimax_vlm(patient_text, None, None, specialty)


# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------

def parse_doctor_response(raw: str) -> dict:
    """
    Parse the structured LLM response into named fields.

    Returns a dict with keys:
        assessment, severity, confidence, recommendation, audio_response, full_text
    """
    result = {
        "assessment":     "",
        "severity":       "Medium",
        "confidence":     "Medium",
        "recommendation": "",
        "audio_response": "",
        "full_text":      raw,
    }

    patterns = {
        "assessment":     r"ASSESSMENT:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "severity":       r"SEVERITY:\s*(Low|Medium|High)",
        "confidence":     r"CONFIDENCE:\s*(Low|Medium|High)",
        "recommendation": r"RECOMMENDATION:\s*(.+?)(?=\n[A-Z_]+:|$)",
        "audio_response": r"AUDIO_RESPONSE:\s*(.+?)(?=\n[A-Z_]+:|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # Fallbacks
    if not result["audio_response"]:
        result["audio_response"] = result["assessment"] or raw[:400]
    if not result["assessment"]:
        result["assessment"] = raw

    return result

"""
brain_of_the_doctor_groq.py
----------------------------
Multi-specialty AI health analysis using Groq vision + chat models.

Specialties: Skin, Dental, Eye, Cardiology, Orthopedics, Mental Health, General
"""

import base64
import logging
import os
import re
from io import BytesIO

from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Specialty Configuration
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
            "mole that changed color", "itchy red patches", "sunburn"
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
            "cavity", "bad breath", "wisdom tooth pain"
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
            "discharge from eye", "sensitivity to light", "floaters"
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
            "irregular heartbeat", "dizziness", "swollen ankles"
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
            "fracture concern", "sports injury", "arthritis pain"
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
            "panic attacks", "stress at work", "mood swings"
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
            "nausea", "cold symptoms", "body ache"
        ],
    },
}


# ---------------------------------------------------------------------------
# Image Encoding
# ---------------------------------------------------------------------------

def encode_image_for_groq(filepath: str) -> str:
    """Resize and base64-encode image for Groq vision API."""
    image = Image.open(filepath)
    image.thumbnail((1024, 1024))

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    logger.info(f"Image encoded: {len(encoded)} chars")
    return encoded


# ---------------------------------------------------------------------------
# Main AI Brain Function
# ---------------------------------------------------------------------------

def brain_of_the_doctor(
    patient_text: str,
    image_filepath: str | None = None,
    video_filepath: str | None = None,
    specialty: str = "skin",
) -> str:
    """
    Analyze patient input using Groq LLM (with optional vision).

    Args:
        patient_text: Transcribed patient description.
        image_filepath: Optional path to uploaded image.
        video_filepath: Optional path to uploaded video (not processed by LLM).
        specialty: One of the keys in SPECIALTY_PROMPTS.

    Returns:
        Structured doctor response string.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please fill in your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    spec = SPECIALTY_PROMPTS.get(specialty, SPECIALTY_PROMPTS["general"])
    logger.info(f"Processing {spec['name']} consultation...")

    # Build structured prompt
    prompt = (
        f"You are {spec['name']} specialist. "
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

    if video_filepath and not image_filepath:
        prompt += (
            "\n\nNote: Patient uploaded a video but this model analyzes still images. "
            "Mention that you have reviewed their description carefully and recommend "
            "they upload a clear still image for better visual assessment."
        )

    # Build user message content
    user_content: list[dict] = [{"type": "text", "text": prompt}]

    if image_filepath:
        try:
            image_data = encode_image_for_groq(image_filepath)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            })
            logger.info("Image attached to Groq request.")
        except Exception as exc:
            logger.warning(f"Could not encode image: {exc}. Proceeding text-only.")

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        max_completion_tokens=1024,
        temperature=0.4,
        messages=[
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": user_content},
        ],
    )

    result = response.choices[0].message.content
    logger.info(f"Groq response received ({len(result)} chars).")
    return result


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
        "assessment": "",
        "severity": "Medium",
        "confidence": "Medium",
        "recommendation": "",
        "audio_response": "",
        "full_text": raw,
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

    # Fallback: use assessment as audio response if AUDIO_RESPONSE not found
    if not result["audio_response"]:
        result["audio_response"] = result["assessment"] or raw[:400]

    # Fallback: if assessment is empty, use full text
    if not result["assessment"]:
        result["assessment"] = raw

    return result

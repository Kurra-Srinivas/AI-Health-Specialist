"""
voice_of_the_patient.py
-----------------------
Handles patient voice transcription using Groq Whisper API.

NOTE: Audio RECORDING is handled by Gradio's browser component.
This module only handles the STT (speech-to-text) transcription
of the recorded/uploaded audio file path provided by Gradio.

Supported audio formats: wav, mp3, webm, m4a, ogg, flac, mp4
"""

import logging
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Common Whisper hallucinations produced when audio is silent or contains background noise
SILENT_HALLUCINATIONS = {
    "thank you",
    "thank you.",
    "thank you very much.",
    "thank you for watching.",
    "thanks for watching.",
    "thanks for watching!",
    "you",
    "you.",
    "bye.",
    "subscribe",
    ".",
    "...",
    "subtitles by the amara.org community",
}


def is_silence_hallucination(text: str) -> bool:
    """Check if transcribed text is a known Whisper artifact from silent audio."""
    cleaned = text.strip().lower()
    cleaned_no_punct = re.sub(r"[^\w\s]", "", cleaned).strip()
    if not cleaned or len(cleaned) <= 1:
        return True
    if cleaned in SILENT_HALLUCINATIONS or cleaned_no_punct in SILENT_HALLUCINATIONS:
        return True
    return False


def transcribe_patient_voice(audio_filepath: str) -> str:
    """
    Transcribe patient audio to text using Groq Whisper Large v3.

    Args:
        audio_filepath: Path to the audio file (wav, mp3, webm, m4a, ogg, etc.)
                        provided by Gradio's Audio component.

    Returns:
        Transcribed text string. Returns empty string if audio is silent.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please fill it in your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    if not audio_filepath:
        return ""

    audio_path = str(audio_filepath)
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        logger.warning("Audio file is empty or does not exist.")
        return ""

    logger.info(f"Transcribing audio: {audio_path}")

    client = Groq(api_key=groq_api_key)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=os.environ.get("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
            language="en",
        )

    text = transcription if isinstance(transcription, str) else transcription.text
    text = text.strip()

    if is_silence_hallucination(text):
        logger.info(f"Whisper output '{text}' recognized as silent audio artifact.")
        return ""

    logger.info(f"Transcription complete ({len(text)} chars): {text[:80]}...")
    return text
"""
voice_of_the_patient.py
-----------------------
Handles patient voice transcription using Groq Whisper API.

NOTE: Audio RECORDING is handled by Gradio's browser component.
This module only handles the STT (speech-to-text) transcription
of the recorded/uploaded audio file path provided by Gradio.
"""

import logging
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def transcribe_patient_voice(audio_filepath: str) -> str:
    """
    Transcribe patient audio to text using Groq Whisper.

    Args:
        audio_filepath: Path to the audio file (wav, mp3, webm, m4a, ogg, etc.)

    Returns:
        Transcribed text string.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
        FileNotFoundError: If the audio file doesn't exist.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please fill in your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    if not audio_filepath:
        raise ValueError("No audio file provided.")

    logger.info(f"Transcribing audio: {audio_filepath}")

    client = Groq(api_key=groq_api_key)
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=os.environ.get("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
        )

    # Groq returns either a string or an object depending on response_format
    text = transcription if isinstance(transcription, str) else transcription.text
    logger.info(f"Transcription complete: {text[:80]}...")
    return text
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
    Transcribe patient audio to text using Groq Whisper Large v3.

    Groq Whisper is the fastest Whisper implementation available —
    it transcribes a 1-minute audio clip in about 1 second.

    Args:
        audio_filepath: Path to the audio file (wav, mp3, webm, m4a, ogg, etc.)
                        provided by Gradio's Audio component.

    Returns:
        Transcribed text string.

    Raises:
        ValueError: If GROQ_API_KEY is not set, or no audio file provided.
        FileNotFoundError: If the audio file does not exist.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please fill it in your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    if not audio_filepath:
        raise ValueError("No audio file provided. Please record or upload your voice.")

    audio_path = str(audio_filepath)
    logger.info(f"Transcribing audio: {audio_path}")

    client = Groq(api_key=groq_api_key)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=os.environ.get("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
            # language="en",  # uncomment to force English for faster transcription
        )

    # Groq returns either a plain string or an object depending on response_format
    text = transcription if isinstance(transcription, str) else transcription.text
    text = text.strip()

    logger.info(f"Transcription complete ({len(text)} chars): {text[:80]}...")
    return text
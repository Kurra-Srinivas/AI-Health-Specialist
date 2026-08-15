"""
voice_of_the_doctor.py
----------------------
Converts doctor response text to speech audio.

Primary:  Deepgram TTS (SDK v7.7.0 — uses DEEPGRAM_API_KEY)
Fallback: gTTS / Google TTS (completely free, no API key needed)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCTOR_AUDIO = BASE_DIR / "doctor_response.mp3"


def convert_text_to_doctor_audio(
    text: str,
    output_filepath: Path = DEFAULT_DOCTOR_AUDIO,
) -> Path:
    """
    Convert text to speech. Tries Deepgram first; falls back to gTTS.

    Args:
        text: The text to convert to speech.
        output_filepath: Where to save the .mp3 file.

    Returns:
        Path to the generated audio file.
    """
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()

    if deepgram_key:
        try:
            logger.info("Using Deepgram TTS (SDK v7+)...")
            return _deepgram_tts(text, output_filepath, deepgram_key)
        except Exception as exc:
            logger.warning(f"Deepgram TTS failed ({exc}). Falling back to gTTS (free).")

    logger.info("Using gTTS (free Google TTS)...")
    return _gtts_tts(text, output_filepath)


# ---------------------------------------------------------------------------
# Deepgram TTS — SDK v7.7.0
# API: client.speak.v1.audio.generate(text=..., model=..., encoding=...)
# Returns: Iterator[bytes]
# ---------------------------------------------------------------------------

def _deepgram_tts(text: str, output_filepath: Path, api_key: str) -> Path:
    """Use Deepgram SDK v7.7.0 to generate TTS and save to file."""
    from deepgram import DeepgramClient  # type: ignore

    client = DeepgramClient(api_key=api_key)
    model = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")

    # SDK v7: speak.v1.audio.generate(**kwargs) → Iterator[bytes]
    audio_iter = client.speak.v1.audio.generate(
        text=text,
        model=model,
        encoding="mp3",
    )

    with output_filepath.open("wb") as f:
        for chunk in audio_iter:
            if chunk:
                f.write(chunk)

    logger.info(f"Deepgram audio saved: {output_filepath}")
    return output_filepath


# ---------------------------------------------------------------------------
# gTTS Fallback (completely free, no API key required)
# ---------------------------------------------------------------------------

def _gtts_tts(text: str, output_filepath: Path) -> Path:
    """Use Google Text-to-Speech (gTTS) — free, no API key needed."""
    from gtts import gTTS  # type: ignore

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(str(output_filepath))

    logger.info(f"gTTS audio saved: {output_filepath}")
    return output_filepath

"""
voice_of_the_doctor.py
----------------------
Converts doctor response text to speech audio.

TTS Priority:
  1. Deepgram TTS (SDK v7.7.0) — neural voice, high quality  [DEEPGRAM_API_KEY]
  2. gTTS / Google TTS         — free fallback, no key needed

Audio files are created with unique timestamps to prevent concurrent
user conflicts (multiple users generating audio at the same time).
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_responses"


def convert_text_to_doctor_audio(text: str) -> Path:
    """
    Convert text to speech and save to a unique timestamped audio file.

    Uses Deepgram if DEEPGRAM_API_KEY is set; otherwise uses gTTS (free).
    Creates unique filenames per request to support concurrent users.

    Args:
        text: The doctor's response text to convert to speech.

    Returns:
        Path to the generated .mp3 audio file.
    """
    # Create audio output directory if it doesn't exist
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Unique filename per request — prevents concurrent user conflicts
    timestamp = int(time.time() * 1000)  # millisecond precision
    output_filepath = AUDIO_DIR / f"doctor_response_{timestamp}.mp3"

    # Clean up old audio files (keep only last 20) to prevent disk bloat
    _cleanup_old_audio_files(AUDIO_DIR, keep_last=20)

    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()

    if deepgram_key:
        try:
            logger.info("Using Deepgram TTS (neural voice)...")
            return _deepgram_tts(text, output_filepath, deepgram_key)
        except Exception as exc:
            logger.warning(f"Deepgram TTS failed ({exc}). Falling back to gTTS.")

    logger.info("Using gTTS (free Google TTS)...")
    return _gtts_tts(text, output_filepath)


def _cleanup_old_audio_files(directory: Path, keep_last: int = 20) -> None:
    """Delete oldest audio response files, keeping only the most recent N."""
    try:
        files = sorted(directory.glob("doctor_response_*.mp3"), key=lambda f: f.stat().st_mtime)
        for old_file in files[:-keep_last]:
            old_file.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug(f"Audio cleanup skipped: {exc}")


# ---------------------------------------------------------------------------
# Deepgram TTS — SDK v7.7.0
# API: client.speak.v1.audio.generate(text=..., model=..., encoding=...)
# Returns: Iterator[bytes]
# ---------------------------------------------------------------------------

def _deepgram_tts(text: str, output_filepath: Path, api_key: str) -> Path:
    """Generate TTS using Deepgram SDK v7.7.0 and save to file."""
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

    logger.info(f"Deepgram audio saved: {output_filepath.name}")
    return output_filepath


# ---------------------------------------------------------------------------
# gTTS Fallback — completely free, no API key needed
# ---------------------------------------------------------------------------

def _gtts_tts(text: str, output_filepath: Path) -> Path:
    """Generate TTS using Google gTTS — free, no key required."""
    from gtts import gTTS  # type: ignore

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(str(output_filepath))

    logger.info(f"gTTS audio saved: {output_filepath.name}")
    return output_filepath

"""OpenAI-compatible speech-to-text server for Whisper and SenseVoice checkpoints."""

from .models import SenseVoiceTranscriber, Transcriber, load_transcriber
from .transcription import Segment, Transcription

__all__ = [
    "Segment",
    "SenseVoiceTranscriber",
    "Transcriber",
    "Transcription",
    "load_transcriber",
]
__version__ = "0.1.0"

"""OpenAI-compatible speech-to-text server for Whisper and SenseVoice checkpoints."""

from .sensevoice import SenseVoiceTranscriber
from .transcriber import Segment, Transcriber, Transcription

__all__ = ["Segment", "SenseVoiceTranscriber", "Transcriber", "Transcription"]
__version__ = "0.1.0"

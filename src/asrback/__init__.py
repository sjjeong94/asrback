"""OpenAI-compatible speech-to-text server for Whisper SafeTensors checkpoints."""

from .transcriber import Segment, Transcriber, Transcription

__all__ = ["Segment", "Transcriber", "Transcription"]
__version__ = "0.1.0"

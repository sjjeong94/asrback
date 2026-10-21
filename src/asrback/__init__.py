"""OpenAI-compatible speech-to-text server for Whisper, SenseVoice, Qwen3-ASR and
Gemma 4 checkpoints."""

from .models import (
    Gemma4Transcriber,
    Qwen3ASRTranscriber,
    SenseVoiceTranscriber,
    Transcriber,
    load_transcriber,
)
from .transcription import Segment, Transcription

__all__ = [
    "Gemma4Transcriber",
    "Qwen3ASRTranscriber",
    "Segment",
    "SenseVoiceTranscriber",
    "Transcriber",
    "Transcription",
    "load_transcriber",
]
__version__ = "0.1.0"

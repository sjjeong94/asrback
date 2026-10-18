"""OpenAI-compatible speech-to-text server for Whisper, SenseVoice and Qwen3-ASR
checkpoints."""

from .models import (
    Qwen3ASRTranscriber,
    SenseVoiceTranscriber,
    Transcriber,
    load_transcriber,
)
from .transcription import Segment, Transcription

__all__ = [
    "Qwen3ASRTranscriber",
    "Segment",
    "SenseVoiceTranscriber",
    "Transcriber",
    "Transcription",
    "load_transcriber",
]
__version__ = "0.1.0"

"""Speech recognition models served by asrback."""

from __future__ import annotations

import os

from .gemma4 import Gemma4Transcriber, is_gemma4
from .qwen3_asr import Qwen3ASRTranscriber, is_qwen3_asr
from .sensevoice import SenseVoiceTranscriber, is_sensevoice
from .whisper import Transcriber

__all__ = [
    "Gemma4Transcriber",
    "Qwen3ASRTranscriber",
    "SenseVoiceTranscriber",
    "Transcriber",
    "is_gemma4",
    "is_qwen3_asr",
    "is_sensevoice",
    "load_transcriber",
]


def load_transcriber(
    model_dir: str | os.PathLike,
    cuda_graphs: bool | None = None,
    compile_encoder: bool = False,
    condition_on_previous_text: bool | None = None,
    **options,
) -> Transcriber | SenseVoiceTranscriber | Qwen3ASRTranscriber | Gemma4Transcriber:
    """Load the Whisper, SenseVoice, Qwen3-ASR or Gemma 4 checkpoint in ``model_dir``.

    ``options`` (device, dtype, max_batch_size, batch_wait_ms, max_audio_seconds)
    apply to every model; ``compile_encoder`` to Whisper and SenseVoice; ``cuda_graphs`` to
    Whisper, Qwen3-ASR and Gemma 4; ``condition_on_previous_text`` to Whisper
    only. Options left as ``None`` take the model's default.
    """
    options = {name: value for name, value in options.items() if value is not None}
    if is_sensevoice(model_dir):
        if cuda_graphs or condition_on_previous_text is not None:
            raise ValueError(
                "--cuda-graphs and --condition-on-previous-text apply to Whisper, "
                "Qwen3-ASR and Gemma 4 models only"
            )
        return SenseVoiceTranscriber(
            model_dir, compile_encoder=compile_encoder, **options
        )
    if is_qwen3_asr(model_dir):
        if compile_encoder or condition_on_previous_text is not None:
            raise ValueError(
                "--compile-encoder and --condition-on-previous-text do not apply to "
                "Qwen3-ASR models"
            )
        return Qwen3ASRTranscriber(model_dir, cuda_graphs=cuda_graphs, **options)
    if is_gemma4(model_dir):
        if compile_encoder or condition_on_previous_text is not None:
            raise ValueError(
                "--compile-encoder and --condition-on-previous-text do not apply to "
                "Gemma 4 models"
            )
        return Gemma4Transcriber(model_dir, cuda_graphs=cuda_graphs, **options)
    if condition_on_previous_text is not None:
        options["condition_on_previous_text"] = condition_on_previous_text
    return Transcriber(
        model_dir, cuda_graphs=cuda_graphs, compile_encoder=compile_encoder, **options
    )

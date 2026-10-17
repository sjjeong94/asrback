"""Speech recognition models served by asrback."""

from __future__ import annotations

import os

from .sensevoice import SenseVoiceTranscriber, is_sensevoice
from .whisper import Transcriber

__all__ = ["SenseVoiceTranscriber", "Transcriber", "is_sensevoice", "load_transcriber"]


def load_transcriber(
    model_dir: str | os.PathLike,
    cuda_graphs: bool | None = None,
    compile_encoder: bool = False,
    **options,
) -> Transcriber | SenseVoiceTranscriber:
    """Load the Whisper or SenseVoice checkpoint in ``model_dir``.

    ``options`` (device, dtype, max_batch_size, batch_wait_ms) apply to both;
    ``cuda_graphs`` and ``compile_encoder`` to Whisper only. Options left as
    ``None`` take the model's default.
    """
    options = {name: value for name, value in options.items() if value is not None}
    if is_sensevoice(model_dir):
        if cuda_graphs or compile_encoder:
            raise ValueError(
                "--cuda-graphs and --compile-encoder apply to Whisper models only"
            )
        return SenseVoiceTranscriber(model_dir, **options)
    return Transcriber(
        model_dir, cuda_graphs=cuda_graphs, compile_encoder=compile_encoder, **options
    )

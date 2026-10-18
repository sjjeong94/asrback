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
    condition_on_previous_text: bool | None = None,
    **options,
) -> Transcriber | SenseVoiceTranscriber:
    """Load the Whisper or SenseVoice checkpoint in ``model_dir``.

    ``options`` (device, dtype, max_batch_size, batch_wait_ms) and
    ``compile_encoder`` apply to both; ``cuda_graphs`` and
    ``condition_on_previous_text`` to Whisper only. Options left as ``None`` take
    the model's default.
    """
    options = {name: value for name, value in options.items() if value is not None}
    if is_sensevoice(model_dir):
        if cuda_graphs or condition_on_previous_text is not None:
            raise ValueError(
                "--cuda-graphs and --condition-on-previous-text apply to Whisper "
                "models only"
            )
        return SenseVoiceTranscriber(
            model_dir, compile_encoder=compile_encoder, **options
        )
    if condition_on_previous_text is not None:
        options["condition_on_previous_text"] = condition_on_previous_text
    return Transcriber(
        model_dir, cuda_graphs=cuda_graphs, compile_encoder=compile_encoder, **options
    )

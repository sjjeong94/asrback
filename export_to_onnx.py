#!/usr/bin/env python3
"""Export OpenAI Whisper models to ONNX.

Examples:

    conda run -n whisper python export_to_onnx.py --model tiny

This exports Whisper's encoder graph. Whisper's encoder expects a fixed
30-second log-Mel input.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_OPSET = 23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Whisper models to ONNX.")
    parser.add_argument(
        "--model",
        default="tiny",
        help="Whisper model name or local checkpoint path. Example: tiny, base, large-v3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for ONNX files. Default: onnx/<sanitized-model-name>",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used during export. Default: cpu",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version. Default: {DEFAULT_OPSET}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for tracing and for the exported ONNX graph.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip onnx.checker validation after export.",
    )
    return parser.parse_args()


def import_whisper():
    try:
        import whisper

        return whisper
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'whisper' package is not installed in this Python environment.\n"
            "Install it with:\n"
            "  python -m pip install -U openai-whisper"
        ) from exc


def import_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'torch' package is not installed in this Python environment."
        ) from exc


def require_onnx() -> None:
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'onnx' package is not installed in this Python environment.\n"
            "Install it with:\n"
            "  python -m pip install -U onnx"
        ) from exc


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "whisper"


def default_output_dir(model_name: str) -> Path:
    return Path("onnx") / sanitize_name(model_name)


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def export_onnx(
    torch,
    module,
    inputs: tuple,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=True,
            external_data=False,
        )


def validate_onnx(output_paths: Iterable[Path]) -> None:
    try:
        import onnx
    except ModuleNotFoundError:
        print(
            "Skipping ONNX validation because the 'onnx' package is not installed.",
            file=sys.stderr,
        )
        return

    for output_path in output_paths:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)


def main() -> int:
    args = parse_args()
    require_positive("--batch-size", args.batch_size)

    torch = import_torch()
    whisper = import_whisper()
    require_onnx()

    output_dir = args.output_dir or default_output_dir(args.model)
    device = torch.device(args.device)

    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model, device=str(device))
    model.eval()

    dims = model.dims
    mel_frames = dims.n_audio_ctx * 2

    mel = torch.zeros(
        args.batch_size,
        dims.n_mels,
        mel_frames,
        dtype=torch.float32,
        device=device,
    )
    encoder_dynamic = None

    exported: list[Path] = []
    print(f"Output directory: {output_dir}")
    print(f"Opset: {args.opset}")

    encoder_path = output_dir / "encoder.onnx"
    print(f"Exporting encoder: {encoder_path}")
    export_onnx(
        torch=torch,
        module=model.encoder,
        inputs=(mel,),
        output_path=encoder_path,
        input_names=["mel"],
        output_names=["audio_features"],
        dynamic_axes=encoder_dynamic,
        opset=args.opset,
    )
    exported.append(encoder_path)

    if not args.no_check:
        validate_onnx(exported)

    print("Exported:")
    for output_path in exported:
        print(f"  {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

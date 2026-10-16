"""``asrback`` command: serve a Whisper checkpoint over the OpenAI audio API."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="asrback",
        description="Serve a Whisper model through OpenAI-compatible /v1/audio endpoints.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/whisper-tiny"),
        help="Directory containing model.safetensors and tokenizer/config JSON files.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model id reported by /v1/models. Default: the model directory name",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device. Default: cuda when available, otherwise cpu",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Default: 8000")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import uvicorn

    from .server import create_app
    from .transcriber import Transcriber

    try:
        transcriber = Transcriber(args.model_dir, device=args.device)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    model_name = args.model_name or args.model_dir.resolve().name
    print(f"Loaded {args.model_dir} on {transcriber.device} as {model_name!r}")
    uvicorn.run(create_app(transcriber, model_name), host=args.host, port=args.port)
    return 0

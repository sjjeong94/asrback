"""``asrback`` command: serve a Whisper or SenseVoice checkpoint over the OpenAI
audio API."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="asrback",
        description="Serve a Whisper or SenseVoice model through OpenAI-compatible "
        "/v1/audio endpoints.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/whisper-tiny"),
        help="Whisper directory (model.safetensors and tokenizer/config JSON files) "
        "or SenseVoice directory (model.pt, config.yaml, am.mvn, BPE model).",
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
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Model precision. Default: float16 on CUDA (bfloat16 for SenseVoice), "
        "otherwise float32",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Most concurrent requests decoded together. Default: 16",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=5.0,
        help="How long a batch waits to fill before decoding. Default: 5",
    )
    parser.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replay decoder steps from captured CUDA graphs. Default: on for CUDA",
    )
    parser.add_argument(
        "--compile-encoder",
        action="store_true",
        help="Compile the encoder with torch.compile (slower startup).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Default: 8000")
    args = parser.parse_args(argv)
    if args.max_batch_size <= 0:
        parser.error("--max-batch-size must be greater than zero")
    if args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    import uvicorn

    from .sensevoice import SenseVoiceTranscriber, is_sensevoice
    from .server import create_app
    from .transcriber import Transcriber

    # TF32 matmuls on Ampere+ GPUs; no effect on CPU or half precision.
    torch.set_float32_matmul_precision("high")
    options = {
        "device": args.device,
        "dtype": args.dtype,
        "max_batch_size": args.max_batch_size,
        "batch_wait_ms": args.batch_wait_ms,
    }
    try:
        if is_sensevoice(args.model_dir):
            if args.cuda_graphs or args.compile_encoder:
                raise ValueError(
                    "--cuda-graphs and --compile-encoder apply to Whisper models only"
                )
            transcriber = SenseVoiceTranscriber(args.model_dir, **options)
        else:
            transcriber = Transcriber(
                args.model_dir,
                cuda_graphs=args.cuda_graphs,
                compile_encoder=args.compile_encoder,
                **options,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    model_name = args.model_name or args.model_dir.resolve().name
    print(
        f"Loaded {args.model_dir} on {transcriber.device} "
        f"({str(transcriber.dtype).removeprefix('torch.')}, "
        f"batch {transcriber.max_batch_size}, "
        f"CUDA graphs {'on' if transcriber.cuda_graphs else 'off'}) as {model_name!r}"
    )
    uvicorn.run(create_app(transcriber, model_name), host=args.host, port=args.port)
    return 0

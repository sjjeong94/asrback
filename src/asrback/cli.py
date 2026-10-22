"""``asrback`` command: serve a Whisper, SenseVoice, Qwen3-ASR or Gemma 4 checkpoint
over the OpenAI audio API."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="asrback",
        description="Serve a Whisper, SenseVoice, Qwen3-ASR or Gemma 4 model through "
        "OpenAI-compatible /v1/audio endpoints.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/whisper-tiny"),
        help="Whisper directory (model.safetensors and tokenizer/config JSON files), "
        "SenseVoice directory (model.pt, config.yaml, am.mvn, BPE model) or "
        "Qwen3-ASR directory (safetensors, config JSON files, vocab.json, merges.txt) "
        "or Gemma 4 directory (model.safetensors, config JSON files, tokenizer.json).",
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
        choices=("float32", "bfloat16"),
        default=None,
        help="Model precision. Default: bfloat16 on CUDA, otherwise float32",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Most concurrent requests decoded together. "
        "Default: 16 for Whisper, 32 for SenseVoice, 64 for Qwen3-ASR "
        "(32 in float32), 32 for Gemma 4 (16 in float32)",
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
        help="Replay decoder steps from captured CUDA graphs (Whisper, Qwen3-ASR, "
        "Gemma 4). "
        "Default: on for CUDA",
    )
    parser.add_argument(
        "--compile-encoder",
        action="store_true",
        help="Compile the encoder with torch.compile (slower startup; Whisper, "
        "SenseVoice).",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Condition each 30-second window on the text before it. With "
        "--no-condition-on-previous-text, the windows of a long file are decoded "
        "together. Default: on",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=None,
        help="Reject (400) audio longer than this many seconds, before most of it "
        "is decoded. Default: no limit",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Answer 429 to audio requests while this many are in flight, before "
        "reading their upload. Default: no limit",
    )
    parser.add_argument(
        "--model-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the model in a child process, so that HTTP handling does not hold "
        "up decoding. Default: on",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Default: 8000")
    args = parser.parse_args(argv)
    if args.max_batch_size is not None and args.max_batch_size <= 0:
        parser.error("--max-batch-size must be greater than zero")
    if args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms must be non-negative")
    if args.max_audio_seconds is not None and args.max_audio_seconds <= 0:
        parser.error("--max-audio-seconds must be greater than zero")
    if args.max_concurrent_requests is not None and args.max_concurrent_requests <= 0:
        parser.error("--max-concurrent-requests must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    import uvicorn

    from .model_process import ProcessTranscriber
    from .models import load_transcriber
    from .server import create_app

    # TF32 matmuls on Ampere+ GPUs; no effect on CPU or half precision.
    torch.set_float32_matmul_precision("high")
    load = ProcessTranscriber if args.model_process else load_transcriber
    try:
        transcriber = load(
            args.model_dir,
            device=args.device,
            dtype=args.dtype,
            max_batch_size=args.max_batch_size,
            batch_wait_ms=args.batch_wait_ms,
            cuda_graphs=args.cuda_graphs,
            compile_encoder=args.compile_encoder,
            condition_on_previous_text=args.condition_on_previous_text,
            max_audio_seconds=args.max_audio_seconds,
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
    app = create_app(transcriber, model_name, args.max_concurrent_requests)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0

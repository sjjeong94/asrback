#!/usr/bin/env python3
"""Evaluate Whisper using an ONNX Runtime encoder and PyTorch decoder.

Run after exporting the encoder with export_to_onnx.py:

    conda run -n whisper python evaluate_onnx.py --model tiny --encoder onnx/tiny/encoder.onnx

The default dataset path matches scripts/download_librispeech_test_clean.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluate import (
    DEFAULT_DATASET,
    corpus_wer,
    get_text_normalizer,
    load_samples,
    resolve_device,
    resolve_fp16,
)

DEFAULT_ENCODER = Path("onnx/tiny/encoder.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper on LibriSpeech test-clean with an ONNX encoder."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to LibriSpeech test-clean. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=DEFAULT_ENCODER,
        help=f"Path to exported encoder.onnx. Default: {DEFAULT_ENCODER}",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Whisper model name or local checkpoint path for tokenizer and decoder weights.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for the PyTorch decoder. Default: auto",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=("auto", "cpu", "cuda", "tensorrt"),
        help="ONNX Runtime provider for the encoder. Default: auto",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code passed to Whisper. Default: en",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=("transcribe", "translate"),
        help="Whisper task. Default: transcribe",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N utterances for a quick smoke test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file for per-utterance predictions.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for Whisper decoding. Default: 5",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature. Default: 0.0",
    )
    parser.add_argument(
        "--fp16",
        choices=("auto", "true", "false"),
        default="auto",
        help="Use fp16 for PyTorch decoding. Default: auto, enabled only on CUDA.",
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


def import_onnxruntime():
    try:
        import onnxruntime as ort

        return ort
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'onnxruntime' package is not installed in this Python environment.\n"
            "Install it with:\n"
            "  python -m pip install -U onnxruntime"
        ) from exc


def import_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'torch' package is not installed in this Python environment."
        ) from exc


def select_providers(ort, provider: str, decoder_device: str) -> list[str]:
    available = set(ort.get_available_providers())
    requested = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }

    if provider == "auto":
        if decoder_device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    selected = requested[provider]
    if selected not in available:
        raise ValueError(
            f"ONNX Runtime provider is unavailable: {selected}. "
            f"Available providers: {sorted(available)}"
        )

    providers = [selected]
    if selected != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def create_onnx_audio_encoder(torch, session, output_device):
    class OnnxAudioEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.session = session
            self.input_name = session.get_inputs()[0].name
            self.output_name = session.get_outputs()[0].name

        def forward(self, mel):
            input_array = mel.detach().to("cpu", dtype=torch.float32).numpy()
            output_array = self.session.run(
                [self.output_name],
                {self.input_name: input_array},
            )[0]
            return torch.from_numpy(output_array).to(
                device=output_device,
                dtype=mel.dtype,
            )

    return OnnxAudioEncoder()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if not args.encoder.is_file():
        raise FileNotFoundError(
            f"ONNX encoder not found: {args.encoder}\n"
            "Run python export_to_onnx.py --model tiny first, or pass --encoder."
        )

    torch = import_torch()
    whisper = import_whisper()
    ort = import_onnxruntime()

    samples = load_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    device, use_cuda = resolve_device(args.device)
    fp16 = resolve_fp16(args.fp16, use_cuda)
    providers = select_providers(ort, args.provider, device)
    normalizer = get_text_normalizer()

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(samples)}")
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder}")
    print(f"Decoder device: {device}")
    print(f"Encoder providers: {providers}")
    print(f"fp16: {fp16}")
    sys.stdout.flush()

    session = ort.InferenceSession(str(args.encoder), providers=providers)
    model = whisper.load_model(args.model, device=device)
    model.encoder = create_onnx_audio_encoder(torch, session, torch.device(device))

    predictions: list[dict[str, str]] = []
    normalized_references: list[str] = []
    normalized_hypotheses: list[str] = []

    output_file = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output.open("w", encoding="utf-8")

    try:
        for index, sample in enumerate(samples, start=1):
            result = model.transcribe(
                str(sample.audio_path),
                language=args.language,
                task=args.task,
                beam_size=args.beam_size,
                temperature=args.temperature,
                fp16=fp16,
                verbose=False,
            )
            hypothesis = result.get("text", "").strip()
            normalized_reference = normalizer(sample.reference)
            normalized_hypothesis = normalizer(hypothesis)

            row = {
                "utterance_id": sample.utterance_id,
                "audio_path": str(sample.audio_path),
                "reference": sample.reference,
                "hypothesis": hypothesis,
                "normalized_reference": normalized_reference,
                "normalized_hypothesis": normalized_hypothesis,
            }
            predictions.append(row)
            normalized_references.append(normalized_reference)
            normalized_hypotheses.append(normalized_hypothesis)

            if output_file is not None:
                output_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                output_file.flush()

            if index == 1 or index % 25 == 0 or index == len(samples):
                print(f"Processed {index}/{len(samples)}")
                sys.stdout.flush()
    finally:
        if output_file is not None:
            output_file.close()

    wer, errors, words = corpus_wer(normalized_references, normalized_hypotheses)
    print(f"WER: {wer * 100:.2f}%")
    print(f"Word errors: {errors}")
    print(f"Reference words: {words}")
    if args.output is not None:
        print(f"Predictions: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

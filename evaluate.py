#!/usr/bin/env python3
"""Evaluate OpenAI Whisper models on LibriSpeech test-clean.

Run with the conda environment requested for this workspace:

    conda run -n whisper python evaluate.py --model tiny

The default dataset path matches scripts/download_librispeech_test_clean.sh.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_DATASET = Path("data/librispeech/LibriSpeech/test-clean")


@dataclass(frozen=True)
class Sample:
    utterance_id: str
    audio_path: Path
    reference: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Whisper model on LibriSpeech test-clean."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to LibriSpeech test-clean. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Whisper model name or local checkpoint path. Example: tiny, base, small, medium, large-v3",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for inference. Default: auto",
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
        help="Use fp16 inference. Default: auto, enabled only on CUDA.",
    )
    return parser.parse_args()


def load_samples(dataset_dir: Path) -> list[Sample]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Run ./scripts/download_librispeech_test_clean.sh first."
        )

    samples: list[Sample] = []
    transcript_paths = sorted(dataset_dir.glob("*/*/*.trans.txt"))
    if not transcript_paths:
        raise FileNotFoundError(f"No transcript files found under {dataset_dir}")

    for transcript_path in transcript_paths:
        with transcript_path.open("r", encoding="utf-8") as transcript_file:
            for line_number, line in enumerate(transcript_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    utterance_id, reference = line.split(" ", 1)
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed transcript line {transcript_path}:{line_number}: {line}"
                    ) from exc

                audio_path = transcript_path.parent / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Missing audio file: {audio_path}")
                samples.append(Sample(utterance_id, audio_path, reference))

    return sorted(samples, key=lambda sample: sample.utterance_id)


def get_text_normalizer() -> Callable[[str], str]:
    try:
        from whisper.normalizers import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return lambda text: normalizer(text).strip()
    except Exception:
        punctuation = str.maketrans("", "", string.punctuation)

        def fallback_normalizer(text: str) -> str:
            text = text.lower().translate(punctuation)
            return re.sub(r"\s+", " ", text).strip()

        return fallback_normalizer


def word_edit_distance(reference_words: list[str], hypothesis_words: list[str]) -> int:
    previous = list(range(len(hypothesis_words) + 1))
    for i, reference_word in enumerate(reference_words, start=1):
        current = [i]
        for j, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if reference_word == hypothesis_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def corpus_wer(
    references: Iterable[str], hypotheses: Iterable[str]
) -> tuple[float, int, int]:
    reference_words: list[str] = []
    hypothesis_words: list[str] = []

    for reference in references:
        reference_words.extend(reference.split())
    for hypothesis in hypotheses:
        hypothesis_words.extend(hypothesis.split())

    total_words = len(reference_words)
    if total_words == 0:
        raise ValueError("Cannot compute WER with zero reference words.")

    errors = word_edit_distance(reference_words, hypothesis_words)
    return errors / total_words, errors, total_words


def resolve_device(device: str) -> tuple[str, bool]:
    if device != "auto":
        return device, device == "cuda"

    try:
        import torch

        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    return ("cuda" if use_cuda else "cpu"), use_cuda


def resolve_fp16(fp16: str, use_cuda: bool) -> bool:
    if fp16 == "true":
        return True
    if fp16 == "false":
        return False
    return use_cuda


def import_whisper():
    try:
        import whisper

        return whisper
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'whisper' package is not installed in this Python environment.\n"
            "Install it in the requested conda env with:\n"
            "  conda run -n whisper python -m pip install -U openai-whisper\n"
            "Then run:\n"
            "  conda run -n whisper python evaluate.py --model tiny"
        ) from exc


def main() -> int:
    args = parse_args()
    samples = load_samples(args.dataset)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        samples = samples[: args.limit]

    device, use_cuda = resolve_device(args.device)
    fp16 = resolve_fp16(args.fp16, use_cuda)
    normalizer = get_text_normalizer()
    whisper = import_whisper()

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(samples)}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"fp16: {fp16}")
    sys.stdout.flush()

    model = whisper.load_model(args.model, device=device)
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

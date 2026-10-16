"""``asrback-eval`` command: LibriSpeech test-clean WER against a running server."""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_DATASET = Path("data/librispeech/LibriSpeech/test-clean")


@dataclass(frozen=True)
class Sample:
    utterance_id: str
    audio_path: Path
    reference: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="asrback-eval",
        description="Evaluate an OpenAI-compatible transcription server on "
        "LibriSpeech test-clean.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ASRBACK_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="Server API base URL. Default: $ASRBACK_BASE_URL or "
        "http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--model", default="whisper-1", help="Default: whisper-1")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to LibriSpeech test-clean. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code sent with each request; '' to auto-detect. Default: en",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Default: 0.0")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N utterances for a quick smoke test.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Requests in flight at once. Default: 4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file for per-utterance predictions.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.concurrency <= 0:
        parser.error("--concurrency must be greater than zero")
    return args


def load_samples(dataset_dir: Path) -> list[Sample]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Run ./scripts/download_librispeech_test_clean.sh first."
        )

    transcript_paths = sorted(dataset_dir.glob("*/*/*.trans.txt"))
    if not transcript_paths:
        raise FileNotFoundError(f"No transcript files found under {dataset_dir}")

    samples = []
    for transcript_path in transcript_paths:
        with transcript_path.open(encoding="utf-8") as transcript_file:
            for line_number, line in enumerate(transcript_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    utterance_id, reference = line.split(" ", 1)
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed transcript line {transcript_path}:{line_number}"
                    ) from exc
                audio_path = transcript_path.parent / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Missing audio file: {audio_path}")
                samples.append(Sample(utterance_id, audio_path, reference))
    return sorted(samples, key=lambda sample: sample.utterance_id)


def get_text_normalizer() -> tuple[Callable[[str], str], str]:
    """Whisper's English normalizer when openai-whisper is installed."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
    except ImportError:
        punctuation = str.maketrans("", "", string.punctuation)

        def fallback(text: str) -> str:
            text = text.lower().translate(punctuation)
            return re.sub(r"\s+", " ", text).strip()

        return fallback, "basic (install openai-whisper for Whisper's normalizer)"

    normalizer = EnglishTextNormalizer()
    return lambda text: normalizer(text).strip(), "whisper EnglishTextNormalizer"


def word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, reference_word in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import openai

    try:
        samples = load_samples(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit is not None:
        samples = samples[: args.limit]

    client = openai.OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "asrback"),
        max_retries=0,
    )
    try:
        served = [model.id for model in client.models.list()]
    except openai.APIConnectionError as exc:
        raise SystemExit(
            f"Cannot connect to {args.base_url}. Start the server first, e.g.\n"
            "  asrback --model-dir models/whisper-tiny"
        ) from exc
    normalize, normalizer_name = get_text_normalizer()

    print(f"Server: {args.base_url} (models: {', '.join(served) or 'none'})")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(samples)}")
    print(f"Normalizer: {normalizer_name}")
    sys.stdout.flush()

    options = {
        "model": args.model,
        "response_format": "verbose_json",
        "temperature": args.temperature,
    }
    if args.language:
        options["language"] = args.language

    def transcribe(sample: Sample) -> tuple[str, float]:
        with sample.audio_path.open("rb") as file:
            result = client.audio.transcriptions.create(file=file, **options)
        return result.text.strip(), float(result.duration)

    output_file = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output.open("w", encoding="utf-8")

    errors = 0
    words = 0
    audio_seconds = 0.0
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        futures = {executor.submit(transcribe, sample): sample for sample in samples}
        for done, future in enumerate(as_completed(futures), start=1):
            sample = futures[future]
            try:
                hypothesis, duration = future.result()
            except openai.APIError as exc:
                raise SystemExit(f"{sample.utterance_id}: {exc}") from exc

            reference_words = normalize(sample.reference).split()
            hypothesis_words = normalize(hypothesis).split()
            utterance_errors = word_edit_distance(reference_words, hypothesis_words)
            errors += utterance_errors
            words += len(reference_words)
            audio_seconds += duration

            if output_file is not None:
                row = {
                    "utterance_id": sample.utterance_id,
                    "audio_path": str(sample.audio_path),
                    "reference": sample.reference,
                    "hypothesis": hypothesis,
                    "normalized_reference": " ".join(reference_words),
                    "normalized_hypothesis": " ".join(hypothesis_words),
                    "errors": utterance_errors,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()

            if done == 1 or done % 25 == 0 or done == len(samples):
                wer = errors / words * 100 if words else 0.0
                print(f"Processed {done}/{len(samples)}  WER so far: {wer:.2f}%")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        # Drop queued requests instead of waiting for them on error or Ctrl-C.
        executor.shutdown(wait=False, cancel_futures=True)
        if output_file is not None:
            output_file.close()

    elapsed = time.perf_counter() - started
    if words == 0:
        raise SystemExit("Cannot compute WER with zero reference words.")
    print(f"WER: {errors / words * 100:.2f}%")
    print(f"Word errors: {errors}")
    print(f"Reference words: {words}")
    print(
        f"Audio: {audio_seconds:.1f}s  Elapsed: {elapsed:.1f}s  "
        f"RTFx: {audio_seconds / elapsed:.1f}"
    )
    if args.output is not None:
        print(f"Predictions: {args.output}")
    return 0

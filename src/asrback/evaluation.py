"""``asrback-eval`` command: WER/CER on LibriSpeech or FLEURS against a running server."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import re
import signal
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_DATASET = Path("data/librispeech/LibriSpeech/test-clean")
# Scored by character error rate by default: no spaces between words (ja, zh, th,
# lo, my, yue) or words too coarse for a meaningful WER (ko).
CER_LANGUAGES = {"ja", "zh", "ko", "th", "lo", "my", "yue"}


@dataclass(frozen=True)
class Sample:
    utterance_id: str
    audio_path: Path
    reference: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="asrback-eval",
        description="Evaluate an OpenAI-compatible transcription server on "
        "LibriSpeech test-clean or a FLEURS test set.",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        default=None,
        help="Server API base URL; repeat to spread requests over several servers. "
        "Default: $ASRBACK_BASE_URL or http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--model", default="whisper-1", help="Default: whisper-1")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="LibriSpeech test-clean directory, or a FLEURS language directory "
        f"such as data/fleurs/ko_kr. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code sent with each request; '' to auto-detect. "
        "Default: from the dataset (en for LibriSpeech, ko for fleurs/ko_kr)",
    )
    parser.add_argument(
        "--metric",
        choices=("wer", "cer"),
        default=None,
        help="Word or character error rate. Default: cer for "
        f"{', '.join(sorted(CER_LANGUAGES))}, otherwise wer",
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
        default=128,
        help="Requests in flight at once; keep it well above the server's batch "
        "size. Default: 128",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Client processes sending the requests. The OpenAI client costs enough "
        "CPU per request that one process can be slower than the server. Default: 4",
    )
    parser.add_argument(
        "--sort-by-length",
        action="store_true",
        help="Send the longest files first so concurrent requests batch well.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file for per-utterance predictions.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip utterances already in --output and include them in the score.",
    )
    args = parser.parse_args(argv)
    if args.base_url is None:
        args.base_url = [os.environ.get("ASRBACK_BASE_URL", "http://127.0.0.1:8000/v1")]
    if args.resume and args.output is None:
        parser.error("--resume requires --output")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.concurrency <= 0:
        parser.error("--concurrency must be greater than zero")
    if args.processes <= 0:
        parser.error("--processes must be greater than zero")
    return args


def is_fleurs(dataset_dir: Path) -> bool:
    return (dataset_dir / "test.tsv").is_file()


def dataset_language(dataset_dir: Path) -> str:
    """Language of a dataset: FLEURS directories are named like ``ko_kr``."""
    if is_fleurs(dataset_dir):
        return dataset_dir.name.split("_")[0]
    return "en"


def load_samples(dataset_dir: Path) -> list[Sample]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Run ./scripts/download_librispeech_test_clean.sh or "
            "./scripts/download_fleurs_test.sh first."
        )
    if is_fleurs(dataset_dir):
        return load_fleurs(dataset_dir)
    return load_librispeech(dataset_dir)


def load_fleurs(dataset_dir: Path) -> list[Sample]:
    """``test.tsv`` rows: id, file name, raw transcription, normalized
    transcription, characters, sample count, gender. Audio is in ``test/``."""
    transcript_path = dataset_dir / "test.tsv"
    samples = []
    with transcript_path.open(encoding="utf-8") as transcript_file:
        for line_number, line in enumerate(transcript_file, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 7:
                raise ValueError(
                    f"Malformed transcript line {transcript_path}:{line_number}"
                )
            file_name, reference = fields[1], fields[2]
            audio_path = dataset_dir / "test" / file_name
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio file: {audio_path}")
            # FLEURS ids repeat across speakers; the file name is unique.
            samples.append(Sample(Path(file_name).stem, audio_path, reference))
    if not samples:
        raise FileNotFoundError(f"No transcripts found in {transcript_path}")
    return sorted(samples, key=lambda sample: sample.utterance_id)


def load_librispeech(dataset_dir: Path) -> list[Sample]:
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


def get_text_normalizer(language: str) -> tuple[Callable[[str], str], str]:
    """Whisper's English or basic normalizer when openai-whisper is installed."""
    try:
        from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
    except ImportError:

        def fallback(text: str) -> str:
            # Lowercase and turn punctuation and symbols into spaces.
            text = unicodedata.normalize("NFKC", text).lower()
            text = "".join(
                " " if unicodedata.category(char)[0] in "PS" else char for char in text
            )
            return re.sub(r"\s+", " ", text).strip()

        return fallback, "basic (install openai-whisper for Whisper's normalizer)"

    if language == "en":
        normalizer = EnglishTextNormalizer()
        name = "whisper EnglishTextNormalizer"
    else:
        normalizer = BasicTextNormalizer()
        name = "whisper BasicTextNormalizer"
    return lambda text: re.sub(r"\s+", " ", normalizer(text)).strip(), name


def score_units(text: str, metric: str) -> list[str]:
    """Words for WER, or characters without spaces for CER."""
    if metric == "cer":
        return list(text.replace(" ", ""))
    return text.split()


def load_previous(output: Path) -> dict[str, dict]:
    """Rows of an earlier ``--output`` file by utterance id (empty if missing)."""
    if not output.is_file():
        return {}
    rows = {}
    with output.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                rows[row["utterance_id"]] = row
    return rows


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
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


def send_requests(
    base_urls: list[str],
    api_key: str,
    options: dict,
    first_thread: int,
    threads: int,
    jobs: multiprocessing.Queue,
    results: multiprocessing.Queue,
) -> None:
    """Client process: transcribe ``(index, path)`` items from ``jobs`` on
    ``threads`` threads until each reads a ``None``, and put
    ``(index, text, duration, error)`` on ``results``, after a ``None`` once
    the clients are ready."""
    # Ctrl-C reaches the whole process group; the main process stops us.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import openai

    clients = [
        openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        for base_url in base_urls
    ]
    results.put(None)

    def work(thread: int) -> None:
        # Threads alternate between servers, counted across all processes.
        client = clients[thread % len(clients)]
        while (item := jobs.get()) is not None:
            index, path = item
            try:
                with open(path, "rb") as file:
                    result = client.audio.transcriptions.create(file=file, **options)
                results.put((index, result.text.strip(), float(result.duration), None))
            except Exception as exc:
                results.put((index, "", 0.0, str(exc)))

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for thread in range(first_thread, first_thread + threads):
            executor.submit(work, thread)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import openai

    try:
        samples = load_samples(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    data_language = dataset_language(args.dataset)
    language = data_language if args.language is None else args.language
    metric = args.metric or ("cer" if data_language in CER_LANGUAGES else "wer")
    metric_name = metric.upper()
    unit_name = "characters" if metric == "cer" else "words"
    if args.limit is not None:
        samples = samples[: args.limit]
    total = len(samples)

    previous = load_previous(args.output) if args.resume else {}
    previous = {
        sample.utterance_id: previous[sample.utterance_id]
        for sample in samples
        if sample.utterance_id in previous
    }
    if any(row.get("metric", "wer") != metric for row in previous.values()):
        raise SystemExit(f"{args.output} was scored with a different metric.")
    samples = [sample for sample in samples if sample.utterance_id not in previous]
    if args.sort_by_length:
        samples.sort(key=lambda sample: sample.audio_path.stat().st_size, reverse=True)

    api_key = os.environ.get("OPENAI_API_KEY", "asrback")
    clients = [
        openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        for base_url in args.base_url
    ]
    served_by_server = []
    for base_url, client in zip(args.base_url, clients):
        try:
            served = [model.id for model in client.models.list()]
        except openai.APIConnectionError as exc:
            raise SystemExit(
                f"Cannot connect to {base_url}. Start the server first, e.g.\n"
                "  asrback --model-dir models/whisper-tiny"
            ) from exc
        served_by_server.append(served)
        print(f"Server: {base_url} (models: {', '.join(served) or 'none'})")
    if any(served != served_by_server[0] for served in served_by_server):
        print("Warning: servers report different models.")
    normalize, normalizer_name = get_text_normalizer(data_language)

    print(f"Dataset: {args.dataset}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Samples: {total}")
    if previous:
        print(f"Resumed: {len(previous)} from {args.output}")
    print(f"Normalizer: {normalizer_name}")
    print(f"Metric: {metric_name}")
    sys.stdout.flush()

    options = {
        "model": args.model,
        "response_format": "verbose_json",
        "temperature": args.temperature,
    }
    if language:
        options["language"] = language

    output_file = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite resumed rows first so a stale or partial line cannot survive.
        output_file = args.output.open("w", encoding="utf-8")
        for row in previous.values():
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_file.flush()

    errors = sum(row["errors"] for row in previous.values())
    units = sum(
        len(score_units(row["normalized_reference"], metric))
        for row in previous.values()
    )
    audio_seconds = 0.0
    # Requests go out from several processes; this one only scores the results.
    context = multiprocessing.get_context("spawn")
    jobs, results = context.Queue(), context.Queue()
    concurrency = min(args.concurrency, max(len(samples), 1))
    process_count = min(args.processes, concurrency)
    processes = []
    for rank in range(process_count):
        first = rank * concurrency // process_count
        threads = (rank + 1) * concurrency // process_count - first
        processes.append(
            context.Process(
                target=send_requests,
                args=(args.base_url, api_key, options, first, threads, jobs, results),
                daemon=True,
            )
        )
    for process in processes:
        process.start()

    def next_result() -> tuple[int, str, float, str | None] | None:
        while True:
            try:
                return results.get(timeout=1)
            except queue.Empty:
                # Processes exit once their threads run out of jobs; a failure
                # shows as an exit code, or as results missing after all exited.
                if any(process.exitcode for process in processes) or not any(
                    process.is_alive() for process in processes
                ):
                    raise SystemExit("A client process exited unexpectedly.") from None

    try:
        # Each process reports None once its clients are ready; time from there.
        for _ in processes:
            next_result()
        started = time.perf_counter()
        for index, sample in enumerate(samples):
            jobs.put((index, str(sample.audio_path)))
        for _ in range(concurrency):
            jobs.put(None)

        for done in range(len(previous) + 1, total + 1):
            index, hypothesis, duration, error = next_result()
            sample = samples[index]
            if error is not None:
                raise SystemExit(f"{sample.utterance_id}: {error}")

            normalized_reference = normalize(sample.reference)
            normalized_hypothesis = normalize(hypothesis)
            reference_units = score_units(normalized_reference, metric)
            utterance_errors = edit_distance(
                reference_units, score_units(normalized_hypothesis, metric)
            )
            errors += utterance_errors
            units += len(reference_units)
            audio_seconds += duration

            if output_file is not None:
                row = {
                    "utterance_id": sample.utterance_id,
                    "audio_path": str(sample.audio_path),
                    "reference": sample.reference,
                    "hypothesis": hypothesis,
                    "normalized_reference": normalized_reference,
                    "normalized_hypothesis": normalized_hypothesis,
                    "metric": metric,
                    "errors": utterance_errors,
                    "duration": duration,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()

            if done == len(previous) + 1 or done % 25 == 0 or done == total:
                rate = errors / units * 100 if units else 0.0
                print(f"Processed {done}/{total}  {metric_name} so far: {rate:.2f}%")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        # Stop requests still in flight on error or Ctrl-C; done otherwise.
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        jobs.cancel_join_thread()
        if output_file is not None:
            output_file.close()

    elapsed = time.perf_counter() - started
    if units == 0:
        raise SystemExit(
            f"Cannot compute {metric_name} with zero reference {unit_name}."
        )
    print(f"{metric_name}: {errors / units * 100:.2f}%")
    print(f"Errors: {errors}")
    print(f"Reference {unit_name}: {units}")
    if samples:
        print(
            f"Audio: {audio_seconds:.1f}s  Elapsed: {elapsed:.1f}s  "
            f"RTFx: {audio_seconds / elapsed:.1f}"
        )
    if args.output is not None:
        print(f"Predictions: {args.output}")
    return 0

"""Greedy / sampled Whisper decoding over 30-second audio windows.

Concurrent requests are decoded together: callers queue one window at a time and a
single worker thread runs them in batches.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
import zlib
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import asrfront
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import Tensor

from .model import DecoderCache, WhisperForConditionalGeneration

AudioSource = Union[str, os.PathLike, bytes, bytearray, memoryview, np.ndarray]

SAMPLE_RATE = asrfront.whisper.SAMPLE_RATE
WINDOW_SAMPLES = asrfront.whisper.N_SAMPLES
# Trailing windows shorter than this are mostly padding and invite hallucinations.
MIN_TAIL_SAMPLES = SAMPLE_RATE // 10

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


@dataclass
class Transcription:
    task: str
    language: str
    duration: float
    text: str
    segments: list[Segment] = field(default_factory=list)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token!r}")
    return token_id


def language_token_ids(tokenizer_config: dict) -> dict[str, int]:
    languages = {}
    for raw_id, entry in tokenizer_config.get("added_tokens_decoder", {}).items():
        content = entry.get("content", "")
        match = re.fullmatch(r"<\|([a-z]{2,3})\|>", content)
        if match:
            languages[match.group(1)] = int(raw_id)
    if not languages:
        raise ValueError("tokenizer_config.json contains no Whisper language tokens")
    return languages


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


@dataclass
class _Job:
    """One 30-second window waiting to be decoded."""

    log_mel: np.ndarray
    previous: list[int]
    language: str | None
    task: str
    temperature: float
    max_new_tokens: int | None
    future: Future = field(default_factory=Future)


@dataclass
class _Result:
    language: str
    tokens: list[int]
    avg_logprob: float
    no_speech_prob: float


class _GraphStep:
    """One decoder step for a fixed batch size, captured as a CUDA graph.

    Inputs are read from ``tokens`` and ``position`` and the step writes into
    ``cache`` in place; :meth:`replay` returns float32 logits for the new token.
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        cache: DecoderCache,
        tokens: Tensor,
        position: Tensor,
    ) -> None:
        def step() -> Tensor:
            hidden = model.decode(tokens, position, cache)
            return model.logits(hidden[:, -1]).float()

        # Warm up on a side stream so lazy initialization stays out of the graph.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.logits = step()

    def replay(self) -> Tensor:
        self.graph.replay()
        return self.logits


class Transcriber:
    """Loads a Whisper checkpoint once and transcribes or translates audio.

    ``transcribe`` is thread-safe: windows from concurrent calls are decoded in
    batches of up to ``max_batch_size``, waiting at most ``batch_wait_ms`` for a
    batch to fill. ``cuda_graphs`` (default: on for CUDA) replays each decoder step
    from a captured graph instead of launching its kernels one by one.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        max_batch_size: int = 16,
        batch_wait_ms: float = 5.0,
        cuda_graphs: bool | None = None,
    ):
        model_dir = Path(model_dir)
        for filename in REQUIRED_FILES:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing model file: {model_dir / filename}")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")

        self.model_dir = model_dir
        self.config = load_json(model_dir / "config.json")
        tokenizer_config = load_json(model_dir / "tokenizer_config.json")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        elif isinstance(dtype, str):
            dtype = DTYPES[dtype]
        self.dtype = dtype
        if cuda_graphs is None:
            cuda_graphs = self.device.type == "cuda"
        if cuda_graphs and self.device.type != "cuda":
            raise ValueError("cuda_graphs requires a CUDA device")
        self.cuda_graphs = cuda_graphs
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000

        model = WhisperForConditionalGeneration(self.config)
        state_dict = load_file(str(model_dir / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.model = model.eval().to(self.device, self.dtype)

        # English-only (*.en) checkpoints have a smaller vocabulary, start decoding
        # without language/task tokens and cannot translate.
        self.multilingual = self.config["vocab_size"] >= 51865
        tasks = ("transcribe", "translate") if self.multilingual else ("transcribe",)
        if self.multilingual:
            self.languages = language_token_ids(tokenizer_config)
        else:
            self.languages = {"en": required_token_id(self.tokenizer, "<|en|>")}
        self.start_id = required_token_id(self.tokenizer, "<|startoftranscript|>")
        self.prev_id = required_token_id(self.tokenizer, "<|startofprev|>")
        self.no_timestamps_id = required_token_id(self.tokenizer, "<|notimestamps|>")
        self.task_ids = {
            task: required_token_id(self.tokenizer, f"<|{task}|>") for task in tasks
        }
        self.no_speech_id = self.tokenizer.token_to_id(
            "<|nospeech|>"
        ) or self.tokenizer.token_to_id("<|nocaptions|>")

        # Tokens never sampled: configured suppressions and every timestamp token.
        vocab_size = self.config["vocab_size"]
        suppress = torch.zeros(vocab_size, dtype=torch.bool)
        suppress[self.config.get("suppress_tokens", [])] = True
        suppress[self.no_timestamps_id + 1 :] = True
        first_suppress = suppress.clone()
        first_suppress[self.config.get("begin_suppress_tokens", [])] = True
        self._suppress = suppress.to(self.device)
        self._first_suppress = first_suppress.to(self.device)
        self._language_ids = torch.tensor(
            list(self.languages.values()), device=self.device
        )
        self._positions = torch.arange(
            self.config["max_target_positions"], device=self.device
        )

        # Batch sizes with their own CUDA graph; a batch is padded up to one.
        self._buckets = sorted(
            {
                min(2**power, max_batch_size)
                for power in range(max_batch_size.bit_length())
            }
            | {max_batch_size}
        )
        self._cache: DecoderCache | None = None
        self._graphs: dict[int, _GraphStep] = {}
        if self.cuda_graphs:
            self._capture_graphs()

        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        threading.Thread(
            target=self._serve, name="asrback-decoder", daemon=True
        ).start()

    def transcribe(
        self,
        audio: AudioSource,
        *,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> Transcription:
        """Decode ``audio`` (a path, encoded file bytes, or a 16 kHz waveform)."""
        if task not in self.task_ids:
            choices = " or ".join(repr(name) for name in self.task_ids)
            raise ValueError(f"task must be {choices} for this model, got {task!r}")
        if language is not None and language not in self.languages:
            choices = ", ".join(sorted(self.languages))
            raise ValueError(
                f"Unsupported language {language!r}. Available codes: {choices}"
            )
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")

        if isinstance(audio, np.ndarray) and audio.dtype != np.uint8:
            waveform = np.asarray(audio, dtype=np.float32)
        else:
            waveform = asrfront.load(audio)
        duration = len(waveform) / SAMPLE_RATE

        history = []
        if prompt and prompt.strip():
            history = self.tokenizer.encode(
                " " + prompt.strip(), add_special_tokens=False
            ).ids
        max_target = self.config["max_target_positions"]

        segments = []
        # Windows are decoded in order: each one is conditioned on earlier text.
        for seek, length, log_mel in self._windows(waveform):
            # Condition on earlier text like openai-whisper: at most half the context.
            previous = history[-(max_target // 2 - 1) :]
            job = _Job(log_mel, previous, language, task, temperature, max_new_tokens)
            self._queue.put(job)
            result: _Result = job.future.result()
            language = result.language

            text = self.tokenizer.decode(result.tokens, skip_special_tokens=True)
            history.extend(result.tokens)
            segments.append(
                Segment(
                    id=len(segments),
                    seek=seek // asrfront.whisper.HOP_LENGTH,
                    start=round(seek / SAMPLE_RATE, 3),
                    end=round((seek + length) / SAMPLE_RATE, 3),
                    text=text,
                    tokens=result.tokens,
                    temperature=temperature,
                    avg_logprob=result.avg_logprob,
                    compression_ratio=compression_ratio(text),
                    no_speech_prob=result.no_speech_prob,
                )
            )

        return Transcription(
            task=task,
            language=language,
            duration=round(duration, 3),
            text="".join(segment.text for segment in segments).strip(),
            segments=segments,
        )

    def _windows(self, waveform: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """Split into 30-second windows: ``(seek, length, log_mel)`` each."""
        windows = []
        for seek in range(0, max(len(waveform), 1), WINDOW_SAMPLES):
            window = waveform[seek : seek + WINDOW_SAMPLES]
            if windows and len(window) < MIN_TAIL_SAMPLES:
                break
            log_mel = asrfront.whisper.log_mel(
                window, n_mels=self.config["num_mel_bins"]
            )
            windows.append((seek, len(window), log_mel))
        return windows

    @torch.inference_mode()
    def _capture_graphs(self) -> None:
        if self.device.index is not None:
            torch.cuda.set_device(self.device)
        self._cache = self.model.allocate_cache(self.max_batch_size)
        self._step_tokens = torch.zeros(
            (self.max_batch_size, 1), dtype=torch.long, device=self.device
        )
        self._step_position = torch.zeros(1, dtype=torch.long, device=self.device)
        for batch_size in self._buckets:
            self._graphs[batch_size] = _GraphStep(
                self.model,
                self._cache.narrow(batch_size),
                self._step_tokens[:batch_size],
                self._step_position,
            )

    def _serve(self) -> None:
        """Worker loop: gather queued windows into batches and decode them."""
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        while True:
            jobs = [self._queue.get()]
            deadline = time.monotonic() + self.batch_wait
            while len(jobs) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                try:
                    if remaining > 0:
                        jobs.append(self._queue.get(timeout=remaining))
                    else:
                        jobs.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            # Rows of a batch must share the prefix length and token budget.
            groups: dict[tuple[int, int], list[_Job]] = {}
            for job in jobs:
                groups.setdefault(self._layout(job), []).append(job)
            for (prefix_length, limit), group in groups.items():
                try:
                    results = self._decode_batch(group, prefix_length, limit)
                except BaseException as exc:
                    for job in group:
                        job.future.set_exception(exc)
                    continue
                for job, result in zip(group, results):
                    job.future.set_result(result)

    def _sot(self, language: str, task: str) -> list[int]:
        sot = [self.start_id]
        if self.multilingual:
            sot += [self.languages[language], self.task_ids[task]]
        return sot + [self.no_timestamps_id]

    def _layout(self, job: _Job) -> tuple[int, int]:
        """``(prefix_length, max_new_tokens)`` for a job."""
        previous = len(job.previous) + 1 if job.previous else 0
        prefix_length = previous + len(self._sot("en", job.task))
        available = self.config["max_target_positions"] - prefix_length
        return prefix_length, min(job.max_new_tokens or available, available)

    @torch.inference_mode()
    def _decode_batch(
        self, jobs: list[_Job], prefix_length: int, limit: int
    ) -> list[_Result]:
        batch_size = len(jobs)
        mel = torch.from_numpy(np.stack([job.log_mel for job in jobs]))
        encoder_output = self.model.model.encoder(mel.to(self.device, self.dtype))

        graph = None
        if self.cuda_graphs:
            bucket = next(size for size in self._buckets if size >= batch_size)
            graph = self._graphs[bucket]
        elif self._cache is None or self._cache.batch_size < batch_size:
            self._cache = self.model.allocate_cache(batch_size)
        cache = self._cache.narrow(batch_size)
        self.model.set_encoder_output(cache, encoder_output)

        languages = [job.language for job in jobs]
        if None in languages:
            detected = self._detect_languages(cache, batch_size)
            languages = [given or found for given, found in zip(languages, detected)]

        prefixes = []
        for job, language in zip(jobs, languages):
            previous = [self.prev_id] + job.previous if job.previous else []
            prefixes.append(previous + self._sot(language, job.task))
        tokens = torch.tensor(prefixes, device=self.device)
        sot_index = prefix_length - len(self._sot("en", jobs[0].task))

        hidden = self.model.decode(
            tokens, self._positions[:prefix_length], cache, length=prefix_length
        )
        logits = self.model.logits(hidden[:, [sot_index, prefix_length - 1]]).float()
        if self.no_speech_id is not None:
            no_speech_probs = logits[:, 0].softmax(dim=-1)[:, self.no_speech_id]
        else:
            no_speech_probs = torch.zeros(batch_size, device=self.device)
        next_logits = logits[:, 1]

        temperatures = torch.tensor(
            [job.temperature for job in jobs], device=self.device
        )
        sampling = any(job.temperature > 0 for job in jobs)
        eos = self.config["eos_token_id"]
        generated = torch.empty(
            (batch_size, limit), dtype=torch.long, device=self.device
        )
        sum_logprobs = torch.zeros(batch_size, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        steps = 0
        for step in range(limit):
            suppress = self._first_suppress if step == 0 else self._suppress
            next_logits.masked_fill_(suppress, -math.inf)
            token = next_logits.argmax(dim=-1)
            if sampling:
                scaled = next_logits / temperatures.clamp(min=1e-6)[:, None]
                sampled = torch.multinomial(F.softmax(scaled, dim=-1), 1).squeeze(1)
                token = torch.where(temperatures > 0, sampled, token)
            logprob = F.log_softmax(next_logits, dim=-1).gather(1, token[:, None])
            # Rows that already ended keep decoding padding; ignore their scores.
            sum_logprobs += logprob.squeeze(1).masked_fill(finished, 0.0)
            generated[:, step] = token
            finished |= token == eos
            steps = step + 1
            # The only host synchronization per step.
            if steps == limit or bool(finished.all()):
                break

            position = prefix_length + step
            if graph is not None:
                self._step_tokens[:batch_size, 0].copy_(token)
                self._step_position.fill_(position)
                next_logits = graph.replay()[:batch_size]
            else:
                hidden = self.model.decode(
                    token[:, None],
                    self._positions[position : position + 1],
                    cache,
                    length=position + 1,
                )
                next_logits = self.model.logits(hidden[:, -1]).float()

        results = []
        rows = generated[:, :steps].tolist()
        for row, language, sum_logprob, no_speech_prob in zip(
            rows, languages, sum_logprobs.tolist(), no_speech_probs.tolist()
        ):
            if eos in row:
                row = row[: row.index(eos)]
            avg_logprob = sum_logprob / (len(row) + 1)
            results.append(_Result(language, row, avg_logprob, no_speech_prob))
        return results

    def _detect_languages(self, cache: DecoderCache, batch_size: int) -> list[str]:
        if not self.multilingual:
            return ["en"] * batch_size
        tokens = torch.full((batch_size, 1), self.start_id, device=self.device)
        # Writes cache position 0, which the prefix pass then overwrites.
        hidden = self.model.decode(tokens, self._positions[:1], cache, length=1)
        logits = self.model.logits(hidden[:, -1])[:, self._language_ids]
        names = list(self.languages)
        return [names[index] for index in logits.argmax(dim=-1).tolist()]

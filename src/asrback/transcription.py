"""Types and helpers shared by the transcribers."""

from __future__ import annotations

import concurrent.futures
import os
import queue
import threading
import time
import zlib
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, Union

import asrfront
import numpy as np
import torch

AudioSource = Union[str, os.PathLike, bytes, bytearray, memoryview, np.ndarray]

SAMPLE_RATE = 16000
# How often a request waiting on its result checks that the worker still runs.
WAIT_POLL_SECONDS = 1.0

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


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


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def load_audio(audio: AudioSource, max_seconds: float | None = None) -> np.ndarray:
    """``audio`` (a path, encoded file bytes, or a 16 kHz waveform) as a float32
    waveform. Audio longer than ``max_seconds`` raises ``ValueError``; encoded
    audio does so before most of it is decoded."""
    if isinstance(audio, np.ndarray) and audio.dtype != np.uint8:
        waveform = np.asarray(audio, dtype=np.float32)
        if max_seconds is not None and len(waveform) > max_seconds * SAMPLE_RATE:
            raise ValueError(
                f"audio is {len(waveform) / SAMPLE_RATE:.1f} s long; "
                f"the limit is {max_seconds:g} s"
            )
        return waveform
    return asrfront.load(audio, max_duration=max_seconds)


class ModelUnavailableError(RuntimeError):
    """The decoding worker or the model process has stopped: this and every later
    request fail until the server restarts."""


class Worker:
    """A transcriber's decoding thread.

    The thread only stops on an error it cannot recover from (see
    :func:`raise_if_device_failed`). :meth:`wait` then fails the requests waiting
    on it and later ones, instead of leaving them waiting forever.
    """

    def __init__(self, target: Callable[[], None], name: str) -> None:
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(target,), name=name, daemon=True
        )
        self._thread.start()

    def _run(self, target: Callable[[], None]) -> None:
        try:
            target()
        except BaseException as exc:
            self.error = exc
            raise  # threading.excepthook prints the traceback

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def wait(self, future: Future):
        """The result of ``future``, which the worker sets."""
        while True:
            try:
                return future.result(timeout=WAIT_POLL_SECONDS)
            except concurrent.futures.TimeoutError:
                if not self.alive and not future.done():
                    raise ModelUnavailableError(
                        f"the decoding worker has stopped: {self.error!r}"
                    ) from self.error


def raise_if_device_failed(device: torch.device) -> None:
    """Called by a worker after a failed batch. An error such as an out-of-memory
    leaves the device usable, but one such as an illegal memory access poisons
    the CUDA context: every later call fails, so this raises and stops the worker
    rather than failing every request that follows."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def split_on_out_of_memory(exc: BaseException, jobs: list) -> bool:
    """Whether a batch of ``jobs`` that failed with ``exc`` should be retried in
    halves: an out-of-memory error leaves the device usable, and smaller batches
    may fit, so one large batch does not fail every request in it."""
    return isinstance(exc, torch.OutOfMemoryError) and len(jobs) > 1


def to_device(data, device: torch.device) -> torch.Tensor:
    """``data`` (an array or a list) copied to ``device`` without waiting for the
    work queued there, as a copy from pageable memory would."""
    tensor = torch.as_tensor(data)
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=True)


def gather_batch(jobs: queue.SimpleQueue, max_size: int, wait: float) -> list:
    """Block for one job, then take more until ``max_size`` or ``wait`` seconds."""
    batch = [jobs.get()]
    deadline = time.monotonic() + wait
    while len(batch) < max_size:
        remaining = deadline - time.monotonic()
        try:
            if remaining > 0:
                batch.append(jobs.get(timeout=remaining))
            else:
                batch.append(jobs.get_nowait())
        except queue.Empty:
            break
    return batch


class CapturedGraph:
    """A function of fixed-address tensors captured once as a CUDA graph.

    Graphs that never replay concurrently can share their memory ``pool``
    (``torch.cuda.graph_pool_handle()``).
    """

    def __init__(self, function: Callable[[], None], pool=None) -> None:
        # Warm up on a side stream so lazy initialization stays out of the graph.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                function()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, capture_error_mode="thread_local"):
            function()

    def replay(self) -> None:
        self.graph.replay()

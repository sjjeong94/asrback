"""Types and helpers shared by the transcribers."""

from __future__ import annotations

import os
import queue
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import torch

AudioSource = Union[str, os.PathLike, bytes, bytearray, memoryview, np.ndarray]

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

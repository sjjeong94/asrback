"""Types and helpers shared by the Whisper and SenseVoice transcribers."""

from __future__ import annotations

import os
import queue
import time
import zlib
from dataclasses import dataclass, field
from typing import Union

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

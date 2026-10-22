"""Shared helpers: the worker thread, audio loading and batching."""

from __future__ import annotations

import io
import queue
import threading
import wave
from concurrent.futures import Future

import numpy as np
import pytest
import torch

from asrback.models.qwen3_asr import fix_repetitions
from asrback.models.sensevoice import _Job, take_similar
from asrback.transcription import (
    SAMPLE_RATE,
    Worker,
    compression_ratio,
    gather_batch,
    load_audio,
    raise_if_device_failed,
)


def wav_bytes(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(bytes(2 * int(seconds * SAMPLE_RATE)))
    return buffer.getvalue()


def test_worker_wait_returns_the_result():
    jobs: queue.SimpleQueue[Future] = queue.SimpleQueue()

    def serve():
        while True:
            jobs.get().set_result("done")

    worker = Worker(serve, "test-worker")
    future: Future = Future()
    jobs.put(future)
    assert worker.wait(future) == "done"
    assert worker.alive


def test_worker_wait_fails_once_the_worker_stops(monkeypatch):
    monkeypatch.setattr("asrback.transcription.WAIT_POLL_SECONDS", 0.01)
    # Keep the expected traceback out of the test output.
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    stop = threading.Event()

    def serve():
        stop.wait()
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    worker = Worker(serve, "test-worker")
    stop.set()
    with pytest.raises(RuntimeError, match="the decoding worker has stopped"):
        worker.wait(Future())
    assert not worker.alive
    assert "illegal memory access" in str(worker.error)


def test_worker_wait_returns_a_result_set_before_the_worker_stopped(monkeypatch):
    monkeypatch.setattr("asrback.transcription.WAIT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    future: Future = Future()

    def serve():
        future.set_result("done")
        raise RuntimeError("stopped")

    assert Worker(serve, "test-worker").wait(future) == "done"


def test_raise_if_device_failed_on_cpu():
    raise_if_device_failed(torch.device("cpu"))


def test_load_audio_waveform():
    waveform = np.zeros(2 * SAMPLE_RATE, dtype=np.float64)
    loaded = load_audio(waveform, max_seconds=2)
    assert loaded.dtype == np.float32 and len(loaded) == 2 * SAMPLE_RATE
    with pytest.raises(ValueError, match="limit is 1 s"):
        load_audio(waveform, max_seconds=1)


def test_load_audio_encoded():
    data = wav_bytes(2.0)
    assert len(load_audio(data)) == 2 * SAMPLE_RATE
    assert len(load_audio(data, max_seconds=2)) == 2 * SAMPLE_RATE
    with pytest.raises(ValueError):
        load_audio(data, max_seconds=1)


def test_load_audio_rejects_unsupported_formats():
    with pytest.raises(ValueError, match="unsupported audio format"):
        load_audio(b"OggS" + bytes(2000))


def test_gather_batch():
    jobs: queue.SimpleQueue[int] = queue.SimpleQueue()
    for job in range(5):
        jobs.put(job)
    assert gather_batch(jobs, 3, wait=0) == [0, 1, 2]
    assert gather_batch(jobs, 3, wait=0) == [3, 4]


def test_compression_ratio():
    assert compression_ratio("") == 0.0
    assert compression_ratio("ha" * 200) > 10


def test_take_similar():
    pending = [_Job(np.zeros((length, 1)), None) for length in (50, 90, 40, 60, 10)]
    jobs = take_similar(pending, 3)
    # The oldest (50) plus the longest no longer than it (40, 10).
    assert [len(job.features) for job in jobs] == [50, 40, 10]
    assert [len(job.features) for job in pending] == [90, 60]
    # Then the shortest longer ones.
    pending = [_Job(np.zeros((length, 1)), None) for length in (50, 90, 60)]
    assert [len(job.features) for job in take_similar(pending, 2)] == [50, 60]


def test_fix_repetitions():
    assert fix_repetitions("a" * 30 + "b") == "ab"
    assert fix_repetitions("hello " + "la " * 25 + "end") == "hello la end"
    assert fix_repetitions("an ordinary sentence") == "an ordinary sentence"

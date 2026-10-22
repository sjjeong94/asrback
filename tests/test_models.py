"""Real checkpoints, when they have been downloaded to ``models/`` (see
``scripts/``) along with LibriSpeech test-clean; otherwise these tests are skipped."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from asrback.models import qwen3_asr, sensevoice
from asrback.models.qwen3_asr import Qwen3ASRTranscriber
from asrback.models.sensevoice import SenseVoiceTranscriber

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "data/librispeech/LibriSpeech/test-clean/4446/2273/4446-2273-0001.flac"
TRANSCRIPT = "they asked him to come to see them in chelsea"
SENSEVOICE = ROOT / "models/sensevoice-small"
QWEN3_ASR = ROOT / "models/qwen3-asr-0.6b"


def require(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            pytest.skip(f"{path.relative_to(ROOT)} has not been downloaded")


def normalized(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum() or c == " ")


def stop_worker(monkeypatch, module, transcriber, method: str) -> None:
    """Make the next batch fail as if the CUDA context were poisoned."""
    monkeypatch.setattr("asrback.transcription.WAIT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(transcriber, method, fail)
    monkeypatch.setattr(module, "raise_if_device_failed", fail)


@pytest.fixture(scope="module")
def sensevoice_model():
    require(SENSEVOICE, AUDIO)
    return SenseVoiceTranscriber(SENSEVOICE, max_audio_seconds=60)


def test_sensevoice(sensevoice_model):
    result = sensevoice_model.transcribe(AUDIO.read_bytes(), language="en")
    assert normalized(result.text).startswith(TRANSCRIPT)
    assert sensevoice_model.alive


def test_sensevoice_max_audio_seconds(sensevoice_model):
    with pytest.raises(ValueError):
        sensevoice_model.transcribe(np.zeros(61 * 16000, dtype=np.float32))


def test_sensevoice_worker_stops(monkeypatch):
    require(SENSEVOICE, AUDIO)
    transcriber = SenseVoiceTranscriber(SENSEVOICE)
    stop_worker(monkeypatch, sensevoice, transcriber, "_launch")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        transcriber.transcribe(AUDIO.read_bytes())
    # Later requests fail rather than wait forever.
    with pytest.raises(RuntimeError, match="the decoding worker has stopped"):
        transcriber.transcribe(AUDIO.read_bytes())
    assert not transcriber.alive


def test_sensevoice_survives_a_failed_batch(monkeypatch, sensevoice_model):
    """An error that leaves the device usable fails only its own batch."""
    launch = sensevoice_model._launch
    calls = []

    def fail_once(jobs):
        calls.append(len(jobs))
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return launch(jobs)

    monkeypatch.setattr(sensevoice_model, "_launch", fail_once)
    with pytest.raises(RuntimeError, match="out of memory"):
        sensevoice_model.transcribe(AUDIO.read_bytes())
    result = sensevoice_model.transcribe(AUDIO.read_bytes(), language="en")
    assert normalized(result.text).startswith(TRANSCRIPT)
    assert sensevoice_model.alive


def test_qwen3_asr_worker_stops(monkeypatch):
    require(QWEN3_ASR, AUDIO)
    transcriber = Qwen3ASRTranscriber(QWEN3_ASR, cuda_graphs=False, max_batch_size=4)
    result = transcriber.transcribe(AUDIO.read_bytes(), language="en")
    assert normalized(result.text).startswith(TRANSCRIPT)

    stop_worker(monkeypatch, qwen3_asr, transcriber, "_advance")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        transcriber.transcribe(AUDIO.read_bytes())
    with pytest.raises(RuntimeError, match="the decoding worker has stopped"):
        transcriber.transcribe(AUDIO.read_bytes())
    assert not transcriber.alive

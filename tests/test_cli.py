"""Command-line options and the server's exit on a model failure."""

from __future__ import annotations

import threading

import pytest

from asrback.cli import parse_args, serve


def test_limits_default_on_and_zero_turns_them_off():
    args = parse_args([])
    assert args.max_audio_seconds == 3600
    assert args.max_concurrent_requests == 512
    assert args.max_upload_mb == 512
    args = parse_args(
        ["--max-audio-seconds", "0", "--max-concurrent-requests", "0"]
        + ["--max-upload-mb", "0"]
    )
    assert args.max_audio_seconds is None
    assert args.max_concurrent_requests is None
    assert args.max_upload_mb is None


def test_negative_limit_is_an_error():
    with pytest.raises(SystemExit):
        parse_args(["--max-upload-mb", "-1"])


class FakeServer:
    def __init__(self) -> None:
        self.should_exit = False

    def run(self) -> None:
        while not self.should_exit:
            threading.Event().wait(0.01)


class FakeTranscriber:
    alive = True


def test_serve_exits_with_an_error_once_the_model_stops():
    transcriber = FakeTranscriber()
    threading.Timer(0.05, setattr, (transcriber, "alive", False)).start()
    assert serve(FakeServer(), transcriber, watch_seconds=0.01) == 1


def test_serve_exits_cleanly_when_stopped():
    server = FakeServer()
    threading.Timer(0.05, setattr, (server, "should_exit", True)).start()
    assert serve(server, FakeTranscriber(), watch_seconds=0.01) == 0

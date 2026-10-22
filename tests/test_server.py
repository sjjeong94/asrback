"""The HTTP layer, against a fake transcriber."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from asrback.server import ConcurrencyLimit, create_app, format_timestamp
from asrback.transcription import Segment, Transcription


class FakeTranscriber:
    max_batch_size = 4
    languages = {"en": 0, "ko": 1}
    task_ids = {"transcribe": 0}

    def __init__(self) -> None:
        self.alive = True
        self.calls: list[dict] = []

    def transcribe(self, audio, **options) -> Transcription:
        self.calls.append(options)
        if options["task"] not in self.task_ids:
            raise ValueError("task must be 'transcribe' for this model")
        segments = [
            Segment(0, 0, 0.0, 30.0, "Hello", [1], 0.0, -0.1, 1.0, 0.01),
            Segment(1, 3000, 30.0, 31.5, "world.", [2], 0.0, -0.2, 1.0, 0.02),
        ]
        return Transcription("transcribe", "en", 31.5, "Hello world.", segments)


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def client(transcriber: FakeTranscriber):
    with TestClient(create_app(transcriber, "fake")) as client:
        yield client


def post(client: TestClient, path="/v1/audio/transcriptions", audio=b"RIFF", **form):
    form.setdefault("model", "whisper-1")
    return client.post(path, files={"file": ("a.wav", audio)}, data=form)


def test_json(client):
    response = post(client)
    assert response.status_code == 200
    assert response.json() == {"text": "Hello world."}


def test_verbose_json(client):
    body = post(client, response_format="verbose_json").json()
    assert body["language"] == "en"
    assert body["duration"] == 31.5
    assert [segment["text"] for segment in body["segments"]] == ["Hello", "world."]


def test_text(client):
    assert post(client, response_format="text").text == "Hello world.\n"


def test_srt(client):
    assert post(client, response_format="srt").text == (
        "1\n00:00:00,000 --> 00:00:30,000\nHello\n\n"
        "2\n00:00:30,000 --> 00:00:31,500\nworld.\n"
    )


def test_vtt(client):
    response = post(client, response_format="vtt")
    assert response.headers["content-type"].startswith("text/vtt")
    assert response.text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:30.000\nHello")


def test_options_reach_the_transcriber(client, transcriber):
    post(client, language="ko", prompt="names", temperature="0.5")
    assert transcriber.calls == [
        {"temperature": 0.5, "language": "ko", "task": "transcribe", "prompt": "names"}
    ]


def test_empty_language_means_detect(client, transcriber):
    post(client, language="")
    assert transcriber.calls[0]["language"] is None


@pytest.mark.parametrize(
    "form, param",
    [
        ({"response_format": "xml"}, "response_format"),
        ({"temperature": "1.5"}, "temperature"),
        ({"language": "xx"}, "language"),
        ({"timestamp_granularities[]": "word"}, "timestamp_granularities"),
    ],
)
def test_invalid_request(client, form, param):
    response = post(client, **form)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param


def test_empty_file(client):
    response = post(client, audio=b"")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "file"


def test_missing_model_field(client):
    response = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "model"


def test_value_error_is_a_400(client):
    # The fake, like most models, cannot translate.
    response = post(client, path="/v1/audio/translations")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_models(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "fake"
    assert client.get("/v1/models/fake").json()["id"] == "fake"
    assert client.get("/v1/models/other").status_code == 404


def test_health(client, transcriber):
    assert client.get("/health").json() == {"status": "ok"}
    transcriber.alive = False
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "the decoding worker has stopped"


def test_format_timestamp():
    assert format_timestamp(3725.4567, ",") == "01:02:05,457"
    assert format_timestamp(0.0, ".") == "00:00:00.000"


def test_concurrency_limit():
    """Audio requests over the limit get a 429; other paths are not counted."""
    release = asyncio.Event()

    async def app(scope, receive, send):
        if scope["path"].startswith("/v1/audio/"):
            await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def request(middleware, path: str) -> int:
        statuses = []

        async def send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {"type": "http", "path": path, "method": "POST", "headers": []}
        await middleware(scope, receive, send)
        return statuses[0]

    async def main():
        middleware = ConcurrencyLimit(app, limit=1)
        first = asyncio.create_task(request(middleware, "/v1/audio/transcriptions"))
        await asyncio.sleep(0)
        assert middleware.active == 1
        assert await request(middleware, "/v1/audio/translations") == 429
        assert await request(middleware, "/health") == 200
        release.set()
        assert await first == 200
        assert middleware.active == 0
        assert await request(middleware, "/v1/audio/transcriptions") == 200

    asyncio.run(main())


def test_concurrency_limit_in_the_app(transcriber):
    started, release = threading.Event(), threading.Event()
    transcribe = transcriber.transcribe

    def blocking_transcribe(audio, **options):
        started.set()
        release.wait()
        return transcribe(audio, **options)

    transcriber.transcribe = blocking_transcribe
    app = create_app(transcriber, "fake", max_concurrent_requests=1)
    with TestClient(app) as client:
        responses = []
        first = threading.Thread(target=lambda: responses.append(post(client)))
        first.start()
        assert started.wait(10)
        busy = post(client)
        assert busy.status_code == 429
        assert busy.headers["retry-after"] == "1"
        assert busy.json()["error"]["type"] == "rate_limit_exceeded"
        release.set()
        first.join()
        assert responses[0].status_code == 200
        assert post(client).status_code == 200

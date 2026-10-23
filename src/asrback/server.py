"""OpenAI-compatible ``/v1/audio`` endpoints backed by :class:`Transcriber`."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

import anyio.to_thread
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import __version__
from .model_process import ProcessTranscriber
from .models import (
    Gemma4Transcriber,
    Qwen3ASRTranscriber,
    SenseVoiceTranscriber,
    Transcriber,
)
from .transcription import ModelUnavailableError, Transcription

RESPONSE_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        param: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.param = param
        self.error_type = error_type


def error_response(
    message: str, status_code: int, param: str | None, error_type: str
) -> JSONResponse:
    body = {"message": message, "type": error_type, "param": param, "code": None}
    return JSONResponse({"error": body}, status_code=status_code)


class ConcurrencyLimit:
    """ASGI middleware: answer 429 to an audio request while ``limit`` others are
    in flight, before its upload is read, so overload cannot pile up requests
    (and their uploads) inside the server."""

    def __init__(self, app, limit: int) -> None:
        self.app = app
        self.limit = limit
        # Requests run on one event loop, so a plain counter is enough.
        self.active = 0

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/audio/"):
            await self.app(scope, receive, send)
            return
        if self.active >= self.limit:
            response = error_response(
                f"The server is busy ({self.limit} requests in flight); retry later.",
                429,
                None,
                "rate_limit_exceeded",
            )
            response.headers["Retry-After"] = "1"
            await response(scope, receive, send)
            return
        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1


class BodyTooLarge(Exception):
    pass


class BodySizeLimit:
    """ASGI middleware: answer 413 to an audio request whose body is over
    ``limit`` bytes, by its ``Content-Length`` before reading it, or once a
    chunked body passes the limit, so an upload cannot fill memory or disk."""

    def __init__(self, app, limit: int) -> None:
        self.app = app
        self.limit = limit

    def _response(self) -> JSONResponse:
        return error_response(
            f"The request body is larger than {self.limit} bytes.",
            413,
            "file",
            "invalid_request_error",
        )

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/audio/"):
            await self.app(scope, receive, send)
            return
        for name, value in scope["headers"]:
            if name == b"content-length" and value.isdigit():
                if int(value) > self.limit:
                    await self._response()(scope, receive, send)
                    return
        received = 0
        exceeded = False
        started = False

        async def limited_receive():
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    exceeded = True
                    raise BodyTooLarge
            return message

        async def guarded_send(message):
            nonlocal started
            # The body parser may turn the error into its own response: replace it.
            if exceeded:
                return
            started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except BodyTooLarge:
            pass
        if exceeded and not started:
            await self._response()(scope, receive, send)


def format_timestamp(seconds: float, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def to_srt(result: Transcription) -> str:
    blocks = []
    for index, segment in enumerate(result.segments, start=1):
        start = format_timestamp(segment.start, ",")
        end = format_timestamp(segment.end, ",")
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(blocks)


def to_vtt(result: Transcription) -> str:
    blocks = ["WEBVTT\n"]
    for segment in result.segments:
        start = format_timestamp(segment.start, ".")
        end = format_timestamp(segment.end, ".")
        blocks.append(f"{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(blocks)


def render(result: Transcription, response_format: str) -> Response:
    if response_format == "json":
        return JSONResponse({"text": result.text})
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": result.task,
                "language": result.language,
                "duration": result.duration,
                "text": result.text,
                "segments": [asdict(segment) for segment in result.segments],
            }
        )
    if response_format == "text":
        return PlainTextResponse(result.text + "\n")
    if response_format == "srt":
        return PlainTextResponse(to_srt(result))
    return PlainTextResponse(to_vtt(result), media_type="text/vtt")


def create_app(
    transcriber: (
        Transcriber
        | SenseVoiceTranscriber
        | Qwen3ASRTranscriber
        | Gemma4Transcriber
        | ProcessTranscriber
    ),
    model_name: str,
    max_concurrent_requests: int | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    """Build the app. Requests may name any model; ``model_name`` is what we list.
    With ``max_concurrent_requests``, audio requests beyond it get a 429; with
    ``max_upload_bytes``, audio requests with a larger body get a 413."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Sync endpoints run in AnyIO's thread pool (40 threads by default) and each
        # request holds a thread while it waits for its batch. Allow several batches'
        # worth so the batcher can group windows of similar length.
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(
            limiter.total_tokens,
            8 * transcriber.max_batch_size,
            max_concurrent_requests or 0,
        )
        yield

    app = FastAPI(title="asrback", version=__version__, lifespan=lifespan)
    if max_upload_bytes is not None:
        app.add_middleware(BodySizeLimit, limit=max_upload_bytes)
    # Added last, so it runs first: a busy server answers before reading anything.
    if max_concurrent_requests is not None:
        app.add_middleware(ConcurrencyLimit, limit=max_concurrent_requests)
    created = int(time.time())

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return error_response(exc.message, exc.status_code, exc.param, exc.error_type)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error = exc.errors()[0]
        param = str(error["loc"][-1]) if error.get("loc") else None
        return error_response(error["msg"], 400, param, "invalid_request_error")

    def run(
        file: UploadFile,
        response_format: str,
        temperature: float,
        **options,
    ) -> Response:
        if response_format not in RESPONSE_FORMATS:
            raise APIError(
                f"response_format must be one of {', '.join(RESPONSE_FORMATS)}",
                param="response_format",
            )
        if not 0 <= temperature <= 1:
            raise APIError("temperature must be between 0 and 1", param="temperature")
        audio = file.file.read()
        if not audio:
            raise APIError("The uploaded file is empty.", param="file")
        try:
            result = transcriber.transcribe(audio, temperature=temperature, **options)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        except ModelUnavailableError as exc:
            raise APIError(str(exc), 503, error_type="server_error") from exc
        except Exception as exc:
            logger.exception("Transcription failed")
            raise APIError(
                "The server failed to transcribe the audio.",
                500,
                error_type="server_error",
            ) from exc
        return render(result, response_format)

    # Plain ``def`` endpoints run in FastAPI's thread pool, keeping the event loop
    # responsive while a request decodes.
    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamp_granularities: Optional[list[str]] = Form(
            None, alias="timestamp_granularities[]"
        ),
    ) -> Response:
        if timestamp_granularities and "word" in timestamp_granularities:
            raise APIError(
                "Word-level timestamps are not supported.",
                param="timestamp_granularities",
            )
        language = language or None
        if language is not None and language not in transcriber.languages:
            raise APIError(f"Unsupported language {language!r}.", param="language")
        return run(
            file,
            response_format,
            temperature,
            language=language,
            task="transcribe",
            prompt=prompt,
        )

    @app.post("/v1/audio/translations")
    def translations(
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
    ) -> Response:
        return run(file, response_format, temperature, task="translate", prompt=prompt)

    def model_card() -> dict:
        return {
            "id": model_name,
            "object": "model",
            "created": created,
            "owned_by": "asrback",
        }

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"object": "list", "data": [model_card()]}

    @app.get("/v1/models/{model_id}")
    def retrieve_model(model_id: str) -> dict:
        if model_id != model_name:
            raise APIError(
                f"The model {model_id!r} does not exist.",
                status_code=404,
                param="model",
            )
        return model_card()

    @app.get("/health")
    def health() -> Response:
        if not transcriber.alive:
            detail = (
                "the model process has exited"
                if isinstance(transcriber, ProcessTranscriber)
                else "the decoding worker has stopped"
            )
            return JSONResponse({"status": "error", "detail": detail}, status_code=503)
        return JSONResponse({"status": "ok"})

    return app

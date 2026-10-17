"""OpenAI-compatible ``/v1/audio`` endpoints backed by :class:`Transcriber`."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import __version__
from .models import SenseVoiceTranscriber, Transcriber
from .transcription import Transcription

RESPONSE_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")


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
    transcriber: Transcriber | SenseVoiceTranscriber, model_name: str
) -> FastAPI:
    """Build the app. Requests may name any model; ``model_name`` is what we list."""
    app = FastAPI(title="asrback", version=__version__)
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
    def health() -> dict:
        return {"status": "ok"}

    return app

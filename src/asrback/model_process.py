"""Run a transcriber in a child process.

In the server process, the event loop (HTTP parsing, multipart forms) holds the GIL
most of the time, and the decoding worker thread waits for it before launching each
batch's kernels, leaving the GPU idle. :class:`ProcessTranscriber` loads the model in
a child process instead: the server only forwards each upload's bytes and gets the
:class:`Transcription` back, while audio decoding, feature extraction and batching
happen in the child.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import Connection

import torch

from .transcription import AudioSource, ModelUnavailableError, Transcription

# Read once from the loaded transcriber and served by the proxy.
ATTRIBUTES = (
    "device",
    "dtype",
    "max_batch_size",
    "cuda_graphs",
    "languages",
    "task_ids",
)
# How often the model process checks that its decoding worker still runs.
WATCH_SECONDS = 1.0


class ProcessTranscriber:
    """Same interface as the transcribers (``transcribe`` is thread-safe) for a
    model loaded by :func:`asrback.models.load_transcriber` in a child process.

    Loading errors (``FileNotFoundError``, ``ValueError``) are raised here, and
    errors of a ``transcribe`` call are raised by that call.
    """

    def __init__(self, model_dir: str | os.PathLike, **options) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe()
        precision = torch.get_float32_matmul_precision()
        self._process = context.Process(
            target=_serve,
            args=(child_connection, model_dir, options, precision),
            name="asrback-model",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        try:
            status, payload = self._connection.recv()
        except EOFError:
            raise RuntimeError("the model process exited while loading") from None
        if status == "error":
            raise payload
        for name, value in payload.items():
            setattr(self, name, value)

        self._ids = itertools.count()
        self._futures: dict[int, Future] = {}
        self._send_lock = threading.Lock()
        self._closed = False
        threading.Thread(
            target=self._receive, name="asrback-model-results", daemon=True
        ).start()

    @property
    def alive(self) -> bool:
        """Whether the model process is still running and serving requests."""
        return not self._closed and self._process.is_alive()

    def transcribe(self, audio: AudioSource, **options) -> Transcription:
        future: Future = Future()
        with self._send_lock:
            if self._closed:
                raise ModelUnavailableError("the model process has exited")
            request_id = next(self._ids)
            self._futures[request_id] = future
            self._connection.send((request_id, audio, options))
        return future.result()

    def _receive(self) -> None:
        while True:
            try:
                request_id, ok, payload = self._connection.recv()
            except (EOFError, OSError):
                break
            future = self._futures.pop(request_id)
            if ok:
                future.set_result(payload)
            else:
                future.set_exception(payload)
        with self._send_lock:
            self._closed = True
            futures, self._futures = self._futures, {}
        for future in futures.values():
            future.set_exception(ModelUnavailableError("the model process has exited"))


def _serve(
    connection: Connection, model_dir: str | os.PathLike, options: dict, precision: str
) -> None:
    """Child process: load the model, then run each request on a thread pool."""
    # Ctrl+C reaches the whole process group; the server process shuts us down.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_float32_matmul_precision(precision)
    from .models import load_transcriber

    try:
        transcriber = load_transcriber(model_dir, **options)
    except Exception as exc:
        connection.send(("error", exc))
        return
    connection.send(
        ("ready", {name: getattr(transcriber, name) for name in ATTRIBUTES})
    )

    send_lock = threading.Lock()

    def watch() -> None:
        # A stopped worker cannot be restarted: exit, so the server process fails
        # the pending requests and reports the model process as gone.
        while transcriber.alive:
            time.sleep(WATCH_SECONDS)
        print(
            "asrback: the decoding worker stopped; exiting the model process",
            file=sys.stderr,
            flush=True,
        )
        os._exit(1)

    threading.Thread(target=watch, name="asrback-watch", daemon=True).start()

    def run(request_id: int, audio: AudioSource, options: dict) -> None:
        try:
            response = (request_id, True, transcriber.transcribe(audio, **options))
        except Exception as exc:
            response = (request_id, False, exc)
        with send_lock:
            try:
                connection.send(response)
            except Exception as exc:  # e.g. an exception that cannot be pickled
                connection.send((request_id, False, RuntimeError(repr(exc))))

    # Requests wait on their batch in these threads, as in the server's thread pool.
    with ThreadPoolExecutor(
        max_workers=8 * transcriber.max_batch_size, thread_name_prefix="asrback-request"
    ) as pool:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            pool.submit(run, *request)

"""SenseVoiceSmall (FunASR) in plain PyTorch: SAN-M encoder with greedy CTC decoding.

The model reads 80-bin fbank features stacked 7 frames at a time (every 6 frames)
and normalized with ``am.mvn``. Four learned queries go in front of the features
(language, event, emotion, text normalization); the CTC output at those positions
names the detected language and tags, and the rest is the transcript.
"""

from __future__ import annotations

import math
import os
import queue
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import asrfront
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sentencepiece import SentencePieceProcessor
from torch import Tensor, nn
from torch.nn.attention.varlen import varlen_attn

from ..transcription import (
    DTYPES,
    AudioSource,
    Segment,
    Transcription,
    Worker,
    compression_ratio,
    gather_batch,
    load_audio,
    raise_if_device_failed,
    to_device,
)

SAMPLE_RATE = asrfront.sensevoice.SAMPLE_RATE
WINDOW_SAMPLES = 30 * SAMPLE_RATE
MIN_TAIL_SAMPLES = SAMPLE_RATE // 10
HOP_LENGTH = 160  # 10 ms fbank frames, the unit of ``Segment.seek``
# The worker picks each batch from up to this many batches' worth of queued windows.
PENDING_FACTOR = 4
# CTC logits are computed for at most about this many frames (batch x length) at once.
CTC_CHUNK_FRAMES = 2048

REQUIRED_FILES = (
    "model.pt",
    "config.yaml",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
)

# Rows of ``SenseVoiceModel.embed`` (16 in all), as in FunASR's SenseVoiceSmall.
QUERY_EMBEDDINGS = 16
LANGUAGE_QUERIES = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12}
EVENT_QUERY, EMOTION_QUERY = 1, 2
TEXT_NORM_QUERIES = {"withitn": 14, "woitn": 15}
QUERY_COUNT = 4
BLANK_ID = 0


class SANMAttention(nn.Module):
    """Self-attention plus a depthwise convolution (FSMN memory) over the values.

    With ``varlen`` set (flash attention on CUDA, bfloat16), a padded batch runs
    attention as variable-length sequences: each row's keys stop at its length, so
    no mask is needed. ``scaled_dot_product_attention`` with a padding mask would
    use the memory-efficient kernel, about 2.5x slower on these shapes; without
    one it already uses flash attention.
    """

    varlen = False

    def __init__(self, in_size: int, size: int, heads: int, kernel_size: int) -> None:
        super().__init__()
        self.heads = heads
        self.size = size
        self.linear_q_k_v = nn.Linear(in_size, size * 3)
        self.linear_out = nn.Linear(size, size)
        self.fsmn_block = nn.Conv1d(
            size, size, kernel_size, padding=0, groups=size, bias=False
        )
        left = (kernel_size - 1) // 2
        self.padding = (left, kernel_size - 1 - left)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        return x.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def forward(self, x: Tensor, mask: Tensor | None, lengths: Tensor | None) -> Tensor:
        """``mask`` is ``(batch, length)``, True for real frames, and ``lengths``
        (int32) counts them per row; both are None when every frame is real."""
        query, key, value = self.linear_q_k_v(x).split(self.size, dim=-1)

        keep = None if mask is None else mask[..., None].to(value.dtype)
        if keep is not None:
            value_masked = value * keep
        else:
            value_masked = value
        memory = F.pad(value_masked.transpose(1, 2), self.padding)
        memory = self.fsmn_block(memory).transpose(1, 2) + value_masked
        if keep is not None:
            memory = memory * keep

        batch, length, _ = x.shape
        if self.varlen and lengths is not None:
            # Rows are laid out back to back, every ``length`` frames.
            starts = torch.arange(
                0, (batch + 1) * length, length, dtype=torch.int32, device=x.device
            )
            output = varlen_attn(
                *(
                    t.reshape(batch * length, self.heads, -1)
                    for t in (query, key, value)
                ),
                starts,
                starts,
                length,
                length,
                seqused_k=lengths,
            )
        else:
            attention_mask = None if mask is None else mask[:, None, None, :]
            output = F.scaled_dot_product_attention(
                self._split_heads(query),
                self._split_heads(key),
                self._split_heads(value),
                attn_mask=attention_mask,
            ).transpose(1, 2)
        output = output.reshape(batch, length, self.size)
        return self.linear_out(output) + memory


class FeedForward(nn.Module):
    def __init__(self, size: int, hidden: int) -> None:
        super().__init__()
        self.w_1 = nn.Linear(size, hidden)
        self.w_2 = nn.Linear(hidden, size)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(F.relu(self.w_1(x)))


class EncoderLayer(nn.Module):
    def __init__(
        self, in_size: int, size: int, heads: int, hidden: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_size)
        self.self_attn = SANMAttention(in_size, size, heads, kernel_size)
        self.norm2 = nn.LayerNorm(size)
        self.feed_forward = FeedForward(size, hidden)
        self.residual = in_size == size

    def forward(self, x: Tensor, mask: Tensor | None, lengths: Tensor | None) -> Tensor:
        attention = self.self_attn(self.norm1(x), mask, lengths)
        # The first layer changes width (560 -> 512) and has no residual path.
        x = x + attention if self.residual else attention
        return x + self.feed_forward(self.norm2(x))


def sinusoidal_positions(length: int, depth: int, device: torch.device) -> Tensor:
    """FunASR's SinusoidalPositionEncoder for positions ``1..length``."""
    positions = torch.arange(1, length + 1, device=device, dtype=torch.float32)
    increment = math.log(10000.0) / (depth / 2 - 1)
    inverse = torch.exp(
        torch.arange(depth // 2, device=device, dtype=torch.float32) * -increment
    )
    scaled = positions[:, None] * inverse[None, :]
    return torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=1)


class SenseVoiceEncoder(nn.Module):
    def __init__(self, input_size: int, config: dict) -> None:
        super().__init__()
        size = config["output_size"]
        heads = config["attention_heads"]
        hidden = config["linear_units"]
        kernel = config["kernel_size"]
        if config.get("sanm_shfit", 0):
            raise ValueError("sanm_shfit is not supported")
        self.size = size
        self.encoders0 = nn.ModuleList(
            [EncoderLayer(input_size, size, heads, hidden, kernel)]
        )
        self.encoders = nn.ModuleList(
            EncoderLayer(size, size, heads, hidden, kernel)
            for _ in range(config["num_blocks"] - 1)
        )
        self.tp_encoders = nn.ModuleList(
            EncoderLayer(size, size, heads, hidden, kernel)
            for _ in range(config["tp_blocks"])
        )
        self.after_norm = nn.LayerNorm(size)
        self.tp_norm = nn.LayerNorm(size)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        positions = sinusoidal_positions(x.shape[1], x.shape[2], x.device)
        x = x * self.size**0.5 + positions.to(x.dtype)
        lengths = None if mask is None else mask.sum(dim=1, dtype=torch.int32)
        for layer in self.encoders0:
            x = layer(x, mask, lengths)
        for layer in self.encoders:
            x = layer(x, mask, lengths)
        x = self.after_norm(x)
        for layer in self.tp_encoders:
            x = layer(x, mask, lengths)
        return self.tp_norm(x)


class CTCHead(nn.Module):
    def __init__(self, size: int, vocab_size: int) -> None:
        super().__init__()
        self.ctc_lo = nn.Linear(size, vocab_size)


class SenseVoiceModel(nn.Module):
    """Module names match the FunASR checkpoint (``model.pt``)."""

    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        frontend = config["frontend_conf"]
        input_size = frontend["n_mels"] * frontend["lfr_m"]
        self.encoder = SenseVoiceEncoder(input_size, config["encoder_conf"])
        self.ctc = CTCHead(config["encoder_conf"]["output_size"], vocab_size)
        self.embed = nn.Embedding(QUERY_EMBEDDINGS, input_size)

    def forward(self, features: Tensor, queries: Tensor, mask: Tensor | None) -> Tensor:
        """Encoder output for ``queries`` (batch, 4) followed by ``features``."""
        x = torch.cat((self.embed(queries).to(features.dtype), features), dim=1)
        return self.encoder(x, mask)


def varlen_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether ``varlen_attn`` (flash attention) runs on this device and dtype."""
    if device.type != "cuda" or dtype != torch.bfloat16:
        return False
    x = torch.zeros((2, 1, 64), dtype=dtype, device=device)
    starts = torch.tensor([0, 2], dtype=torch.int32, device=device)
    try:
        varlen_attn(x, x, x, starts, starts, 2, 2)
    except RuntimeError:
        return False
    return True


def greedy_ctc(logits: Tensor, no_speech_id: int) -> tuple[Tensor, Tensor, Tensor]:
    """Best token and its log-probability per frame, and P(<|nospeech|>) at the
    first frame, without materializing float32 log-probabilities for the vocab."""
    best_logits, best = logits.max(dim=-1)
    normalizer = torch.logsumexp(logits, dim=-1).float()
    best_log_probs = best_logits.float() - normalizer
    no_speech = (logits[:, 0, no_speech_id].float() - normalizer[:, 0]).exp()
    return best, best_log_probs, no_speech


@dataclass
class _Job:
    features: np.ndarray
    language: str | None
    future: Future = field(default_factory=Future)


@dataclass
class _Batch:
    """A batch whose forward pass is queued on the device; its outputs arrive in
    host memory once ``done`` (a CUDA event, or None on the CPU) completes."""

    jobs: list[_Job]
    lengths: list[int]
    best: Tensor
    avg_logprobs: Tensor
    no_speech: Tensor
    done: torch.cuda.Event | None


@dataclass
class _Result:
    language: str
    tokens: list[int]
    text: str
    avg_logprob: float
    no_speech_prob: float


def take_similar(pending: list[_Job], count: int) -> list[_Job]:
    """Remove and return the oldest job plus up to ``count - 1`` others that pad
    little against it: the longest jobs no longer than it first (they pad to its
    length), then the shortest longer ones. The oldest job is always served."""
    target = len(pending[0].features)

    def padding_order(index: int) -> tuple[bool, int]:
        length = len(pending[index].features)
        return (length > target, abs(length - target))

    order = sorted(range(len(pending)), key=padding_order)
    chosen = set(order[:count])
    jobs = [job for i, job in enumerate(pending) if i in chosen]
    pending[:] = [job for i, job in enumerate(pending) if i not in chosen]
    return jobs


class SenseVoiceTranscriber:
    """Loads SenseVoiceSmall once and transcribes audio.

    Same interface as :class:`asrback.Transcriber`: ``transcribe`` is thread-safe
    and windows from concurrent calls are decoded in batches. Decoding is greedy
    CTC, so ``prompt`` and ``temperature`` have no effect, and translation is not
    supported. ``use_itn`` asks the model for punctuation and inverse text
    normalization ("withitn"). ``compile_encoder`` compiles the encoder layers
    with ``torch.compile`` (dynamic shapes) at startup.
    """

    cuda_graphs = False

    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        max_batch_size: int = 32,
        batch_wait_ms: float = 5.0,
        use_itn: bool = True,
        compile_encoder: bool = False,
        max_audio_seconds: float | None = None,
    ):
        model_dir = Path(model_dir)
        for filename in REQUIRED_FILES:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing model file: {model_dir / filename}")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if max_audio_seconds is not None and max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be greater than zero")

        self.model_dir = model_dir
        with (model_dir / "config.yaml").open(encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.tokenizer = SentencePieceProcessor(
            model_file=str(model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        frontend = self.config["frontend_conf"]
        self._fbank_options = {
            "cmvn": asrfront.sensevoice.load_cmvn(model_dir / "am.mvn"),
            "n_mels": frontend["n_mels"],
            "lfr_m": frontend["lfr_m"],
            "lfr_n": frontend["lfr_n"],
        }

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif isinstance(dtype, str):
            dtype = DTYPES[dtype]
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.max_audio_seconds = max_audio_seconds

        model = SenseVoiceModel(self.config, self.tokenizer.get_piece_size())
        # model.pt is a pickle; weights_only refuses anything but tensors.
        state_dict = torch.load(
            model_dir / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        self.model = model.eval().to(self.device, self.dtype)
        if varlen_supported(self.device, self.dtype):
            for module in self.model.modules():
                if isinstance(module, SANMAttention):
                    module.varlen = True

        self.languages = {
            name: index for name, index in LANGUAGE_QUERIES.items() if name != "auto"
        }
        self.task_ids = {"transcribe": 0}
        self._text_norm = TEXT_NORM_QUERIES["withitn" if use_itn else "woitn"]
        self._language_tokens = {
            self.tokenizer.piece_to_id(f"<|{name}|>"): name for name in self.languages
        }
        self._no_speech_id = self.tokenizer.piece_to_id("<|nospeech|>")
        self._special_ids = {
            index
            for index in range(self.tokenizer.get_piece_size())
            if self.tokenizer.id_to_piece(index).startswith("<|")
        }

        self.compile_encoder = compile_encoder
        if compile_encoder:
            self._compile_encoder()

        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._worker = Worker(self._serve, "asrback-sensevoice")

    @property
    def alive(self) -> bool:
        """Whether the decoding worker still runs."""
        return self._worker.alive

    def transcribe(
        self,
        audio: AudioSource,
        *,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> Transcription:
        """Decode ``audio`` (a path, encoded file bytes, or a 16 kHz waveform)."""
        if task != "transcribe":
            raise ValueError(f"task must be 'transcribe' for this model, got {task!r}")
        if language is not None and language not in self.languages:
            choices = ", ".join(sorted(self.languages))
            raise ValueError(
                f"Unsupported language {language!r}. Available codes: {choices}"
            )
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        waveform = load_audio(audio, self.max_audio_seconds)
        duration = len(waveform) / SAMPLE_RATE

        # Windows are independent, so all of them are queued at once.
        windows = []
        for seek in range(0, max(len(waveform), 1), WINDOW_SAMPLES):
            window = waveform[seek : seek + WINDOW_SAMPLES]
            if windows and len(window) < MIN_TAIL_SAMPLES:
                break
            features = asrfront.sensevoice.fbank(window, **self._fbank_options)
            job = _Job(features, language)
            self._queue.put(job)
            windows.append((seek, len(window), job))

        segments = []
        detected = None
        for seek, length, job in windows:
            result: _Result = self._worker.wait(job.future)
            detected = detected or result.language
            segments.append(
                Segment(
                    id=len(segments),
                    seek=seek // HOP_LENGTH,
                    start=round(seek / SAMPLE_RATE, 3),
                    end=round((seek + length) / SAMPLE_RATE, 3),
                    text=result.text,
                    tokens=result.tokens,
                    temperature=0.0,
                    avg_logprob=result.avg_logprob,
                    compression_ratio=compression_ratio(result.text),
                    no_speech_prob=result.no_speech_prob,
                )
            )

        text = " ".join(segment.text for segment in segments if segment.text)
        return Transcription(
            task=task,
            language=language or detected or "en",
            duration=round(duration, 3),
            text=text,
            segments=segments,
        )

    def _serve(self) -> None:
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        pending: list[_Job] = []
        # The batch running on the device: the next one is queued before reading its
        # results, so the device never waits for the host to pad and parse.
        running: _Batch | None = None
        while True:
            if not pending and running is None:
                pending = gather_batch(
                    self._queue, self.max_batch_size, self.batch_wait
                )
            # Look at everything already queued so similar lengths share a batch.
            while len(pending) < PENDING_FACTOR * self.max_batch_size:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            launched = None
            if pending:
                jobs = take_similar(pending, self.max_batch_size)
                try:
                    launched = self._launch(jobs)
                except BaseException as exc:
                    for job in jobs:
                        job.future.set_exception(exc)
                    raise_if_device_failed(self.device)
            if running is not None:
                try:
                    results = self._collect(running)
                except BaseException as exc:
                    for job in running.jobs:
                        job.future.set_exception(exc)
                    raise_if_device_failed(self.device)
                else:
                    for job, result in zip(running.jobs, results):
                        job.future.set_result(result)
            running = launched

    @torch.inference_mode()
    def _compile_encoder(self) -> None:
        # Each layer is compiled on its own, so the identical layers share one graph
        # and compiling takes seconds. Batches are not padded to fixed shapes, hence
        # dynamic shapes.
        encoder = self.model.encoder
        for layer in (*encoder.encoders0, *encoder.encoders, *encoder.tp_encoders):
            layer.compile(dynamic=True)
        # Compile now rather than on the first requests: a batch of one (size-1
        # dimensions are specialized), and larger ones with and without a mask.
        frontend = self.config["frontend_conf"]
        width = frontend["n_mels"] * frontend["lfr_m"]
        for lengths, masked in (([100], False), ([100, 100], False), ([100, 90], True)):
            features = torch.zeros(
                (len(lengths), max(lengths), width),
                dtype=self.dtype,
                device=self.device,
            )
            queries = torch.zeros((len(lengths), QUERY_COUNT), dtype=torch.long)
            self._forward(
                features,
                queries.to(self.device),
                torch.tensor(lengths, device=self.device),
                masked,
            )

    def _forward(
        self, features: Tensor, queries: Tensor, lengths: Tensor, masked: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Best token per frame, mean speech log-probability and P(no speech) per
        row; ``lengths`` counts each row's feature frames."""
        frames = torch.arange(QUERY_COUNT + features.shape[1], device=self.device)
        ends = lengths + QUERY_COUNT
        mask = frames[None, :] < ends[:, None] if masked else None
        hidden = self.model(features, queries, mask)
        # Full-vocabulary logits for a whole batch take gigabytes; do a few rows at once.
        rows = max(1, CTC_CHUNK_FRAMES // hidden.shape[1])
        chunks = [
            greedy_ctc(self.model.ctc.ctc_lo(chunk), self._no_speech_id)
            for chunk in hidden.split(rows)
        ]
        best, best_log_probs, no_speech = (torch.cat(parts) for parts in zip(*chunks))
        speech = (frames[None, :] >= QUERY_COUNT) & (frames[None, :] < ends[:, None])
        avg_logprobs = (best_log_probs * speech).sum(dim=1) / lengths.clamp(min=1)
        return best, avg_logprobs, no_speech

    @torch.inference_mode()
    def _launch(self, jobs: list[_Job]) -> _Batch:
        """Queue the forward pass of ``jobs`` and the copy of its outputs to the
        host, without waiting for either."""
        batch_size = len(jobs)
        lengths = [len(job.features) for job in jobs]
        longest = max(lengths)
        width = jobs[0].features.shape[1]
        features = np.zeros((batch_size, longest, width), dtype=np.float32)
        for row, job in enumerate(jobs):
            features[row, : len(job.features)] = job.features
        queries = np.array(
            [
                [
                    LANGUAGE_QUERIES[job.language or "auto"],
                    EVENT_QUERY,
                    EMOTION_QUERY,
                    self._text_norm,
                ]
                for job in jobs
            ]
        )
        # Copied from pinned memory: a copy from pageable memory would wait for the
        # batch running before this one.
        outputs = self._forward(
            to_device(features, self.device).to(self.dtype),
            to_device(queries, self.device),
            to_device(lengths, self.device),
            masked=min(lengths) != longest,
        )
        if self.device.type != "cuda":
            return _Batch(jobs, lengths, *outputs, None)
        best, avg_logprobs, no_speech = (
            torch.empty(output.shape, dtype=output.dtype, pin_memory=True).copy_(
                output, non_blocking=True
            )
            for output in outputs
        )
        done = torch.cuda.Event()
        done.record()
        return _Batch(jobs, lengths, best, avg_logprobs, no_speech, done)

    def _collect(self, batch: _Batch) -> list[_Result]:
        """Wait for ``batch`` and turn its outputs into results."""
        if batch.done is not None:
            batch.done.synchronize()
        rows = batch.best.tolist()
        avg_logprobs = batch.avg_logprobs.tolist()
        no_speech = batch.no_speech.tolist()

        results = []
        for row, length, job in zip(range(len(batch.jobs)), batch.lengths, batch.jobs):
            frames = rows[row][: QUERY_COUNT + length]
            # Greedy CTC: merge repeats, then drop blanks.
            tokens = [
                token
                for index, token in enumerate(frames)
                if token != BLANK_ID and (index == 0 or token != frames[index - 1])
            ]
            language = next(
                (
                    self._language_tokens[t]
                    for t in tokens
                    if t in self._language_tokens
                ),
                job.language or "en",
            )
            text_tokens = [token for token in tokens if token not in self._special_ids]
            text = self.tokenizer.decode(text_tokens).strip()
            results.append(
                _Result(language, text_tokens, text, avg_logprobs[row], no_speech[row])
            )
        return results


def is_sensevoice(model_dir: str | os.PathLike) -> bool:
    """Whether ``model_dir`` holds a FunASR SenseVoice checkpoint."""
    config = Path(model_dir) / "config.yaml"
    if not config.is_file():
        return False
    with config.open(encoding="utf-8") as file:
        return str(yaml.safe_load(file).get("model", "")).startswith("SenseVoice")

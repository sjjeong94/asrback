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
import threading
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

from .transcriber import (
    DTYPES,
    AudioSource,
    Segment,
    Transcription,
    compression_ratio,
    gather_batch,
)

SAMPLE_RATE = asrfront.sensevoice.SAMPLE_RATE
WINDOW_SAMPLES = 30 * SAMPLE_RATE
MIN_TAIL_SAMPLES = SAMPLE_RATE // 10
HOP_LENGTH = 160  # 10 ms fbank frames, the unit of ``Segment.seek``

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


class LayerNorm(nn.LayerNorm):
    """Layer norm computed in float32, as the checkpoint was trained."""

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).type_as(x)


class SANMAttention(nn.Module):
    """Self-attention plus a depthwise convolution (FSMN memory) over the values."""

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

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """``mask`` is ``(batch, length)``, True for real frames (None: all real)."""
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

        attention_mask = None if mask is None else mask[:, None, None, :]
        output = F.scaled_dot_product_attention(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            attn_mask=attention_mask,
        )
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.size)
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
        self.norm1 = LayerNorm(in_size)
        self.self_attn = SANMAttention(in_size, size, heads, kernel_size)
        self.norm2 = LayerNorm(size)
        self.feed_forward = FeedForward(size, hidden)
        self.residual = in_size == size

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        attention = self.self_attn(self.norm1(x), mask)
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
        self.after_norm = LayerNorm(size)
        self.tp_norm = LayerNorm(size)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        positions = sinusoidal_positions(x.shape[1], x.shape[2], x.device)
        x = x * self.size**0.5 + positions.to(x.dtype)
        for layer in self.encoders0:
            x = layer(x, mask)
        for layer in self.encoders:
            x = layer(x, mask)
        x = self.after_norm(x)
        for layer in self.tp_encoders:
            x = layer(x, mask)
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
        """CTC log-probabilities for ``queries`` (batch, 4) followed by ``features``."""
        x = torch.cat((self.embed(queries).to(features.dtype), features), dim=1)
        hidden = self.encoder(x, mask)
        return self.ctc.ctc_lo(hidden).float().log_softmax(dim=-1)


@dataclass
class _Job:
    features: np.ndarray
    language: str | None
    future: Future = field(default_factory=Future)


@dataclass
class _Result:
    language: str
    tokens: list[int]
    text: str
    avg_logprob: float
    no_speech_prob: float


class SenseVoiceTranscriber:
    """Loads SenseVoiceSmall once and transcribes audio.

    Same interface as :class:`asrback.Transcriber`: ``transcribe`` is thread-safe
    and windows from concurrent calls are decoded in batches. Decoding is greedy
    CTC, so ``prompt`` and ``temperature`` have no effect, and translation is not
    supported. ``use_itn`` asks the model for punctuation and inverse text
    normalization ("withitn").
    """

    cuda_graphs = False

    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        max_batch_size: int = 16,
        batch_wait_ms: float = 5.0,
        use_itn: bool = True,
    ):
        model_dir = Path(model_dir)
        for filename in REQUIRED_FILES:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing model file: {model_dir / filename}")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")

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
        if dtype == torch.float16:
            # Encoder activations grow past 65504 in the last layers.
            raise ValueError("SenseVoice overflows in float16; use bfloat16 or float32")
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000

        model = SenseVoiceModel(self.config, self.tokenizer.get_piece_size())
        # model.pt is a pickle; weights_only refuses anything but tensors.
        state_dict = torch.load(
            model_dir / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        self.model = model.eval().to(self.device, self.dtype)

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

        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        threading.Thread(
            target=self._serve, name="asrback-sensevoice", daemon=True
        ).start()

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

        if isinstance(audio, np.ndarray) and audio.dtype != np.uint8:
            waveform = np.asarray(audio, dtype=np.float32)
        else:
            waveform = asrfront.load(audio)
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
            result: _Result = job.future.result()
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
        while True:
            jobs = gather_batch(self._queue, self.max_batch_size, self.batch_wait)
            try:
                results = self._decode_batch(jobs)
            except BaseException as exc:
                for job in jobs:
                    job.future.set_exception(exc)
                continue
            for job, result in zip(jobs, results):
                job.future.set_result(result)

    @torch.inference_mode()
    def _decode_batch(self, jobs: list[_Job]) -> list[_Result]:
        lengths = [len(job.features) for job in jobs]
        longest = max(lengths)
        width = jobs[0].features.shape[1]
        features = np.zeros((len(jobs), longest, width), dtype=np.float32)
        for row, job in enumerate(jobs):
            features[row, : len(job.features)] = job.features
        features = torch.from_numpy(features).to(self.device, self.dtype)

        queries = torch.tensor(
            [
                [
                    LANGUAGE_QUERIES[job.language or "auto"],
                    EVENT_QUERY,
                    EMOTION_QUERY,
                    self._text_norm,
                ]
                for job in jobs
            ],
            device=self.device,
        )
        mask = None
        if min(lengths) != longest:
            frames = torch.arange(QUERY_COUNT + longest, device=self.device)
            totals = torch.tensor(lengths, device=self.device) + QUERY_COUNT
            mask = frames[None, :] < totals[:, None]

        log_probs = self.model(features, queries, mask)
        best_log_probs, best = log_probs.max(dim=-1)
        no_speech = log_probs[:, 0, self._no_speech_id].exp()

        results = []
        rows = best.tolist()
        for row, length, job in zip(range(len(jobs)), lengths, jobs):
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
            speech = best_log_probs[row, QUERY_COUNT : QUERY_COUNT + length]
            avg_logprob = float(speech.mean()) if length else 0.0
            results.append(
                _Result(language, text_tokens, text, avg_logprob, float(no_speech[row]))
            )
        return results


def is_sensevoice(model_dir: str | os.PathLike) -> bool:
    """Whether ``model_dir`` holds a FunASR SenseVoice checkpoint."""
    config = Path(model_dir) / "config.yaml"
    if not config.is_file():
        return False
    with config.open(encoding="utf-8") as file:
        return str(yaml.safe_load(file).get("model", "")).startswith("SenseVoice")

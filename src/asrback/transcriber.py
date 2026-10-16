"""Greedy / sampled Whisper decoding over 30-second audio windows."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import asrfront
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import Tensor

from .model import WhisperForConditionalGeneration

AudioSource = Union[str, os.PathLike, bytes, bytearray, memoryview, np.ndarray]

SAMPLE_RATE = asrfront.whisper.SAMPLE_RATE
WINDOW_SAMPLES = asrfront.whisper.N_SAMPLES
# Trailing windows shorter than this are mostly padding and invite hallucinations.
MIN_TAIL_SAMPLES = SAMPLE_RATE // 10

REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


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


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token!r}")
    return token_id


def language_token_ids(tokenizer_config: dict) -> dict[str, int]:
    languages = {}
    for raw_id, entry in tokenizer_config.get("added_tokens_decoder", {}).items():
        content = entry.get("content", "")
        match = re.fullmatch(r"<\|([a-z]{2,3})\|>", content)
        if match:
            languages[match.group(1)] = int(raw_id)
    if not languages:
        raise ValueError("tokenizer_config.json contains no Whisper language tokens")
    return languages


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


class Transcriber:
    """Loads a Whisper checkpoint once and transcribes or translates audio."""

    def __init__(self, model_dir: str | os.PathLike, device: str | None = None):
        model_dir = Path(model_dir)
        for filename in REQUIRED_FILES:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing model file: {model_dir / filename}")

        self.model_dir = model_dir
        self.config = load_json(model_dir / "config.json")
        tokenizer_config = load_json(model_dir / "tokenizer_config.json")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model = WhisperForConditionalGeneration(self.config)
        state_dict = load_file(str(model_dir / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.model = model.eval().to(self.device)
        # The model and its decode caches are not safe to share across threads.
        self._lock = threading.Lock()

        # English-only (*.en) checkpoints have a smaller vocabulary, start decoding
        # without language/task tokens and cannot translate.
        self.multilingual = self.config["vocab_size"] >= 51865
        tasks = ("transcribe", "translate") if self.multilingual else ("transcribe",)
        if self.multilingual:
            self.languages = language_token_ids(tokenizer_config)
        else:
            self.languages = {"en": required_token_id(self.tokenizer, "<|en|>")}
        self.start_id = required_token_id(self.tokenizer, "<|startoftranscript|>")
        self.prev_id = required_token_id(self.tokenizer, "<|startofprev|>")
        self.no_timestamps_id = required_token_id(self.tokenizer, "<|notimestamps|>")
        self.task_ids = {
            task: required_token_id(self.tokenizer, f"<|{task}|>") for task in tasks
        }
        self.no_speech_id = self.tokenizer.token_to_id(
            "<|nospeech|>"
        ) or self.tokenizer.token_to_id("<|nocaptions|>")

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
        if task not in self.task_ids:
            choices = " or ".join(repr(name) for name in self.task_ids)
            raise ValueError(f"task must be {choices} for this model, got {task!r}")
        if language is not None and language not in self.languages:
            choices = ", ".join(sorted(self.languages))
            raise ValueError(
                f"Unsupported language {language!r}. Available codes: {choices}"
            )
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")

        if isinstance(audio, np.ndarray) and audio.dtype != np.uint8:
            waveform = np.asarray(audio, dtype=np.float32)
        else:
            waveform = asrfront.load(audio)
        duration = len(waveform) / SAMPLE_RATE

        with self._lock:
            return self._transcribe(
                waveform, duration, language, task, prompt, temperature, max_new_tokens
            )

    @torch.inference_mode()
    def _transcribe(
        self,
        waveform: np.ndarray,
        duration: float,
        language: str | None,
        task: str,
        prompt: str | None,
        temperature: float,
        max_new_tokens: int | None,
    ) -> Transcription:
        history = []
        if prompt and prompt.strip():
            history = self.tokenizer.encode(
                " " + prompt.strip(), add_special_tokens=False
            ).ids

        segments = []
        for seek in range(0, max(len(waveform), 1), WINDOW_SAMPLES):
            window = waveform[seek : seek + WINDOW_SAMPLES]
            if segments and len(window) < MIN_TAIL_SAMPLES:
                break
            encoder_output = self._encode(window)
            if language is None:
                language = (
                    self._detect_language(encoder_output) if self.multilingual else "en"
                )

            sot = [self.start_id]
            if self.multilingual:
                sot += [self.languages[language], self.task_ids[task]]
            sot.append(self.no_timestamps_id)
            max_target = self.config["max_target_positions"]
            # Condition on earlier text like openai-whisper: at most half the context.
            previous = history[-(max_target // 2 - 1) :]
            prefix = ([self.prev_id] + previous if previous else []) + sot
            available = max_target - len(prefix)
            limit = min(max_new_tokens or available, available)

            tokens, avg_logprob, no_speech_prob = self._generate(
                encoder_output, prefix, len(prefix) - len(sot), limit, temperature
            )
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            history.extend(tokens)
            segments.append(
                Segment(
                    id=len(segments),
                    seek=seek // asrfront.whisper.HOP_LENGTH,
                    start=round(seek / SAMPLE_RATE, 3),
                    end=round((seek + len(window)) / SAMPLE_RATE, 3),
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio(text),
                    no_speech_prob=no_speech_prob,
                )
            )

        return Transcription(
            task=task,
            language=language,
            duration=round(duration, 3),
            text="".join(segment.text for segment in segments).strip(),
            segments=segments,
        )

    def _encode(self, window: np.ndarray) -> Tensor:
        log_mel = asrfront.whisper.log_mel(window, n_mels=self.config["num_mel_bins"])
        mel = torch.from_numpy(log_mel).float().unsqueeze(0).to(self.device)
        return self.model.model.encoder(mel)

    def _detect_language(self, encoder_output: Tensor) -> str:
        token = torch.tensor([[self.start_id]], device=self.device)
        logits, _ = self.model.decode(token, encoder_output)
        language_ids = list(self.languages.values())
        best_index = logits[0, -1, language_ids].argmax().item()
        return list(self.languages.keys())[best_index]

    def _generate(
        self,
        encoder_output: Tensor,
        prefix: list[int],
        sot_index: int,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], float, float]:
        tokens = torch.tensor([prefix], device=self.device)
        logits, caches = self.model.decode(tokens, encoder_output)

        no_speech_prob = 0.0
        if self.no_speech_id is not None:
            sot_probs = logits[0, sot_index].float().softmax(dim=-1)
            no_speech_prob = float(sot_probs[self.no_speech_id])

        generated = []
        sum_logprob = 0.0
        eos = self.config["eos_token_id"]
        suppress = set(self.config.get("suppress_tokens", []))
        begin_suppress = set(self.config.get("begin_suppress_tokens", []))

        for step in range(max_new_tokens):
            next_logits = logits[0, -1].float()
            blocked = suppress | (begin_suppress if step == 0 else set())
            if blocked:
                next_logits[list(blocked)] = -math.inf
            next_logits[self.no_timestamps_id + 1 :] = -math.inf
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
            else:
                next_token = int(next_logits.argmax().item())
            sum_logprob += float(F.log_softmax(next_logits, dim=-1)[next_token])
            if next_token == eos:
                break
            generated.append(next_token)
            token = torch.tensor([[next_token]], device=self.device)
            logits, caches = self.model.decode(token, encoder_output, caches)

        avg_logprob = sum_logprob / (len(generated) + 1)
        return generated, avg_logprob, no_speech_prob

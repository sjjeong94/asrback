# asrback

OpenAI-compatible speech-to-text server for Whisper checkpoints in Hugging Face
SafeTensors format, implemented in plain PyTorch (no `transformers` or `openai-whisper`
at runtime).

## Setup

```bash
uv sync
scripts/download_whisper_hf.sh tiny   # -> models/whisper-tiny (tiny, base, small, *.en)
```

## Server

```bash
uv run asrback --model-dir models/whisper-tiny --port 8000
```

Endpoints:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /v1/models`
- `GET /health`

Supported response formats are `json`, `text`, `srt`, `vtt` and `verbose_json`.
Audio longer than 30 seconds is decoded in 30-second windows.

Concurrent requests are decoded together in batches. Options that affect speed:

| Option | Default | |
| --- | --- | --- |
| `--dtype` | `float16` on CUDA, else `float32` | Model precision (`float32`, `float16`, `bfloat16`) |
| `--max-batch-size` | `16` | Most requests decoded together |
| `--batch-wait-ms` | `5` | How long a batch waits to fill |
| `--cuda-graphs` / `--no-cuda-graphs` | on for CUDA | Replay each decoder step from a captured CUDA graph |
| `--compile-encoder` | off | `torch.compile` the encoder; slower startup, faster for larger models |

Any OpenAI client works against the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
with open("audio.flac", "rb") as file:
    print(client.audio.transcriptions.create(model="whisper-1", file=file).text)
```

The `model` field is ignored; the server always uses the model it loaded.

## Client

An interactive prompt that transcribes the audio files you enter:

```bash
uv run asrback-client --base-url http://127.0.0.1:8000/v1
audio> path/to/audio.flac
```

## Evaluation

Word error rate on LibriSpeech test-clean, against a running server:

```bash
scripts/download_librispeech_test_clean.sh   # -> data/librispeech
uv run asrback-eval --base-url http://127.0.0.1:8000/v1
```

Use `--limit N` for a quick run and `--output predictions.jsonl` to save per-utterance
results; add `--resume` to skip utterances already in that file. Text is normalized
with Whisper's `EnglishTextNormalizer` (`pip install asrback[eval]`).

For speed, keep `--concurrency` (default 32) above the server's `--max-batch-size` and
pass `--sort-by-length` so similar-length files share a batch. `--base-url` can be
repeated to spread requests over several servers.

```bash
uv run asrback-eval --base-url http://127.0.0.1:8000/v1 --sort-by-length
```

Greedy decoding, server defaults, RTX 5060 Ti; time is the whole test-clean run
(5.4 hours of audio) with `--sort-by-length`:

| Model | WER float16 | WER float32 | Time float16 | Time float32 |
| --- | --- | --- | --- | --- |
| whisper-tiny | 7.53% | 7.54% | 12.5 s | 24.7 s |
| whisper-tiny.en | 5.56% | 5.62% | 12.0 s | 25.0 s |
| whisper-base | 5.00% | 4.99% | 19.4 s | 45.6 s |
| whisper-base.en | 4.25% | 4.23% | 18.9 s | 46.8 s |
| whisper-small | 3.39% | 3.39% | 52.7 s | 144.3 s |
| whisper-small.en | 3.02% | 3.03% | 52.3 s | 134.7 s |

`--compile-encoder` brings whisper-small to 49.3 s (small.en 50.2 s) at the cost of
about 20 s more startup; it does not help tiny or base.

## Limitations

- Greedy or temperature sampling only; no beam search.
- No word-level timestamps; segments are fixed 30-second windows.
- One model per server process.
- English-only (`*.en`) models support transcription only, not translation.

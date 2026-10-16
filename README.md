# asrback

OpenAI-compatible speech-to-text server for Whisper checkpoints in Hugging Face
SafeTensors format, implemented in plain PyTorch (no `transformers` or `openai-whisper`
at runtime).

## Setup

```bash
uv sync
scripts/download_whisper_hf.sh tiny   # -> models/whisper-tiny (tiny, base, small)
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
results. Text is normalized with Whisper's `EnglishTextNormalizer`
(`pip install asrback[eval]`).

| Model | Decoding | WER (test-clean) |
| --- | --- | --- |
| whisper-tiny | greedy | 7.54% |

## Limitations

- Greedy or temperature sampling only; no beam search.
- No word-level timestamps; segments are fixed 30-second windows.
- One model per server process.

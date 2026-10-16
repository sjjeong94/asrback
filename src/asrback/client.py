"""``asrback-client`` command: transcribe files interactively against a server."""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

EXIT_COMMANDS = {"exit", "quit", "q"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="asrback-client",
        description="Interactively transcribe audio files with an asrback server.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ASRBACK_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="Server API base URL. Default: $ASRBACK_BASE_URL or "
        "http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--model", default="whisper-1", help="Default: whisper-1")
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Default: transcribe",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code such as en or ko (transcribe only). Default: detect",
    )
    parser.add_argument("--prompt", default=None, help="Optional decoding prompt.")
    parser.add_argument(
        "--response-format",
        choices=("text", "json", "verbose_json", "srt", "vtt"),
        default="text",
        help="Default: text",
    )
    return parser.parse_args(argv)


def enable_path_completion() -> None:
    try:
        import readline
    except ImportError:  # e.g. Windows without pyreadline
        return

    def complete(text: str, state: int) -> str | None:
        matches = [
            match + ("/" if os.path.isdir(match) else "")
            for match in sorted(glob.glob(os.path.expanduser(text) + "*"))
        ]
        return matches[state] if state < len(matches) else None

    readline.set_completer_delims(" \t\n")
    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")


def parse_path(line: str) -> Path:
    # Accept paths pasted or dragged in with surrounding quotes.
    if len(line) >= 2 and line[0] == line[-1] and line[0] in "'\"":
        line = line[1:-1]
    return Path(line).expanduser()


def request(client, args: argparse.Namespace, path: Path) -> str:
    options = {"model": args.model, "response_format": args.response_format}
    if args.prompt:
        options["prompt"] = args.prompt
    with path.open("rb") as file:
        if args.task == "translate":
            result = client.audio.translations.create(file=file, **options)
        else:
            if args.language:
                options["language"] = args.language
            result = client.audio.transcriptions.create(file=file, **options)

    if isinstance(result, str):
        return result.rstrip("\n")
    if args.response_format == "json":
        return result.text
    return result.model_dump_json(indent=2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import openai

    client = openai.OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "asrback"),
        max_retries=0,
    )
    enable_path_completion()
    print(f"Connected to {args.base_url} ({args.task}).")
    print("Enter an audio file path to transcribe; 'exit' or Ctrl-D to quit.")

    while True:
        try:
            line = input("audio> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in EXIT_COMMANDS:
            return 0

        path = parse_path(line)
        if not path.is_file():
            print(f"File not found: {path}")
            continue

        started = time.perf_counter()
        try:
            output = request(client, args, path)
        except openai.APIStatusError as exc:
            body = exc.body if isinstance(exc.body, dict) else {}
            print(f"Error {exc.status_code}: {body.get('message', exc.message)}")
            continue
        except openai.APIConnectionError:
            print(f"Cannot connect to {args.base_url}. Is the server running?")
            continue
        except KeyboardInterrupt:
            print("\nCancelled.")
            continue
        print(output)
        print(f"({time.perf_counter() - started:.2f}s)")

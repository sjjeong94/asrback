#!/usr/bin/env bash
# Download the FLEURS test split from Hugging Face (google/fleurs).
#
# Usage: download_fleurs_test.sh [LANGUAGE ...]
#   LANGUAGE: en_us, ko_kr, ja_jp (default: all three)
#
# Each language ends up in $DATA_DIR/<language>/ with test.tsv (transcripts) and
# test/*.wav (16 kHz audio).
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/fleurs}"
REVISION="${REVISION:-main}"
BASE_URL="https://huggingface.co/datasets/google/fleurs/resolve/$REVISION/data"
LANGUAGES=()

for arg in "$@"; do
  case "$arg" in
    en_us|ko_kr|ja_jp) LANGUAGES+=("$arg") ;;
    *)
      echo "Error: unsupported argument '$arg' (expected en_us, ko_kr, or ja_jp)." >&2
      exit 1
      ;;
  esac
done
if (( ${#LANGUAGES[@]} == 0 )); then
  LANGUAGES=(en_us ko_kr ja_jp)
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: install wget to download the dataset." >&2
  exit 1
fi

for language in "${LANGUAGES[@]}"; do
  language_dir="$DATA_DIR/$language"
  archive="$language_dir/test.tar.gz"
  extracted_dir="$language_dir/test"
  mkdir -p "$language_dir"

  echo "Downloading FLEURS $language test transcripts..."
  wget -c --tries=3 "$BASE_URL/$language/test.tsv" -O "$language_dir/test.tsv"

  if [[ -d "$extracted_dir" ]]; then
    echo "Audio already extracted: $extracted_dir"
  else
    if [[ -f "$archive" ]]; then
      echo "Archive already exists: $archive"
    else
      echo "Downloading FLEURS $language test audio..."
      wget -c --tries=3 "$BASE_URL/$language/audio/test.tar.gz" -O "$archive"
    fi

    echo "Verifying archive..."
    tar -tzf "$archive" >/dev/null

    echo "Extracting to $language_dir..."
    tar -xzf "$archive" -C "$language_dir"
  fi

  wav_count="$(find "$extracted_dir" -type f -name '*.wav' | wc -l)"
  utterance_count="$(wc -l < "$language_dir/test.tsv")"
  echo "Ready: $language_dir"
  echo "WAV files: $wav_count"
  echo "Transcript lines: $utterance_count"
done

echo "All requested FLEURS test sets are ready in $DATA_DIR."

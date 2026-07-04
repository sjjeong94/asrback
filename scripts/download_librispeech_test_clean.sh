#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-datasets/librispeech}"
URL="${URL:-https://www.openslr.org/resources/12/test-clean.tar.gz}"
ARCHIVE="$DATA_DIR/test-clean.tar.gz"
EXTRACTED_DIR="$DATA_DIR/LibriSpeech/test-clean"

mkdir -p "$DATA_DIR"

if [[ -d "$EXTRACTED_DIR" ]]; then
  echo "Dataset already extracted: $EXTRACTED_DIR"
else
  if [[ -f "$ARCHIVE" ]]; then
    echo "Archive already exists: $ARCHIVE"
  else
    echo "Downloading LibriSpeech test-clean..."
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail --continue-at - "$URL" -o "$ARCHIVE"
    elif command -v wget >/dev/null 2>&1; then
      wget -c "$URL" -O "$ARCHIVE"
    else
      echo "Error: install curl or wget to download the dataset." >&2
      exit 1
    fi
  fi

  echo "Verifying archive..."
  tar -tzf "$ARCHIVE" >/dev/null

  echo "Extracting to $DATA_DIR..."
  tar -xzf "$ARCHIVE" -C "$DATA_DIR"
fi

flac_count="$(find "$EXTRACTED_DIR" -type f -name '*.flac' | wc -l)"
transcript_count="$(find "$EXTRACTED_DIR" -type f -name '*.trans.txt' | wc -l)"

echo "Ready: $EXTRACTED_DIR"
echo "FLAC files: $flac_count"
echo "Transcript files: $transcript_count"

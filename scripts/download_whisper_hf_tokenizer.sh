#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/huggingface}"
REVISION="${REVISION:-main}"
FILES=(config.json tokenizer.json tokenizer_config.json)

if (( $# > 0 )); then
  MODELS=("$@")
else
  MODELS=(tiny base small)
fi

download_file() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 "$url" -O "$output"
  else
    echo "Error: install curl or wget to download model assets." >&2
    return 1
  fi
}

for model in "${MODELS[@]}"; do
  case "$model" in
    tiny|base|small) ;;
    *)
      echo "Error: unsupported model '$model' (expected tiny, base, or small)." >&2
      exit 1
      ;;
  esac

  repo="openai/whisper-$model"
  output_dir="$MODEL_DIR/whisper-$model"
  mkdir -p "$output_dir"

  echo "Downloading tokenizer assets for $repo@$REVISION..."
  for file in "${FILES[@]}"; do
    download_file \
      "https://huggingface.co/$repo/resolve/$REVISION/$file" \
      "$output_dir/$file"
  done
  echo "Ready: $output_dir"
done

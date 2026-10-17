#!/usr/bin/env bash
# Download the SenseVoiceSmall files needed for inference from Hugging Face.
#
# Usage: download_sensevoice_hf.sh
#   Files go to $MODEL_DIR/sensevoice-small (default: models/sensevoice-small).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
REVISION="${REVISION:-main}"
REPO="FunAudioLLM/SenseVoiceSmall"
FILES=(
  config.yaml
  configuration.json
  am.mvn
  chn_jpn_yue_eng_ko_spectok.bpe.model
  model.pt
)

if (( $# > 0 )); then
  echo "Error: unexpected argument '$1' (this script takes none)." >&2
  exit 1
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: install wget to download models." >&2
  exit 1
fi

output_dir="$MODEL_DIR/sensevoice-small"
mkdir -p "$output_dir"

echo "Downloading $REPO@$REVISION to $output_dir..."
for file in "${FILES[@]}"; do
  wget -c --tries=3 \
    "https://huggingface.co/$REPO/resolve/$REVISION/$file" \
    -O "$output_dir/$file"
done

echo "Ready: $output_dir"

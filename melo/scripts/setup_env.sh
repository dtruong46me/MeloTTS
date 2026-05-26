#!/usr/bin/env bash
# One-time environment setup for MeloTTS.
#
# Run this script once after ``pip install -e .`` to download all required
# language-model assets:
#   1. Japanese tokeniser data (unidic)
#   2. All MeloTTS language model checkpoints (from Hugging Face Hub)
#
# Usage (from the project root):
#   bash melo/scripts/setup_env.sh

set -euo pipefail

echo "==> Downloading Japanese tokeniser data (unidic)..."
python -m unidic download

echo "==> Pre-downloading all MeloTTS language models..."
python melo/scripts/init_downloads.py

echo "Done. Environment is ready."

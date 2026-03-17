#!/bin/bash
# ZoonoticSense — Mac M4 Setup Script
# Run: bash setup.sh

set -e
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     ZoonoticSense — Setup (Mac M4)       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Check Python ────────────────────────────────────────────────
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# ── Check ffmpeg ─────────────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
    echo "[!] ffmpeg not found. Installing via Homebrew..."
    brew install ffmpeg
fi
echo "[✓] ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# ── Install Python packages ──────────────────────────────────────
echo ""
echo "[*] Installing Python packages..."
pip3 install --break-system-packages \
    flask \
    mlx-whisper \
    mlx-lm \
    sentence-transformers \
    scikit-learn \
    numpy \
    kokoro \
    soundfile

echo ""
echo "[✓] All packages installed"

# ── Create directory structure ───────────────────────────────────
echo ""
echo "[*] Creating knowledge base directories..."
mkdir -p knowledge_base/{avian_flu,rabies,fmd,nipah_hendra,leptospirosis,general}/raw
mkdir -p models data

echo "[✓] Directory structure ready"

# ── Download models (first run) ──────────────────────────────────
echo ""
echo "[*] Pre-downloading models (this may take a few minutes)..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading all-MiniLM-L6-v2...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('  Done.')
"

echo ""
echo "[*] Training router (using built-in seed data)..."
python3 models/router.py --train --out-dir models
echo "[✓] Router trained"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║              Setup Complete!             ║"
echo "║                                          ║"
echo "║  Run:  python3 app.py                    ║"
echo "║  Open: http://localhost:7860             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "NOTE: First run will download Whisper (~150MB)"
echo "      and Qwen3-4B (~2.5GB) automatically."
echo ""

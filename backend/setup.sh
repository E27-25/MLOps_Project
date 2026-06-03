#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  ZoonoMoE — One-shot setup + launch (Lanta / SLURM GPU node)
#  Usage:  bash setup.sh
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 0. Environment ───────────────────────────────────────────────────
MODULE_CUDA="${MODULE_CUDA:-cuda/12.1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/venv}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
PORT="${PORT:-7860}"
export HF_HOME

echo "▶ ZoonoMoE setup starting in: $PROJECT_ROOT"

# ── 1. System modules (Lanta) ────────────────────────────────────────
if command -v module &>/dev/null; then
    module purge || true
    module load "$MODULE_CUDA" || echo "  (cuda module not found, continuing)"
fi

# ── 2. Python venv ───────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "▶ Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 3. Python deps ───────────────────────────────────────────────────
pip install --quiet --upgrade pip
if [[ -f requirements.txt ]]; then
    pip install --quiet -r requirements.txt
fi

# ── 4. Launch ────────────────────────────────────────────────────────
echo "▶ Launching ZoonoMoE backend on port $PORT"
cd "$PROJECT_ROOT"
python3 app.py

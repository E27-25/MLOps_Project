#!/bin/bash
#SBATCH -p gpu                       # GPU partition
#SBATCH --gpus-per-node=1            # 1 GPU
#SBATCH -N 1 -c 16                   # 1 node, 16 CPUs
#SBATCH -t 1:00:00                   # 1 hour
#SBATCH -A zz991016                  # project / account
#SBATCH -J zoono-smoke               # job name
#SBATCH -o zoono-smoke-%j.out        # stdout+stderr log
# ════════════════════════════════════════════════════════════════════════════
#  ZoonoMoE — end-to-end pipeline smoke test on a GPU node.
#  Submit from the backend/ directory:
#       sbatch zoono_smoke.sbatch              # English TTS (Kokoro)
#       TTS_LANG=th sbatch zoono_smoke.sbatch  # Thai TTS (JaiTTS/F5)
#  Env (conda `kj`) + all models must be pre-cached on the LOGIN node first
#  (compute nodes have no internet) — see prep_smoke_env.sh.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

module reset
# venv built by prep_smoke_env.sh (system python3.12 + CUDA torch wheel)
source /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/zoono-smoke-env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

# ── runtime config ───────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1                 # no internet on compute node → cache only
export TRANSFORMERS_OFFLINE=1
export USE_MLX=false
export USE_TRITON=false
export LLM_MODEL="${LLM_MODEL:-/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/Qwen3-14B}"  # local full model dir (offline)
# 14B fp16 (~29.6GB) is tight on a 40GB A100 — eager mode skips CUDA-graph
# capture (saves VRAM) and a touch more memory headroom avoids OOM.
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WHISPER_SIZE="${WHISPER_SIZE:-base}"
export TTS_LANG="${TTS_LANG:-en}"
export JAITTS_REF_VOICE="${JAITTS_REF_VOICE:-$SLURM_SUBMIT_DIR/voices/th_ref.wav}"
# flowtts (JaiTTS) lives as a namespace package in the thonburian-tts checkout
export THONBURIAN_TTS_DIR="${THONBURIAN_TTS_DIR:-/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/thonburian-tts}"
# provide ref transcript so F5 skips auto-transcribe (an uncached ASR fetch)
export JAITTS_REF_TEXT="${JAITTS_REF_TEXT:-$(cat "$SLURM_SUBMIT_DIR/voices/th_ref.txt" 2>/dev/null)}"

echo "▶ node=$(hostname)  TTS_LANG=$TTS_LANG  LLM=$LLM_MODEL"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ── run the smoke test (exit code = number of failed stages) ─────────────────
python scripts/smoke_test.py --wav voices/th_ref.wav --tts-lang "$TTS_LANG"
RC=$?
echo "▶ smoke_test exit code (failed stages): $RC"
exit $RC

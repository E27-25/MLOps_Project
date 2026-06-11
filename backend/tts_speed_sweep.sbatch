#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 8
#SBATCH -t 0:20:00
#SBATCH -A zz991016
#SBATCH -J zoono-speed
#SBATCH -o zoono-speed-%j.out
set -uo pipefail
module reset
source /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/zoono-smoke-env/bin/activate
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAITTS_REF_VOICE=$SLURM_SUBMIT_DIR/voices/th_ref.wav
export JAITTS_REF_TEXT="$(cat "$SLURM_SUBMIT_DIR/voices/th_ref.txt" 2>/dev/null)"
export THONBURIAN_TTS_DIR=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/thonburian-tts
python scripts/tts_speed_sweep.py
echo DONE

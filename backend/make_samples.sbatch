#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 1:00:00
#SBATCH -A zz991016
#SBATCH -J zoono-samples
#SBATCH -o zoono-samples-%j.out
# Generate input/output demo samples (EN Kokoro + TH JaiTTS) in one job.
set -uo pipefail
module reset
source /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/zoono-smoke-env/bin/activate
cd "${SLURM_SUBMIT_DIR}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_MLX=false USE_TRITON=false
export LLM_MODEL=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/Qwen3-14B
export WHISPER_SIZE=base
export VLLM_ENFORCE_EAGER=true VLLM_GPU_MEM_UTIL=0.92
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export JAITTS_REF_VOICE=$SLURM_SUBMIT_DIR/voices/th_ref.wav
export THONBURIAN_TTS_DIR=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/thonburian-tts
export JAITTS_REF_TEXT="$(cat "$SLURM_SUBMIT_DIR/voices/th_ref.txt" 2>/dev/null)"

echo "================ EN (Kokoro) ================"
TTS_LANG=en python scripts/make_samples.py --wav samples/input_report_en.wav --tts-lang en
echo "================ TH (JaiTTS) ================"
TTS_LANG=th python scripts/make_samples.py --wav samples/input_report_en.wav --tts-lang th
echo "DONE"

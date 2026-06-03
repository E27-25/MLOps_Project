#!/usr/bin/env bash
#SBATCH --job-name=zoono-bench
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=zoono-bench-%j.out
#SBATCH --error=zoono-bench-%j.out
# ════════════════════════════════════════════════════════════════════
#  ZoonoMoE — end-to-end pipeline latency benchmark on a GPU node.
#
#  Submit from the backend/ directory:
#     sbatch -A <YOUR_ACCOUNT> run_benchmark.sbatch
#  (add  --qos=<qos>  if your project needs it.)
#
#  What it does, all inside one GPU allocation:
#    1. set up env + venv (via setup.sh deps), load CUDA
#    2. (re)train the corrected MLP router + the confidence-gated cascade
#    3. start app.py in the background, wait until /health is ready
#    4. run scripts/benchmark_pipeline.py  (5 runs x 6 inputs)
#    5. write eval_out/pipeline_{results.json,table.md}, then shut down
# ════════════════════════════════════════════════════════════════════
set -uo pipefail

MODULE_CUDA="${MODULE_CUDA:-cuda/12.1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/venv}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
PORT="${PORT:-7860}"
RUNS="${RUNS:-5}"
export HF_HOME
cd "$PROJECT_ROOT"

echo "▶ Node: $(hostname)   GPU(s):"; nvidia-smi -L || true

# ── 1. modules + venv ────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module purge || true
    module load "$MODULE_CUDA" || echo "  (cuda module not found, continuing)"
fi
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
[[ -f requirements.txt ]] && pip install --quiet -r requirements.txt

# ── 2. (re)train router + cascade so the deployed model is current ────
echo "▶ Training corrected router + cascade ..."
python3 models/router.py --train         --extra data/router_training.jsonl --out-dir models
python3 models/router.py --train-cascade --extra data/router_training.jsonl --out-dir models --tau 0.54

# ── 3. start server, wait for health ─────────────────────────────────
echo "▶ Starting backend on :$PORT ..."
USE_TRITON="${USE_TRITON:-false}" PORT="$PORT" python3 app.py > server-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "▶ Waiting for /health (model load can take several minutes) ..."
for i in $(seq 1 120); do                 # up to ~20 min (120 x 10s)
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "  ✓ server healthy after ~$((i*10))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  ✗ server process died — see server-$SLURM_JOB_ID.log"; exit 1
    fi
    sleep 10
done

# ── 4. run the end-to-end latency benchmark ──────────────────────────
echo "▶ Running pipeline benchmark ($RUNS runs x 6 inputs) ..."
python3 scripts/benchmark_pipeline.py --base-url "http://localhost:$PORT" --runs "$RUNS"
RC=$?

# ── 5. show results ──────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  RESULTS  (also in scripts/eval_out/pipeline_table.md)"
echo "════════════════════════════════════════════════════════════════"
cat scripts/eval_out/pipeline_table.md 2>/dev/null || echo "(no table produced)"

exit $RC

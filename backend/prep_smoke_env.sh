#!/usr/bin/env bash
# Login-node prep for the ZoonoMoE smoke test.
# Builds a plain venv (system python3.12) with CUDA torch + the full backend
# stack, then warms every model into the local cache (compute nodes are
# offline, so everything must be present before sbatch). Avoids conda — the
# cluster's central conda pkgs cache is read-only and breaks env links.
set -uo pipefail

ENV_DIR=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/zoono-smoke-env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
BASE_PY=/usr/bin/python3.12          # stable OS path, present on compute nodes
unset PYTHONPATH

echo "▶ [1/4] venv at $ENV_DIR ($($BASE_PY --version)) ..."
for i in 1 2 3 4 5; do rm -rf "$ENV_DIR" 2>/dev/null; [[ -d "$ENV_DIR" ]] || break; done
[[ -d "$ENV_DIR" ]] && { echo "‼ could not clear $ENV_DIR"; exit 1; }
"$BASE_PY" -m venv "$ENV_DIR"
PIP="$ENV_DIR/bin/pip"; VPY="$ENV_DIR/bin/python"
[[ -x "$PIP" ]] || { echo "‼ venv creation FAILED"; exit 1; }
"$PIP" install -q -U pip wheel 2>&1 | tail -1

echo "▶ [2/4] CUDA torch (cu124) ..."
"$PIP" install -q --default-timeout=1000 torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

echo "▶ [3/4] backend deps ..."
"$PIP" install -q --default-timeout=1000 \
    fastapi "uvicorn[standard]" python-multipart jinja2 \
    "transformers>=4.44" "accelerate>=0.27.0" \
    openai-whisper kokoro "sentence-transformers>=3.0.0" \
    faiss-cpu soundfile imageio-ffmpeg scikit-learn scipy sqlalchemy 2>&1 | tail -4

echo "▶ verifying imports ..."
"$VPY" - <<'PY'
import importlib
for m in ["torch","transformers","fastapi","whisper","kokoro","sentence_transformers","faiss","soundfile","sklearn"]:
    try:
        mod=importlib.import_module(m); print("  OK  %-22s %s"%(m,getattr(mod,'__version__','?')))
    except Exception as e: print("  FAIL %-20s %s: %s"%(m,type(e).__name__,e))
import torch; print("  torch cuda build tag:", torch.version.cuda)
PY

echo "▶ [4/4] warming models into cache (login node has internet) ..."
"$VPY" - <<'PY'
def warm(name, fn):
    try: fn(); print("  OK  ", name)
    except Exception as e: print("  FAIL", name, ":", repr(e)[:160])
warm("whisper base", lambda: __import__("whisper").load_model("base", device="cpu"))
warm("all-MiniLM-L6-v2", lambda: __import__("sentence_transformers").SentenceTransformer("all-MiniLM-L6-v2"))
def _path():
    from transformers import pipeline; pipeline("automatic-speech-recognition", model="nectec/Pathumma-whisper-th-medium")
warm("Pathumma-whisper-th-medium", _path)
warm("kokoro", lambda: __import__("kokoro").KPipeline(lang_code="a", device="cpu"))
PY

echo "▶ confirm Qwen3-14B in cache:"; ls "$HF_HOME/hub" | grep -i "Qwen3-14B" || echo "  (missing!)"
echo "▶ PREP DONE."

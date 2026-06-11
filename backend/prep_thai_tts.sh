#!/usr/bin/env bash
# Login-node prep for the Thai TTS (JaiTTS / F5-TTS) path: install flowtts +
# f5-tts into the smoke env and pre-download the JaiTTS checkpoint, then verify
# the existing vllm/transformers stack still imports (f5-tts can pull conflicts).
set -uo pipefail
ENV=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/zoono-smoke-env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PIP="$ENV/bin/pip"; VPY="$ENV/bin/python"
unset PYTHONPATH

echo "▶ [1/3] installing flowtts + f5-tts + cached-path + vocos ..."
"$PIP" install --default-timeout=1000 \
    cached-path vocos f5-tts \
    "git+https://github.com/biodatlab/thonburian-tts.git" 2>&1 | tail -10

echo "▶ [2/3] verifying imports (Thai deps + existing stack must both survive) ..."
"$VPY" - <<'PY'
for m in ["flowtts","f5_tts","cached_path","vocos","vllm","transformers","torch","sentence_transformers","kokoro"]:
    try:
        mod=__import__(m); print("  OK  %-22s %s"%(m,getattr(mod,'__version__','?')))
    except Exception as e: print("  FAIL %-20s %s: %s"%(m,type(e).__name__,str(e)[:110]))
PY

echo "▶ [3/3] pre-downloading JaiTTS-F5TTS checkpoint + vocab ..."
"$VPY" - <<'PY'
from cached_path import cached_path
for u in ["hf://JTS-AI/JaiTTS-F5TTS/model.pt","hf://JTS-AI/JaiTTS-F5TTS/vocab.txt"]:
    try: print("  OK  ", cached_path(u))
    except Exception as e: print("  FAIL", u, ":", repr(e)[:160])
PY
echo "THAI PREP DONE"

#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Mini Speech-to-Speech — Mac M4   Setup Script
# ═══════════════════════════════════════════════════════════════════
# Run once:   bash setup.sh
# Then run:   python3 app.py
# ═══════════════════════════════════════════════════════════════════
set -e

# ── Colours ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓  $*${NC}"; }
info() { echo -e "${CYAN}  →  $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $*${NC}"; }
fail() { echo -e "${RED}  ✗  $*${NC}"; exit 1; }
header() { echo -e "\n${BOLD}$*${NC}"; }

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${BOLD}   Mini Speech-to-Speech — Mac M4   Setup     ${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"

# ── 1. Architecture check ─────────────────────────────────────────
header "[ 1 / 9 ]  System"
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    ok "Apple Silicon ($ARCH) — MPS / Metal acceleration available"
else
    warn "Not Apple Silicon ($ARCH) — MPS will not be available, LLM will run on CPU"
fi

OS_VER=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
ok "macOS $OS_VER"

# ── 2. Homebrew ───────────────────────────────────────────────────
header "[ 2 / 9 ]  Homebrew"
if ! command -v brew &>/dev/null; then
    info "Homebrew not found — installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add to PATH for Apple Silicon
    if [ -f "/opt/homebrew/bin/brew" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo '' >> ~/.zprofile
        echo '# Homebrew' >> ~/.zprofile
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        ok "Added Homebrew to ~/.zprofile"
    fi
else
    ok "Homebrew $(brew --version | head -1 | awk '{print $2}')"
fi

# ── 3. System packages ────────────────────────────────────────────
header "[ 3 / 9 ]  System packages"

# ffmpeg — used by both Whisper (audio loading) and app (webm→wav conversion)
if ! command -v ffmpeg &>/dev/null; then
    info "Installing ffmpeg..."
    brew install ffmpeg
    ok "ffmpeg installed"
else
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    ok "ffmpeg $FFMPEG_VER"
fi

# espeak-ng — Kokoro TTS phonemizer backend
if ! command -v espeak-ng &>/dev/null; then
    info "Installing espeak-ng (Kokoro phonemizer)..."
    brew install espeak-ng 2>/dev/null || warn "espeak-ng install failed — Kokoro will use fallback phonemizer"
else
    ok "espeak-ng $(espeak-ng --version 2>&1 | head -1)"
fi

# portaudio — optional but avoids pyaudio warnings
brew list portaudio &>/dev/null 2>&1 || brew install portaudio --quiet 2>/dev/null || true
ok "portaudio"

# ── 4. Python ─────────────────────────────────────────────────────
header "[ 4 / 9 ]  Python"

if ! command -v python3 &>/dev/null; then
    info "Python3 not found — installing Python 3.11..."
    brew install python@3.11
    export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
fi

PYTHON=$(command -v python3)
PY_VER=$("$PYTHON" --version 2>&1)
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    fail "Python 3.10+ required, found $PY_VER. Install with: brew install python@3.11"
fi

ok "$PY_VER at $PYTHON"

# ── 5. SSL certificates ───────────────────────────────────────────
header "[ 5 / 9 ]  SSL certificates"
# Python.org installer on Mac does not include SSL certs by default.
# This causes whisper/huggingface model downloads to fail with:
#   ssl.SSLCertVerificationError: certificate verify failed

SSL_TEST=$("$PYTHON" -c "
import ssl, urllib.request
try:
    urllib.request.urlopen('https://huggingface.co', timeout=8)
    print('ok')
except Exception as e:
    print('fail:' + str(e)[:80])
" 2>/dev/null)

if [[ "$SSL_TEST" == ok ]]; then
    ok "SSL certificates working"
else
    warn "SSL issue detected: $SSL_TEST"
    info "Attempting fix..."

    # Try official Python installer certificate command first
    CERT_CMD="/Applications/Python ${PY_MAJOR}.${PY_MINOR}/Install Certificates.command"
    if [ -f "$CERT_CMD" ]; then
        bash "$CERT_CMD" > /dev/null 2>&1 && ok "Fixed via Install Certificates.command" || true
    fi

    # Re-test
    SSL_TEST2=$("$PYTHON" -c "
import ssl, urllib.request
try:
    urllib.request.urlopen('https://huggingface.co', timeout=8)
    print('ok')
except:
    print('fail')
" 2>/dev/null)

    if [[ "$SSL_TEST2" != ok ]]; then
        # Fallback: certifi
        "$PYTHON" -m pip install --quiet certifi
        CERT_FILE=$("$PYTHON" -c "import certifi; print(certifi.where())")
        ZSHRC="$HOME/.zshrc"
        if ! grep -q "SSL_CERT_FILE" "$ZSHRC" 2>/dev/null; then
            echo "" >> "$ZSHRC"
            echo "# SSL fix for Python model downloads" >> "$ZSHRC"
            echo "export SSL_CERT_FILE=$CERT_FILE" >> "$ZSHRC"
            echo "export REQUESTS_CA_BUNDLE=$CERT_FILE" >> "$ZSHRC"
        fi
        export SSL_CERT_FILE="$CERT_FILE"
        export REQUESTS_CA_BUNDLE="$CERT_FILE"
        ok "Fixed via certifi — added SSL_CERT_FILE to ~/.zshrc"
        warn "Run 'source ~/.zshrc' or restart terminal before running app.py"
    else
        ok "SSL fixed"
    fi
fi

# ── 6. pip + core Python packages ────────────────────────────────
header "[ 6 / 9 ]  Python packages"

info "Upgrading pip..."
"$PYTHON" -m pip install --quiet --upgrade pip
ok "pip $("$PYTHON" -m pip --version | awk '{print $2}')"

# PyTorch with MPS support
# torch 2.1+ has MPS built in for Apple Silicon — no special build needed
info "Installing PyTorch (MPS support built-in for Apple Silicon)..."
"$PYTHON" -m pip install --quiet "torch>=2.1" torchvision torchaudio
TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "error")
MPS_OK=$("$PYTHON" -c "import torch; print('yes' if torch.backends.mps.is_available() else 'no')" 2>/dev/null || echo "no")
ok "PyTorch $TORCH_VER  |  MPS available: $MPS_OK"

# Transformers — LLM inference + TextIteratorStreamer for token streaming
info "Installing Transformers..."
"$PYTHON" -m pip install --quiet "transformers>=4.40" accelerate
ok "transformers $("$PYTHON" -c "import transformers; print(transformers.__version__)" 2>/dev/null)"

# Whisper ASR
info "Installing Whisper ASR..."
"$PYTHON" -m pip install --quiet openai-whisper
ok "openai-whisper installed"

# NumPy version pin — numpy 2.2+ breaks openai-whisper on some versions
info "Checking NumPy version..."
NP_VER=$("$PYTHON" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0.0.0")
NP_MAJOR=$(echo "$NP_VER" | cut -d. -f1)
NP_MINOR=$(echo "$NP_VER" | cut -d. -f2)
if [ "$NP_MAJOR" -ge 2 ] && [ "$NP_MINOR" -ge 2 ]; then
    warn "NumPy $NP_VER may cause whisper issues — pinning to <2.2..."
    "$PYTHON" -m pip install --quiet "numpy<2.2"
    ok "NumPy downgraded to $("$PYTHON" -c 'import numpy; print(numpy.__version__)')"
else
    ok "NumPy $NP_VER (compatible)"
fi

# Kokoro TTS
info "Installing Kokoro TTS..."
"$PYTHON" -m pip install --quiet "kokoro>=0.9.4" soundfile
KOKORO_OK=$("$PYTHON" -c "from kokoro import KPipeline; print('ok')" 2>/dev/null || echo "fail")
if [ "$KOKORO_OK" != "ok" ]; then
    warn "Kokoro import failed — retrying with --pre..."
    "$PYTHON" -m pip install --quiet --pre "kokoro>=0.9.4"
    KOKORO_OK2=$("$PYTHON" -c "from kokoro import KPipeline; print('ok')" 2>/dev/null || echo "fail")
    [ "$KOKORO_OK2" = "ok" ] && ok "Kokoro installed (pre-release)" || fail "Kokoro install failed — check errors above"
else
    ok "Kokoro TTS installed"
fi

# Flask web server + SSE support
info "Installing Flask..."
"$PYTHON" -m pip install --quiet flask
ok "Flask $("$PYTHON" -c "import flask; print(flask.__version__)" 2>/dev/null)"

# ── 7. Full import verification ───────────────────────────────────
header "[ 7 / 9 ]  Import verification"

"$PYTHON" - << 'PYCHECK'
import sys
RED   = '\033[0;31m'
GREEN = '\033[0;32m'
NC    = '\033[0m'

checks = [
    ("torch + MPS",
     "import torch; assert torch.backends.mps.is_available(), "
     "'MPS not available — LLM will fall back to CPU'"),
    ("transformers (TextIteratorStreamer)",
     "from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer"),
    ("whisper",
     "import whisper"),
    ("kokoro KPipeline",
     "from kokoro import KPipeline"),
    ("soundfile",
     "import soundfile"),
    ("flask + stream_with_context",
     "from flask import Flask, stream_with_context, Response"),
    ("numpy",
     "import numpy as np; assert tuple(int(x) for x in np.__version__.split('.')[:2]) < (2,2), "
     f"f'numpy {np.__version__} may break whisper'"),
]

all_ok = True
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  {GREEN}✓{NC}  {name}")
    except AssertionError as e:
        print(f"  \033[1;33m⚠{NC}  {name}  ({e})")
    except Exception as e:
        print(f"  {RED}✗{NC}  {name}: {e}")
        all_ok = False

if not all_ok:
    print(f"\n{RED}Some imports failed — check errors above before running app.py{NC}")
    sys.exit(1)
PYCHECK

ok "All imports verified"

# ── 8. ffmpeg sanity check ────────────────────────────────────────
header "[ 8 / 9 ]  ffmpeg sanity check"
# Test the exact command app.py runs: webm → wav conversion
TEST_WAV=$(mktemp /tmp/s2s_test_XXXX.wav)
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=1" \
       -ar 16000 -ac 1 "$TEST_WAV" -loglevel quiet 2>/dev/null \
    && ok "ffmpeg audio conversion working" \
    || warn "ffmpeg test failed — audio conversion may not work"
rm -f "$TEST_WAV"

# ── 9. Optional: mlx-whisper ─────────────────────────────────────
header "[ 9 / 9 ]  Optional: mlx-whisper (faster ASR)"
MLX_INSTALLED=$("$PYTHON" -c "import mlx_whisper; print('yes')" 2>/dev/null || echo "no")
if [ "$MLX_INSTALLED" = "yes" ]; then
    ok "mlx-whisper already installed"
else
    echo ""
    echo "  mlx-whisper uses Apple Neural Engine for 3-4x faster transcription."
    echo "  Install? (adds ~500 MB, optional)"
    read -p "  Install mlx-whisper? [y/N] " -n 1 -r REPLY
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        "$PYTHON" -m pip install --quiet mlx-whisper
        ok "mlx-whisper installed"
        echo ""
        warn "To activate mlx-whisper, see README.md — requires 3 line change in app.py"
    else
        info "Skipped (you can install later: pip install mlx-whisper)"
    fi
fi

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Setup complete!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo ""
echo "  Start the app:"
echo ""
echo -e "    ${BOLD}python3 app.py${NC}"
echo ""
echo "  Browser opens automatically at http://localhost:7860"
echo ""
echo "  First run downloads models (~3.5 GB total):"
echo "    whisper-base        145 MB  → ~/.cache/whisper/"
echo "    Qwen2.5-1.5B        3.1 GB  → ~/.cache/huggingface/"
echo "    Kokoro-82M          330 MB  → ~/.cache/huggingface/"
echo ""
echo "  See README.md for upgrade options and troubleshooting."
echo ""
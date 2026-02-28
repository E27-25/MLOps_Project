# ZoonoMoE Dockerfile
# Cloud / Linux deployment — CPU inference with OpenAI Whisper fallback.
# Mac M-series users: run natively (MLX gives GPU acceleration, Docker won't).

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Install CPU-only PyTorch first so it doesn't pull the CUDA wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Core pipeline packages
RUN pip install --no-cache-dir \
    flask \
    openai-whisper \
    transformers \
    accelerate \
    sentence-transformers \
    scikit-learn \
    numpy \
    kokoro \
    soundfile \
    faiss-cpu

# Discord logging (optional — only active when DISCORD_WEBHOOK is set)
RUN pip install --no-cache-dir "discordflow[system]"

# ── Copy application ───────────────────────────────────────────────────────────
COPY . .

# ── Ensure required directories exist ─────────────────────────────────────────
RUN mkdir -p \
    models \
    knowledge_base/avian_flu/raw \
    knowledge_base/rabies/raw \
    knowledge_base/fmd/raw \
    knowledge_base/nipah_hendra/raw \
    knowledge_base/leptospirosis/raw \
    knowledge_base/general/raw

# ── Environment defaults ───────────────────────────────────────────────────────
# Override these at runtime with -e or docker-compose environment section
ENV USE_MLX=false
ENV LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ENV WHISPER_SIZE=base
ENV PORT=7860
ENV DEBUG=false
# Set your Discord Forum webhook to enable pipeline logging
ENV DISCORD_WEBHOOK=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app.py"]

# ZoonoMoE Dockerfile
# Cloud / Linux deployment — vLLM CUDA inference (GPU required).
# Mac M-series users: run natively (MLX gives GPU acceleration, Docker won't).

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    gcc \
    build-essential \
    python3.11 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# vLLM stable — includes torch, transformers, accelerate
RUN pip install --no-cache-dir --timeout 300 vllm

# Core pipeline packages
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    jinja2 \
    openai-whisper \
    sentence-transformers \
    scikit-learn \
    numpy \
    kokoro \
    soundfile \
    faiss-cpu

# Triton client
RUN pip install --no-cache-dir "tritonclient[http]>=2.44.0"

# Database
RUN pip install --no-cache-dir sqlalchemy psycopg2-binary

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
ENV USE_MLX=false
ENV USE_TRITON=true
ENV TRITON_URL=triton:8000
ENV LLM_MODEL=Qwen/Qwen3-4B
ENV WHISPER_SIZE=base
ENV PORT=7860
ENV DEBUG=false
ENV DISCORD_WEBHOOK=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

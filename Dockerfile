# ZoonoticSense Dockerfile
# For cloud/Linux deployment (CPU inference with smaller model)
# Mac M-series users: run natively (MLX provides GPU acceleration)

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (CPU-only stack)
COPY requirements.txt .
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
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY . .

# Create model and knowledge base directories
RUN mkdir -p models knowledge_base/{avian_flu,rabies,fmd,nipah_hendra,leptospirosis,general}/raw

# Environment: use CPU-friendly smaller model for cloud
ENV USE_MLX=false
ENV LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ENV WHISPER_SIZE=base
ENV PORT=7860

EXPOSE 7860

# Pre-download models at build time (optional — comment out to speed up build)
# RUN python -c "import whisper; whisper.load_model('base')"

CMD ["python", "app.py"]

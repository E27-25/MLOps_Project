FROM nvcr.io/nvidia/tritonserver:26.02-py3

# Install Python packages needed by Python backends
RUN pip install --no-cache-dir \
    openai-whisper \
    sentence-transformers \
    transformers \
    "srsly>=2.4.8" \
    "spacy>=3.7.0" \
    kokoro \
    soundfile \
    numpy

# Bake model configs into image (no volume mount needed)
COPY triton_models /models

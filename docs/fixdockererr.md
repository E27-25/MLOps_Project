# Docker Error Fixes

สรุปปัญหาและวิธีแก้ที่เกิดขึ้นระหว่าง docker compose up --build

---

## 1. TYPE_BYTES ไม่มีใน Triton

**Error:**
```
Invalid argument: Unexpected data type TYPE_BYTES
```

**สาเหตุ:** config.pbtxt ใช้ `TYPE_BYTES` ซึ่ง Triton ไม่รู้จัก

**แก้:** เปลี่ยนเป็น `TYPE_STRING` ในทุก config.pbtxt

ไฟล์ที่แก้:
- `triton_models/whisper_asr/config.pbtxt`
- `triton_models/kokoro_tts/config.pbtxt`
- `triton_models/sentence_embedder/config.pbtxt`

---

## 2. whisper_asr TypeError: string indices must be integers

**Error:**
```
TypeError: string indices must be integers, not 'str'
```
at `triton_models/whisper_asr/1/model.py` line 13

**สาเหตุ:** parse parameters ผิด — ใช้ list comprehension แต่ Triton ส่ง dict มา

**แก้:** เปลี่ยน model.py

```python
# ผิด
params = {p['key']: p['value'] for p in params}

# ถูก
params = model_config.get('parameters', {})
size = params.get('whisper_size', {}).get('string_value', 'base')
```

---

## 3. kokoro_tts Unpack failed: incomplete input

**Error:**
```
ValueError: Unpack failed: incomplete input
```

**สาเหตุ:** srsly/msgpack version conflict ระหว่าง kokoro กับ spaCy ใน Triton container

**แก้:** เพิ่ม pin version ใน `triton.Dockerfile`

```dockerfile
RUN pip install --no-cache-dir \
    ...
    "srsly>=2.4.8" \
    "spacy>=3.7.0" \
    ...
```

---

## 4. DNS resolution failure

**Error:**
```
[Errno -3] Temporary failure in name resolution
```

**สาเหตุ:** Docker container ไม่สามารถ resolve hostname ภายนอกได้ (เช่น huggingface.co)

**แก้:** เพิ่ม DNS ใน `docker-compose.yml` ทั้ง triton และ backend

```yaml
dns:
  - 8.8.8.8
  - 1.1.1.1
```

---

## 5. โหลดโมเดลผิดตัว

**อาการ:** backend โหลด `Qwen/Qwen2.5-1.5B-Instruct` แทน `Qwen/Qwen3-4B`

**สาเหตุ:** `docker-compose.yml` กำหนด `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct` ทับค่าใน Dockerfile

**หลักการ:** environment ใน docker-compose.yml มี priority สูงกว่า ENV ใน Dockerfile เสมอ

**แก้:** แก้ `docker-compose.yml`

```yaml
environment:
  - LLM_MODEL=Qwen/Qwen3-4B
```

---

## 6. Python.h: No such file or directory

**Error:**
```
fatal error: Python.h: No such file or directory
compilation terminated.
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpXXXX/cuda_utils.c', ...
'-I/usr/include/python3.10']' returned non-zero exit status 1.
```

**สาเหตุ:** vLLM ต้อง JIT-compile CUDA utility code ด้วย gcc แต่ขาด Python dev headers
(`Python.h` มาจาก package `python3-dev`)

**แก้:** เพิ่ม `python3-dev` ใน `backend/Dockerfile`

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    gcc \
    build-essential \
    python3.11 \
    python3-pip \
    python3-dev \       # <-- เพิ่มตรงนี้
    && rm -rf /var/lib/apt/lists/*
```

---

## 7. โมเดลต้อง re-download ทุกครั้งที่ rebuild

**อาการ:** ทุกครั้งที่ `docker compose up --build` ต้องโหลดโมเดลใหม่ทั้งหมด

**สาเหตุ:** ไม่มี persistent volume สำหรับ model cache

**แก้:** เพิ่ม named volumes ใน `docker-compose.yml`

```yaml
services:
  triton:
    volumes:
      - huggingface_cache:/root/.cache/huggingface
      - whisper_cache:/root/.cache/whisper
  backend:
    volumes:
      - huggingface_cache:/root/.cache/huggingface
      - whisper_cache:/root/.cache/whisper

volumes:
  huggingface_cache:
  whisper_cache:
```

---

## 8. vLLM install command ผิด

**Error:**
```
ERROR: option --torch-backend not recognized
```

**สาเหตุ:** ใช้ flag ที่ pip ไม่รู้จัก

**แก้:** เปลี่ยน `backend/Dockerfile`

```dockerfile
# ผิด
RUN pip install -U vllm --torch-backend=auto

# ถูก
RUN pip install --no-cache-dir vllm
```

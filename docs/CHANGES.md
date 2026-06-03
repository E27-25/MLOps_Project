# ZoonoticSense — สรุปการเปลี่ยนแปลง (MLOps Upgrade)

---

## 1. LLM Model Selection

เปลี่ยน default model ตาม hardware จริง

| Platform | เก่า | ใหม่ |
|---|---|---|
| Mac M-series | `mlx-community/Qwen3-4B-4bit` | `mlx-community/Qwen3-4B-4bit` — เดิม |
| Windows/Linux + GPU | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3-4B` (text-only, vLLM compatible) |
| Windows/Linux + CPU | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` — เดิม |

`Qwen3.5-4B` เป็น Vision-Language model → vLLM พยายามโหลด vision encoder → Triton ล้มเหลว
จึงเปลี่ยนเป็น `Qwen3-4B` (text-only)

---

## 2. NER (Named Entity Recognition)

### 2.1 CPU Fallback — ข้ามถ้าไม่มี GPU
```python
if not USE_MLX and DEVICE == 'cpu':
    return {"species": [], "symptoms": [], ...}
```
NER บน CPU ใช้เวลา 60+ วินาที → timeout ทุกครั้ง แก้โดยคืน fallback ทันที

### 2.2 เปลี่ยนจาก raw tokenize → chat template
```python
# เก่า
inputs = llm_tok(prompt, return_tensors='pt')

# ใหม่
inputs = llm_tok.apply_chat_template(
    messages, return_tensors='pt', add_generation_prompt=True,
    enable_thinking=False,
)
```
`apply_chat_template` จัด format ถูกต้องตาม Qwen3 instruction format → โมเดลตอบ JSON ได้ดีขึ้น

### 2.3 ปิด Thinking Mode
`enable_thinking=False` — Qwen3 จะสร้าง `<think>...</think>` ก่อนตอบ → เพิ่มเวลาโดยไม่จำเป็นสำหรับ NER

### 2.4 ลด `max_new_tokens`
`512 → 200` — NER ต้องการแค่ JSON สั้นๆ ไม่จำเป็นต้องใช้ 512 token

### 2.5 Wrap try/except
```python
try:
    out = llm.generate(...)
except Exception as e:
    log.error(f"NER generate failed: {e}")
    return fallback_dict  # แทนที่จะ crash ทั้ง request
```

---

## 3. Inference Engine: transformers → vLLM

| ส่วน | เก่า (transformers) | ใหม่ (vLLM) |
|---|---|---|
| Loading | `AutoModelForCausalLM` + `BitsAndBytesConfig` | `AsyncLLMEngine` + `AsyncEngineArgs` |
| NER | `llm.generate(**inputs, ...)` | `_run_async(_vllm_full(prompt, 200))` |
| Streaming | `TextIteratorStreamer` + thread | async generator → `queue.Queue` |
| Tokenize | `return_tensors='pt'` + `.to(DEVICE)` | `tokenize=False` → string prompt |
| Quantization | bitsandbytes NF4 | fp8 (vLLM built-in) |

### Engine Config (CUDA)
```python
AsyncEngineArgs(
    dtype="bfloat16",
    quantization="fp8",        # bitsandbytes ต้องการ CUDA 13.x ซึ่งไม่มีใน CUDA 12.4
    max_model_len=1024,        # ลดเพื่อลด activation profiling memory
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)
```

### Multiprocessing Guard
vLLM v1 spawn subprocess ที่ re-import `app.py` → เกิด infinite loop
```python
if _mp.current_process().name == 'MainProcess':
    llm = AsyncLLMEngine.from_engine_args(_eargs)
```

### Async Bridge (vLLM async ↔ Flask sync)
```python
_vllm_loop = asyncio.new_event_loop()
threading.Thread(target=_vllm_loop.run_forever, daemon=True).start()

def _run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _vllm_loop).result()
```

---

## 4. VRAM Management

ย้าย peripheral models → CPU เพื่อให้ GPU ว่างสำหรับ vLLM

| Model | เก่า | ใหม่ | เหตุผล |
|---|---|---|---|
| Whisper | `device=DEVICE` (cuda) | `device="cpu"` | ให้ GPU แก่ vLLM |
| SentenceTransformer | `device=DEVICE` (cuda) | `device="cpu"` | ให้ GPU แก่ vLLM |
| Kokoro TTS | `device=DEVICE` (cuda) | `device="cpu"` | ให้ GPU แก่ vLLM |

**ก่อน/หลัง VRAM (RTX 4070 Laptop 8 GiB):**
```
ก่อน: free = 6.17/8.0 GiB  (Whisper + ST + Kokoro บน GPU)
หลัง: free = 6.87/8.0 GiB  → vLLM budget = 0.85×8 = 6.8 GiB ✓

vLLM budget breakdown:
  Model weights (fp8 Qwen3-4B): 4.12 GiB
  Activation profiling (1024t):  ~1.00 GiB
  KV cache:                      ~1.68 GiB ✓
```

---

## 5. ASR: Thai Language Support

เพิ่ม `nectec/Pathumma-whisper-th-medium` สำหรับภาษาไทย

**Flow:**
```
audio
  ↓
Whisper base — detect_language()  (encoder pass เดียว ~0.2-0.5s)
  ↓
detected_lang == "th" ?
  ├── YES → Pathumma-whisper-th-medium.transcribe()
  └── NO  → Whisper base.transcribe()
```

Whisper base โหลดครั้งเดียว ใช้ได้ 2 บทบาท:
- **detect language** — ทุก request (เร็ว ไม่ต้อง transcribe เต็ม)
- **transcribe EN** — เฉพาะเมื่อไม่ใช่ภาษาไทย

---

## 6. Frontend — AssessmentCard.tsx

**Duplicate React key bug:** TTS lines push ซ้ำเมื่อ SSE buffer parse event เดิม 2 รอบ

```tsx
// เก่า — อาจ push ซ้ำ
setTtsLines(prev => [...prev, { idx: evt.idx, sentence: evt.sentence }])

// ใหม่ — dedup on insert
setTtsLines(prev =>
  prev.some(x => x.idx === evt.idx) ? prev : [...prev, { idx: evt.idx, sentence: evt.sentence }]
)
```

---

## 7. Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

ENV LLM_MODEL=Qwen/Qwen3-4B
```

---

## สรุป Performance ที่คาดหวัง

| ส่วน | เก่า | ใหม่ |
|---|---|---|
| NER | ~2-3s (512 tokens) | ~0.5-1s (200 tokens + vLLM) |
| Expert LLM | ~6-8s (transformers) | ~2-3s (vLLM fp8) |
| Whisper EN | ~0.5s (GPU) | ~2-3s (CPU) |
| รวม (approx) | ~10-12s | ~5-7s |

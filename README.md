<div align="center">

# ZoonoMoE

### 🌍 *Frictionless zoonotic surveillance, routed at the edge.*

> <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Microbe.png" alt="Microbe" width="25" align="top" /> **Speak a field report. Get a veterinary risk assessment spoken back — fully on-device, in under 20 seconds.**

<p align="center">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=009688&center=true&vCenter=true&width=800&lines=Speak+a+field+report...;Get+a+veterinary+risk+assessment...;Fully+on-device+in+under+20+seconds!" alt="Typing SVG" /></a>
</p>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Triton](https://img.shields.io/badge/NVIDIA_Triton-26.02-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/triton-inference-server/server)
[![Whisper](https://img.shields.io/badge/Whisper-ASR-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-MIT-A6E3A1?style=for-the-badge)](LICENSE)

<br/>

```
 User says → "Three chickens died overnight, cyanotic combs, one found convulsing."
                              ↓  ~18 seconds later
 ZoonoMoE → "RISK LEVEL: HIGH — consistent with HPAI. Isolate birds. Report to DLD."
```

</div>

---

## 🌟 Pipeline at a Glance

```
Voice / Text
     │
     ▼
[1] ASR          Whisper + Pathumma-th (lang detect → route)
     │
     ▼
[2] NER          Qwen JSON extraction (species, symptoms, location ...)
     │
     ▼
[3] Router       all-MiniLM-L6-v2 + MLP → 6 disease domains
     │
     ▼
[4] RAG          FAISS per-domain → top-3 chunks
     │
     ▼
[5] LLM Expert   Qwen streaming → risk card (LOW/MEDIUM/HIGH/CRITICAL)
     │
     ▼
[6] TTS          Kokoro-82M → WAV chunks → AudioContext queue
```

**All processing is on-device — no cloud API, no data leaves your machine.**

---

## 💻 Platform Support

| Platform | ASR | LLM | TTS | Inference |
|---|---|---|---|---|
| Apple Silicon (Mac M-series) | mlx-whisper | mlx-lm Qwen3-4B | Kokoro local pool | MLX |
| Linux + NVIDIA GPU (cc≥8.0) | Whisper + Pathumma-th via **Triton** | vLLM + CUDA fp16 + compressed-tensors | Kokoro via **Triton** | Triton + vLLM |
| Linux + NVIDIA GPU (T4/cc=7.5) | Whisper + Pathumma-th via **Triton** | vLLM + CUDA fp16 (AWQ 8-bit) | Kokoro via **Triton** | Triton + vLLM |
| Linux CPU only | Whisper + Pathumma-th | vLLM CPU | Kokoro local pool | vLLM |

---

## 🚀 Quickstart

### Mac (Apple Silicon)

```bash
git clone https://github.com/E27-25/MLOps_Project.git
cd MLOps_Project

# Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
USE_MLX=true python3 app.py

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**

### Docker (Linux + NVIDIA GPU) — Recommended

```bash
git clone https://github.com/E27-25/MLOps_Project.git
cd MLOps_Project
docker compose up --build
```

Services:
- Frontend: **http://localhost:3000**
- Backend API: **http://localhost:7860**
- Triton: **http://localhost:8000**

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Next.js :3000  │
└────────┬────────┘
         │ REST + SSE
         ▼
┌──────────────────────────────────────────┐
│           FastAPI Backend :7860          │
│                                          │
│  /upload  → Triton whisper_asr          │
│  /analyze → Triton sentence_embedder    │
│           → vLLM NER                    │
│           → FAISS RAG                   │
│  /stream  → vLLM SSE stream             │
│           → Triton kokoro_tts           │
└──────────────────┬───────────────────────┘
                   │ HTTP
                   ▼
┌──────────────────────────────────────────┐
│        NVIDIA Triton Server :8000        │
│                                          │
│  whisper_asr       ×2 CPU instances     │
│    Whisper base (lang detect)           │
│    + Pathumma-whisper-th (Thai)         │
│                                          │
│  sentence_embedder  ×1 CPU instance     │
│    all-MiniLM-L6-v2                     │
│                                          │
│  kokoro_tts        ×3 CPU instances     │
│    Kokoro-82M → WAV → base64            │
└──────────────────────────────────────────┘
```

> **Why not LLM in Triton?** vLLM SSE streaming requires Triton Decoupled Mode (gRPC only) — significant added complexity for no throughput gain since vLLM already handles concurrency internally via continuous batching + PagedAttention.

---

## ⚡ Concurrency

| Component | Parallel capacity | Mechanism |
|---|---|---|
| Whisper ASR | ×2 simultaneous | Triton instance group |
| Sentence Embedder | ×1 (fast, async) | Triton |
| Kokoro TTS | ×3 simultaneous | Triton instance group |
| vLLM LLM | N requests batched | Continuous batching + PagedAttention |

Multiple users are served concurrently — no request blocks another.

---

## 🔄 Step-by-Step Workflow

<details>
<summary><b>Step 1 — ASR</b></summary>
<br/>

| | |
|---|---|
| **Input** | WebM blob from browser `MediaRecorder` |
| **Convert** | `ffmpeg` → 16 kHz mono WAV |
| **Lang detect** | Whisper base `detect_language()` |
| **Thai** | `nectec/Pathumma-whisper-th-medium` |
| **Other** | `openai/whisper-base` |
| **Guard** | Word repeat >6× or top-3 words >80% → reject, prompt re-record |

</details>

<details>
<summary><b>Step 2 — NER</b></summary>
<br/>

Structured JSON-mode prompt extracts 8 fields:

| Field | Example |
|---|---|
| `species` | `["chicken", "duck"]` |
| `symptoms` | `["cyanotic combs", "twisted neck"]` |
| `mortality_count` | `30` |
| `affected_count` | `50` |
| `location` | `"near the pond"` |
| `timeframe` | `"this morning"` |
| `reporter_role` | `"farmer"` |
| `raw_summary` | one-liner for downstream prompts |

</details>

<details>
<summary><b>Step 3 — MoE Router</b></summary>
<br/>

`all-MiniLM-L6-v2` sentence embeddings → `MLPClassifier(128, 64)` → 6 disease domains

| Domain | Training samples |
|---|---|
| `avian_flu` | 27 |
| `fmd` | 25 |
| `general` | 25 |
| `leptospirosis` | 26 |
| `nipah_hendra` | 30 |
| `rabies` | 27 |

**Cross-val F1 (macro, 5-fold): `0.740`**

Off-topic guard: regex checks for greetings/chat before MLP runs. If matched with no mortality signal → routes to `general` friendly advisor.

To retrain:
```bash
python3 -c "
from pathlib import Path
from models.router import train
train(model_dir=Path('models'), extra_data=Path('data/router_training.jsonl'))
"
```

</details>

<details>
<summary><b>Step 4 — RAG</b></summary>
<br/>

- One FAISS index per disease domain (`knowledge_base/{domain}/index.pkl`)
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Top-3 chunks by cosine similarity
- Fallback to built-in seed knowledge if index missing

</details>

<details>
<summary><b>Step 5 — LLM Expert</b></summary>
<br/>

| Platform | Model | Backend |
|---|---|---|
| Apple Silicon | `mlx-community/Qwen3-4B-4bit` | mlx-lm |
| CUDA (cc≥8.0) | `Qwen/Qwen3-4B` fp16 | vLLM |
| CUDA T4 (cc=7.5) | `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-8bit` | vLLM + compressed-tensors |
| CPU | `Qwen/Qwen2.5-1.5B-Instruct` | vLLM |

- Domain-specific expert persona + NER fields + RAG context
- `<think>...</think>` blocks stripped from SSE stream
- Output: live text stream + `RISK LEVEL` badge + `report_to_authorities` flag

</details>

<details>
<summary><b>Step 6 — TTS</b></summary>
<br/>

- Model: `hexgrad/Kokoro-82M`
- Each sentence → WAV → base64 SSE → `AudioContext` queue
- First audio plays ~3–5 s after LLM starts (no wait for full response)
- Voice per domain: `af_heart` / `am_michael` / `am_adam` / `af_bella`
- Controls: Pause / Resume / Stop

</details>

---

## 🧪 Test Inputs

| Domain | Input |
|---|---|
| Avian Flu | `30 chickens died this morning with purple combs and twisted necks, labored breathing` |
| FMD | `My cattle have blisters on tongue and feet, limping badly and salivating heavily` |
| Nipah/Hendra | `Pig farmer here — pigs died suddenly overnight, two workers now have fever and confusion` |
| Leptospirosis | `Five rice farmers have fever, muscle pain, red eyes after wading in flooded paddy fields` |
| Rabies | `A stray dog bit two children — foaming at the mouth, running in circles before collapsing` |
| Chat | `Hi, how do I protect myself when working near livestock?` |

---

## 📁 Project Structure

```
MLOps_Project/
├── docker-compose.yml            # Triton + Backend + Frontend
├── triton.Dockerfile             # Custom Triton image with Python deps
│
├── triton_models/                # Triton model repository
│   ├── whisper_asr/              # Whisper + Pathumma-th (×2 instances)
│   │   ├── config.pbtxt
│   │   └── 1/model.py
│   ├── sentence_embedder/        # all-MiniLM-L6-v2 (×1 instance)
│   │   ├── config.pbtxt
│   │   └── 1/model.py
│   └── kokoro_tts/               # Kokoro-82M (×3 instances)
│       ├── config.pbtxt
│       └── 1/model.py
│
├── backend/
│   ├── app.py                    # FastAPI — 6-stage pipeline
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── data/
│   │   └── router_training.jsonl
│   ├── knowledge_base/           # FAISS indexes per domain
│   │   ├── avian_flu/
│   │   ├── fmd/
│   │   ├── nipah_hendra/
│   │   ├── rabies/
│   │   ├── leptospirosis/
│   │   └── general/
│   ├── models/
│   │   ├── router.pkl
│   │   ├── router_meta.json
│   │   └── router.py
│   ├── scripts/
│   │   ├── discord_logger.py
│   │   └── evaluate.py
│   ├── static/
│   └── templates/
│
└── frontend/
    ├── src/app/                  # Next.js 15 App Router
    └── Dockerfile
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `USE_MLX` | `auto` | `true` = MLX, `false` = vLLM, `auto` = detect |
| `USE_TRITON` | `false` | Enable Triton for ASR / Embedder / TTS |
| `TRITON_URL` | `triton:8000` | Triton server address |
| `LLM_MODEL` | _(platform default)_ | HuggingFace model repo |
| `WHISPER_SIZE` | `base` | `tiny` · `base` · `small` · `medium` |
| `PORT` | `7860` | Backend port |
| `DEBUG` | `false` | FastAPI debug mode |
| `DISCORD_WEBHOOK` | _(empty)_ | Discord Forum webhook for pipeline logging |
| `DATABASE_URL` | _(empty)_ | PostgreSQL connection string (Neon) |

---

## 🐳 Docker Deployment

```bash
# GPU (Triton enabled)
docker compose up --build

# With Discord logging + Database
DATABASE_URL="postgresql://..." DISCORD_WEBHOOK="https://discord.com/api/webhooks/..." docker compose up --build
```

Persisted volumes:

| Volume | Purpose |
|---|---|
| `./backend/knowledge_base` | RAG FAISS indexes |
| `./backend/models` | Trained MLP router |
| `./backend/data` | Router training data |
| `huggingface_cache` | Downloaded HuggingFace models |

> **Note:** `triton_models/` is baked into the Triton Docker image via `COPY triton_models /models` — no bind mount required.

---

## ☁️ Google Cloud Deployment (GPU VM)

```bash
# 1. Build + push images
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
docker compose build
docker tag zoonmoe-backend:latest asia-southeast1-docker.pkg.dev/<PROJECT_ID>/zoonotic/backend:latest
docker tag zoonmoe-triton:latest  asia-southeast1-docker.pkg.dev/<PROJECT_ID>/zoonotic/triton:latest
docker tag zoonmoe-frontend:latest asia-southeast1-docker.pkg.dev/<PROJECT_ID>/zoonotic/frontend:latest
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/zoonotic/backend:latest
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/zoonotic/triton:latest
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/zoonotic/frontend:latest

# 2. On VM — pull and run
docker compose pull
DATABASE_URL="postgresql://..." docker compose up -d
```

Requirements: NVIDIA driver ≥ 590, `nvidia-container-toolkit`, GPU with ≥ 16 GB VRAM (T4 or better).

---

## 📊 Discord Pipeline Logging

Every inference run logged to a Discord Forum channel as its own thread.

```bash
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
python3 backend/scripts/discord_logger.py   # dry-run test
```

| Stage | Logged data |
|---|---|
| ASR | Backend, latency, transcript |
| NER | All 8 extracted fields |
| Router | Domain, confidence, all 6 scores |
| RAG | Chunks, latency, sources |
| LLM | Risk level, report flag, full assessment |
| TTS | Audio chunks, latency |
| Summary | Total latency + CPU/RAM metrics |

---

## 📈 Performance

| Metric | Value |
|---|---|
| Router F1 (macro, 6-class) | **0.740** |
| ASR latency | ~1–2 s (Whisper base) |
| Full pipeline | ~15–20 s end-to-end |
| TTS first chunk | ~3–5 s after LLM starts |
| Concurrent users | ×3 TTS / ×2 ASR / vLLM batched |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| ASR | Whisper + Pathumma-whisper-th / mlx-whisper |
| LLM / NER | Qwen3 via vLLM (CUDA) or mlx-lm (Apple Silicon) |
| Router | all-MiniLM-L6-v2 + scikit-learn MLP |
| RAG | FAISS + sentence-transformers |
| TTS | Kokoro-82M |
| Inference Server | NVIDIA Triton 26.02 (ASR + Embedder + TTS) |
| Backend | FastAPI + SSE streaming |
| Frontend | Next.js 15 + Three.js / R3F |
| Database | Neon PostgreSQL — SQLAlchemy (backend) + Drizzle ORM (frontend) |
| Dashboard | Next.js `/dashboard` — report history, stats, risk distribution |
| Experiment Tracking | discordflow — team-built Discord Forum ML logger |

---

<div align="center">

*Built for MLOps coursework · KMITL · 2026*

</div>

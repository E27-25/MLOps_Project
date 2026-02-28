<div align="center">

# 🦠 ZoonoMoE

### *Frictionless zoonotic surveillance, routed at the edge.*

> **Speak a field report. Get a veterinary risk assessment spoken back — fully on-device, in under 20 seconds.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MLX](https://img.shields.io/badge/Apple_MLX-Accelerated-FF6B35?style=for-the-badge&logo=apple&logoColor=white)](https://ml-explore.github.io/mlx/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
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

## ⚡ Pipeline at a Glance

```
🎤 Voice / ⌨️  Text
        │
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  [1] ASR     │ ──▶│  [2] NER     │ ──▶│  [3] Router  │ ──▶│  [4] RAG     │ ──▶│  [5] Expert  │ ──▶│  [6] TTS     │
│  MLX Whisper │    │  Qwen3 JSON  │    │  MiniLM+MLP  │    │  FAISS/domain│    │  Qwen3-4B    │    │  Kokoro-82M  │
│  + halluc.   │    │  extraction  │    │  6 domains   │    │  top-3 chunks│    │  streaming   │    │  streaming   │
│  guard       │    │              │    │  + off-topic │    │              │    │  risk card   │    │  audio       │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**All processing is on-device — no cloud API, no data leaves your machine.**

---

## 🚀 Quickstart

**Requirements:** macOS + Apple Silicon (M1–M4) · Python 3.11+ · `ffmpeg` in PATH

```bash
# Clone and install
git clone https://github.com/E27-25/MLOps_Project.git
cd MLOps_Project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
python3 app.py
```

Open **http://localhost:7860** — landing page → click **Launch App** → record or type a field report.

```bash
# Optional: reduce memory usage with a smaller Whisper model
WHISPER_SIZE=tiny python3 app.py
```

---

## 🧬 Step-by-Step Workflow

<details>
<summary><b>Step 1 — ASR · MLX Whisper</b></summary>
<br/>

| | |
|---|---|
| **Model** | `mlx-community/whisper-base-mlx` |
| **Input** | WebM blob from browser `MediaRecorder` |
| **Convert** | `ffmpeg` → 16 kHz mono WAV |
| **Guard** | Single word repeating >6× or top-3 words covering >80% → rejected, user prompted to re-record |

```python
result = mlx_whisper.transcribe(wav_path, path_or_hf_repo="mlx-community/whisper-base-mlx")
transcript = result["text"].strip()
```

</details>

<details>
<summary><b>Step 2 — NER · Qwen3 JSON Extraction</b></summary>
<br/>

Structured JSON-mode prompt extracts 8 fields from the raw transcript:

| Field | Example output |
|---|---|
| `species` | `["chicken", "duck"]` |
| `symptoms` | `["cyanotic combs", "twisted neck"]` |
| `mortality_count` | `30` |
| `affected_count` | `50` |
| `location` | `"near the pond"` |
| `timeframe` | `"this morning"` |
| `reporter_role` | `"farmer"` |
| `raw_summary` | Plain one-liner for downstream prompts |

</details>

<details>
<summary><b>Step 3 — MoE Router · MiniLM + MLP</b></summary>
<br/>

**Architecture:** `all-MiniLM-L6-v2` sentence embeddings → `MLPClassifier(128, 64)` → 6 disease domains

**Off-topic guard:** Regex checks for greetings/chat before the MLP runs. If matched with no mortality signal → routes to `general` instantly.

| Domain | Training samples |
|---|---|
| `avian_flu` | 27 |
| `fmd` | 25 |
| `general` | 25 |
| `leptospirosis` | 26 |
| `nipah_hendra` | 30 |
| `rabies` | 27 |

**Cross-val F1 (macro, 5-fold): `0.740`**

To retrain after editing `data/router_training.jsonl`:
```bash
python3 -c "
from pathlib import Path
from models.router import train
train(model_dir=Path('models'), extra_data=Path('data/router_training.jsonl'))
"
```

</details>

<details>
<summary><b>Step 4 — RAG · Per-domain FAISS</b></summary>
<br/>

- One vector index per disease domain (`knowledge_base/{domain}/index.pkl`)
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Top-3 chunks retrieved by cosine similarity
- Fallback: built-in seed knowledge if index fails to load (no crash)
- Retrieved chunks shown in the **RAG Sources** collapsible panel

</details>

<details>
<summary><b>Step 5 — LLM Expert · Qwen3-4B streaming</b></summary>
<br/>

- Model: `mlx-community/Qwen3-4B-4bit`
- Domain-specific expert persona prompt + NER fields + RAG context
- `<think>...</think>` blocks stripped transparently from the SSE stream
- Output includes: live-streamed text, `RISK LEVEL` badge, `report_to_authorities` flag
- Markdown rendered client-side (`**bold**`, `*italic*`, bullet lists)

</details>

<details>
<summary><b>Step 6 — TTS · Kokoro-82M streaming</b></summary>
<br/>

- Model: `hexgrad/Kokoro-82M`, voice `af_heart`
- Each complete sentence → WAV chunk → base64 SSE event → `AudioContext` queue
- **No wait** for full response — first audio plays ~3–5s after LLM starts
- Controls: ⏸ Pause · ▶ Resume · ⏹ Stop

</details>

---

## 🧪 Test Inputs

Use these in the text field to verify domain routing:

| Domain | Input |
|---|---|
| 🐦 Avian Flu | `30 chickens died this morning with purple combs and twisted necks, labored breathing` |
| 🐄 FMD | `My cattle have blisters on tongue and feet, limping badly and salivating heavily` |
| 🐖 Nipah/Hendra | `Pig farmer here — pigs died suddenly overnight, two workers now have fever and confusion` |
| 🐀 Leptospirosis | `Five rice farmers have fever, muscle pain, red eyes after wading in flooded paddy fields` |
| 🐕 Rabies | `A stray dog bit two children — foaming at the mouth, running in circles before collapsing` |
| 💬 Chat | `Hi, how do I protect myself when working near livestock?` |

---

## 🗂️ Project Structure

```
MLOps_Project/
├── app.py                        # Flask app — full 6-stage pipeline
├── requirements.txt
├── setup.sh                      # First-run setup
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   └── router_training.jsonl     # 100 training examples (6 domains)
│
├── knowledge_base/
│   ├── avian_flu/                # FAISS index + raw veterinary docs
│   ├── fmd/
│   ├── nipah_hendra/
│   ├── rabies/
│   ├── leptospirosis/
│   └── general/
│
├── models/
│   ├── router.pkl                # Trained MLP classifier
│   ├── router_meta.json
│   └── router.py                 # Training + inference code
│
├── static/
│   ├── css/style.css             # Catppuccin Mocha theme
│   ├── js/app.js                 # Frontend pipeline controller
│   └── favicon.svg
│
├── templates/
│   ├── landing.html              # Landing page (orbital particle canvas)
│   └── index.html                # Main app UI
│
├── scripts/
│   ├── discord_logger.py         # Forum-mode pipeline + retrain logger
│   └── evaluate.py               # Router evaluation script
│
└── utils/
    └── rag.py                    # RAG retrieval + compat unpickler
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `USE_MLX` | `1` | MLX Whisper (`1`) or OpenAI Whisper (`0`) |
| `LLM_MODEL` | `mlx-community/Qwen3-4B-4bit` | LLM repo path |
| `WHISPER_SIZE` | `base` | `tiny` · `base` · `small` · `medium` |
| `PORT` | `7860` | Server port |
| `DEBUG` | `0` | Flask debug mode |
| `DISCORD_WEBHOOK` | _(empty)_ | Discord Forum webhook URL for pipeline logging |

---

## 🐳 Docker Deployment

> For cloud / Linux servers. Mac M-series users should run natively — Docker won't use MLX/GPU.

```bash
# Build and start
docker compose up --build

# With Discord logging enabled
DISCORD_WEBHOOK="https://discord.com/api/webhooks/..." docker compose up --build
```

Persisted volumes (survive container restarts):

| Volume | Purpose |
|---|---|
| `./knowledge_base` | RAG FAISS indexes |
| `./models` | Trained MLP router |
| `./data` | Router training data |
| `./scripts` | Logger + evaluation scripts |

The container automatically health-checks `GET /health` every 30 s with a 60 s grace period for model loading.

---

## 📡 Discord Pipeline Logging

Every inference run can be logged to a **Discord Forum channel** as its own thread.

```bash
pip install "discordflow[system]"
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
python3 app.py
```

Each Forum thread contains:

| Stage | Logged data |
|---|---|
| ASR | Backend, model size, latency, transcript attachment |
| NER | All 8 extracted fields |
| Router | Domain, confidence, all 6 domain scores |
| RAG | Chunks count, latency, `rag_sources.txt` attachment |
| LLM | Risk level, report flag, `llm_assessment.txt` attachment |
| TTS | Audio chunks, latency |
| Summary | Total latency + CPU/RAM system metrics |

Test without a real webhook (dry-run prints to stdout):

```bash
python3 scripts/discord_logger.py
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| Router F1 (macro, 6-class) | **0.740** |
| ASR latency | **~1–2 s** (Whisper base, Apple M-series) |
| Full pipeline | **~15–20 s** end-to-end |
| TTS first chunk | **~3–5 s** after LLM starts |

---

## ⚠️ Known Issues

| Issue | Cause | Fix |
|---|---|---|
| Segfault (exit 139) | MLX + PyTorch competing for unified memory | Use `WHISPER_SIZE=tiny` or upgrade to ≥32 GB RAM |
| Whisper hallucination | Noisy/silent audio causes token loop | Hallucination guard rejects + prompts re-record |
| Router misclassification on edge cases | Overlapping symptom profiles | Retrain with expanded `router_training.jsonl` |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| ASR | [MLX Whisper](https://github.com/ml-exploration/mlx-examples) |
| LLM / NER | [Qwen3-4B-4bit](https://huggingface.co/mlx-community/Qwen3-4B-4bit) via MLX |
| Router | `all-MiniLM-L6-v2` + scikit-learn MLP |
| RAG | FAISS + `sentence-transformers` |
| TTS | [Kokoro-82M](https://github.com/hexgrad/kokoro) |
| Backend | Flask 3.x + SSE streaming |
| Frontend | Vanilla JS · Canvas API · Web Audio API |

---

<div align="center">

*Built for MLOps coursework · KMITL · 2026*

</div>

# ZoonoMoE

**Frictionless zoonotic surveillance, routed at the edge.**

> *"Speak a field report. Get a veterinary risk assessment spoken back — fully on-device, in under 20 seconds."*

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![MLX](https://img.shields.io/badge/Apple%20MLX-Accelerated-orange?style=flat-square)](https://ml-explore.github.io/mlx/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## What It Does

```
User says:   "Three chickens died overnight, cyanotic combs,
              one found convulsing near the pond."
                             ↓  ~18 seconds
ZoonoMoE speaks:
  "Based on the neurological signs and cyanosis in your poultry,
   this is consistent with Highly Pathogenic Avian Influenza.
   RISK LEVEL: HIGH. Isolate surviving birds immediately and
   contact your District Livestock Officer. Report to authorities."
```

**All processing is local — no cloud API, no internet required after first model download.**

---

## Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       ZoonoMoE v2.0                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🎤 Voice Input / ⌨️  Text Input                                 │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │  [1] ASR    │  MLX Whisper — real-time speech-to-text       │
│   │  Whisper    │  + hallucination guard (repetition filter)     │
│   └──────┬──────┘                                               │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  [2] NER    │  Qwen3-4B JSON-mode → extracts:               │
│   │  Qwen3      │  species, symptoms, mortality, location,       │
│   └──────┬──────┘  timeframe, reporter role                     │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  [3] Router │  Off-topic guard → MiniLM + MLP classifier    │
│   │  MoE        │  → 6 disease domains                          │
│   └──────┬──────┘  (avian_flu / fmd / nipah_hendra /           │
│          │          rabies / leptospirosis / general)           │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  [4] RAG    │  FAISS per-domain vector search               │
│   │  Retrieval  │  → top-3 veterinary document chunks           │
│   └──────┬──────┘                                               │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │ [5] LLM      │  Qwen3 Expert streams risk card + assessment  │
│   │ Expert Agent │  <think> tags stripped from output           │
│   └──────┬───────┘                                              │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  [6] TTS    │  Kokoro — sentence-by-sentence streaming audio │
│   │  Kokoro     │  (no waiting for full response)               │
│   └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Workflow

### Step 1 — ASR (Automatic Speech Recognition)

- **Model:** MLX Whisper (`whisper-base-mlx` by default)
- **Input:** WebM audio blob from browser MediaRecorder API
- **Process:** `ffmpeg` converts WebM → 16kHz mono WAV → `mlx_whisper.transcribe()`
- **Hallucination guard:** If any single word repeats >6 times consecutively, or the top 3 words cover >80% of the transcript, the result is discarded with an error message. This catches Whisper's looping bug on noisy audio.
- **Output:** Raw transcript string + timing info

```python
# app.py — /upload route
result = mlx_whisper.transcribe(wav_path, path_or_hf_repo="mlx-community/whisper-base-mlx")
transcript = result["text"].strip()
```

---

### Step 2 — NER (Named Entity Recognition)

- **Model:** Qwen3-4B-4bit (MLX) with a structured JSON prompt
- **Input:** Raw transcript text
- **Extracted fields:**

| Field | Example |
|---|---|
| `species` | `["chicken", "duck"]` |
| `symptoms` | `["cyanotic combs", "twisted neck"]` |
| `mortality_count` | `30` |
| `affected_count` | `50` |
| `location` | `"near the pond"` |
| `timeframe` | `"this morning"` |
| `reporter_role` | `"farmer"` |
| `raw_summary` | one-line plain summary |

- **Output:** JSON dict shown in the collapsible **EPI Fields** panel

---

### Step 3 — MoE Router (Domain Classification)

- **Architecture:** MLP classifier (`hidden_layer_sizes=(128, 64)`) on top of `all-MiniLM-L6-v2` sentence embeddings
- **Training data:** `data/router_training.jsonl` — 100 diverse field report examples across 6 domains
- **Off-topic guard:** Before the MLP is consulted, a regex checks for greetings / general questions. If matched AND no mortality signal → route to `general` without calling the MLP
- **Output:** Chosen domain + confidence score + all 6 domain scores (shown as normalized confidence bars in UI)

**Domain distribution after training:**

| Domain | Training Samples |
|---|---|
| `avian_flu` | 27 (including seed + JSONL) |
| `fmd` | 25 |
| `general` | 25 |
| `leptospirosis` | 26 |
| `nipah_hendra` | 30 |
| `rabies` | 27 |

**Router cross-val F1 (macro, 5-fold): 0.740**

---

### Step 4 — RAG (Retrieval-Augmented Generation)

- **Vector store:** FAISS (one index per disease domain)
- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2`
- **Source docs:** `knowledge_base/{domain}/` — curated veterinary guidelines per domain
- **Process:** Query text is embedded and top-3 chunks (by cosine similarity) are retrieved from the selected domain's index
- **Fallback:** If the FAISS index fails to load, the system uses seed text chunks so the pipeline never breaks
- **Output:** 3 text chunks displayed in the collapsible **RAG Sources** panel

---

### Step 5 — LLM Expert (Qwen3 Assessment)

- **Model:** `mlx-community/Qwen3-4B-4bit` via `mlx-lm`
- **Input:** Domain + NER fields + RAG chunks
- **Prompt:** Domain-specific expert persona (e.g. "Avian Influenza Expert") with structured RAG context
- **Think-tag stripping:** Qwen3 uses `<think>...</think>` reasoning blocks — these are stripped from the streamed output before the user sees them
- **Output:** SSE stream of text tokens → displayed live in the **Assessment** panel with:
  - `RISK LEVEL` badge (LOW / MEDIUM / HIGH / CRITICAL)
  - `report_to_authorities` flag
  - Markdown rendered: `*italic*` → *italic*, `**bold**` → **bold**, bullet lists

---

### Step 6 — TTS (Text-to-Speech)

- **Model:** Kokoro-82M (`hexgrad/Kokoro-82M`) via PyTorch, voice `af_heart`
- **Streaming:** Each complete sentence from the LLM is immediately synthesised as a WAV chunk and sent via SSE as base64 audio
- **Playback:** Browser queues audio chunks and plays them sequentially using `AudioContext`
- **Controls:**
  - **⏸ Pause** — pauses current audio, preserves queue, shows ▶ Resume
  - **▶ Resume** — resumes from same position
  - **Full stop** — clears all queued and current audio

---

## Pages

| URL | Page |
|-----|------|
| `http://localhost:7860/` | **Landing page** — animated showcase with orbital particle background |
| `http://localhost:7860/app` | **App** — full voice-to-assessment interface |

---

## Project Structure

```
MLOps_Project/
├── app.py                        # Flask app — full 6-stage pipeline
├── requirements.txt
├── setup.sh                      # First-run setup script
├── Dockerfile                    # Docker deployment
├── docker-compose.yml
│
├── data/
│   └── router_training.jsonl     # 100 training examples (6 domains)
│
├── knowledge_base/
│   ├── avian_flu/                # Veterinary docs for RAG retrieval
│   ├── fmd/
│   ├── nipah_hendra/
│   ├── rabies/
│   ├── leptospirosis/
│   └── general/
│
├── models/
│   ├── router.pkl                # Trained MLP classifier
│   ├── router_meta.json          # Label mapping + domain info
│   └── router.py                 # Router training + inference code
│
├── scripts/
│   └── evaluate.py               # Router evaluation script
│
├── static/
│   ├── css/style.css             # Catppuccin Mocha theme + all UI styles
│   ├── js/app.js                 # Frontend pipeline controller
│   └── favicon.svg               # ZoonoMoE "Zx" brand mark
│
├── templates/
│   ├── landing.html              # Landing page (orbital particle canvas BG)
│   └── index.html                # Main app UI
│
└── utils/
    ├── __init__.py
    └── rag.py                    # RAG retrieval utilities
```

---

## Quickstart

### Requirements
- macOS + Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- `ffmpeg` in PATH

### Install

```bash
git clone https://github.com/E27-25/MLOps_Project.git
cd MLOps_Project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python3 app.py
```

Then open **http://localhost:7860**

### Retrain the Router

If you add new training data to `data/router_training.jsonl`:

```bash
python3 -c "
from pathlib import Path
from models.router import train
train(model_dir=Path('models'), extra_data=Path('data/router_training.jsonl'))
"
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MLX` | `1` | Use MLX Whisper (`1`) or OpenAI Whisper (`0`) |
| `LLM_MODEL` | `mlx-community/Qwen3-4B-4bit` | LLM model path |
| `WHISPER_SIZE` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium` |
| `PORT` | `7860` | Server port |
| `DEBUG` | `0` | Flask debug mode |

```bash
# Example: reduce memory usage with smaller Whisper
WHISPER_SIZE=tiny python3 app.py
```

---

## Example Field Reports (for Testing)

Use these to test domain routing in the text input:

| Domain | Example Input |
|--------|--------------|
| **Avian Flu** | `30 chickens died this morning with purple combs and twisted necks, labored breathing` |
| **FMD** | `My cattle have blisters on their tongue and feet, limping badly and salivating heavily` |
| **Nipah/Hendra** | `I'm a pig farmer, pigs died suddenly overnight, two workers who touched them now have fever and confusion` |
| **Leptospirosis** | `Five rice farmers have fever, muscle pain, and red eyes after wading in flooded paddy fields last week` |
| **Rabies** | `A stray dog bit two children, it was foaming, drooling, and running in circles before collapsing` |
| **General** | `Hi, how do I protect myself from disease when working near livestock?` |

---

## UI Features

- **Orbital particle background** — 70 particles trace elliptical orbits at variable speed/tilt; cursor disrupts orbits; clicks trigger multi-ring explosions with spark trails
- **Typewriter hero** — cycles through short captions (≤ 2 lines each)
- **Live pipeline tracker** — per-step status dots (ASR → NER → Router → RAG → Expert → TTS) with timing
- **Risk card** — live-streamed assessment text with RISK LEVEL badge (LOW / MEDIUM / HIGH / CRITICAL)
- **EPI Fields panel** — collapsible table of structured NER output
- **Domain Confidence panel** — bar chart of all 6 domain scores (normalized; selected domain always pinned first)
- **RAG Sources panel** — collapsible list of retrieved knowledge chunks
- **TTS audio bar** — sentence-by-sentence playback with ⏸ Pause / ▶ Resume
- **Session history** — collapsible log of past reports and assessments
- **Whisper hallucination guard** — noisy/garbled audio shows user-friendly error instead of garbage text

---

## Known Issues & Mitigations

| Issue | Cause | Mitigation |
|-------|-------|------------|
| Segfault (exit 139) on inference | MLX + PyTorch competing for unified memory when running Qwen3 + Kokoro simultaneously | Use `WHISPER_SIZE=tiny`, or upgrade to ≥32 GB RAM Mac |
| RAG index load warning (`SimpleVectorStore`) | Indexes pickled under a different Python context | Rebuild indexes by running `setup.sh` or the RAG build script |
| Whisper hallucination (`cucucucuc...`) | Noisy/silent audio causes Whisper token loop | Hallucination guard in `/upload` now rejects and prompts re-record |
| Router misclassification on edge cases | Overlapping symptom profiles between domains | Expanded training data to 100 samples; retrain with `models/router.py` |

---

## Performance

| Metric | Value |
|--------|-------|
| Router F1 (macro, 6-class) | **0.740** (100-sample dataset) |
| ASR latency | **~1–2s** (Whisper base, Apple M-series) |
| Full pipeline | **~15–20s** end-to-end |
| TTS first chunk | **~3–5s** after LLM starts |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| ASR | [MLX Whisper](https://github.com/ml-exploration/mlx-examples) |
| LLM | [Qwen3-4B-4bit](https://huggingface.co/mlx-community/Qwen3-4B-4bit) via MLX |
| NER | Qwen3 with JSON-mode structured prompt |
| Router | `all-MiniLM-L6-v2` + MLP (scikit-learn) |
| RAG | FAISS + `sentence-transformers` |
| TTS | [Kokoro-82M](https://github.com/hexgrad/kokoro) streaming |
| Backend | Flask 3.x + SSE streaming |
| Frontend | Vanilla JS + Canvas API + Web Audio API |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built for MLOps coursework · Chulalongkorn University · 2026*

# 🦠 ZoonoticSense

**Voice-Driven Zoonotic Disease Surveillance System**

> *"Speak a field report. Hear a veterinary risk assessment spoken back in under 20 seconds."*

ZoonoticSense is a fully on-device, voice-first AI pipeline that converts natural speech from farmers, rangers, and field workers into structured epidemiological risk assessments — spoken aloud via TTS.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![MLX](https://img.shields.io/badge/Apple%20MLX-Accelerated-orange?style=flat-square)](https://ml-explore.github.io/mlx/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## 🎯 What It Does

```
User says:   "Three chickens died overnight, cyanotic combs,
              one found convulsing near the pond."
                           ↓  ~18 seconds
ZoonoticSense speaks:
  "Based on the neurological signs and cyanosis in your poultry,
   this is consistent with Highly Pathogenic Avian Influenza.
   RISK LEVEL: HIGH. Isolate surviving birds immediately and
   contact your District Livestock Officer. Report to authorities."
```

**All processing is local — no cloud API, no internet required.**

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZoonoticSense v2.0                          │
├────────────────────────────────────────────────────────────────-┤
│                                                                  │
│  🎤 Voice / ⌨️ Text                                              │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │  Whisper ASR │  MLX Whisper — real-time transcription       │
│   └──────┬──────┘                                               │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  NER (Qwen3) │  Extracts species / symptoms / mortality /   │
│   └──────┬──────┘  location / timeframe (JSON mode)            │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  MoE Router  │  MLP on MiniLM embeddings → 6 domains       │
│   └──────┬──────┘  avian_flu / fmd / nipah_hendra /            │
│          │         rabies / leptospirosis / general             │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  RAG Lookup  │  FAISS per-domain vector search →            │
│   └──────┬──────┘  top-3 veterinary document chunks            │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │  Qwen3 Expert │  Streams risk card + assessment text        │
│   └──────┬───────┘                                             │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  Kokoro TTS  │  Sentence-by-sentence streaming audio        │
│   └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Pages

| URL | Page |
|-----|------|
| `http://localhost:7860/` | **Landing page** — overview and feature showcase |
| `http://localhost:7860/app` | **App** — main analysis interface |

---

## ⚡ Quickstart

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

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MLX` | `1` | Use MLX Whisper (`1`) or OpenAI Whisper (`0`) |
| `LLM_MODEL` | `mlx-community/Qwen3-4B-4bit` | LLM model path |
| `WHISPER_SIZE` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium` |
| `PORT` | `7860` | Server port |
| `DEBUG` | `0` | Flask debug mode |

```bash
# Example: use smaller whisper for faster ASR
WHISPER_SIZE=tiny python3 app.py
```

---

## 🗂️ Project Structure

```
MLOps_Project/
├── app.py                   # Flask app + pipeline logic
├── requirements.txt
├── data/
│   └── router_training.jsonl   # Training data for domain router
├── rag_docs/
│   └── {domain}/               # Veterinary documents per domain
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   ├── landing.html            # Landing page (/)
│   └── index.html              # App (/app)
└── utils/
    └── router.py               # MoE domain classifier
```

---

## 🦠 Supported Disease Domains

| Domain | Trigger Keywords |
|--------|-----------------|
| `avian_flu` | avian influenza, HPAI, H5N1, bird flu, cyanotic comb |
| `fmd` | foot-and-mouth, blister, vesicle, drool, lame |
| `nipah_hendra` | bat, encephalitis, nipah, hendra, flying fox |
| `rabies` | bite, aggression, hydrophobia, paralysis, mad dog |
| `leptospirosis` | rat, flood, urine, jaundice, leptospira |
| `general` | off-topic / greetings / general questions |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| ASR | [MLX Whisper](https://github.com/ml-exploration/mlx-examples) |
| LLM | [Qwen3-4B-4bit](https://huggingface.co/mlx-community/Qwen3-4B-4bit) via MLX |
| NER | Qwen3 with JSON-mode prompt |
| Router | MiniLM + MLP (scikit-learn) |
| RAG | FAISS + `sentence-transformers` |
| TTS | [Kokoro](https://github.com/hexgrad/kokoro) |
| Backend | Flask 3.x + SSE streaming |
| Frontend | Vanilla JS + Canvas API |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Router F1 | **0.83** (6-class) |
| ASR latency | **~2s** (Whisper base, M4) |
| Full pipeline | **~15–20s** end-to-end |
| TTS first chunk | **~3–5s** after ASR |

---

## 📝 License

MIT License — see [LICENSE](LICENSE)

---

*Built for MLOps coursework · Chulalongkorn University · 2026*

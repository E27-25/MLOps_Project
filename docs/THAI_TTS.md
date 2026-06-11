# 🇹🇭 Thai TTS (JaiTTS / F5-TTS)

ZoonoMoE can speak its verdict in **Thai** instead of English, using
**JaiTTS** (Thonburian-TTS, an F5-TTS voice-cloning model) for stage 6 of the
pipeline. This is **opt-in** — the default build still uses English Kokoro, so
nothing changes unless you turn it on.

## ⚠️ Two coupled changes (why a flag isn't enough on its own)

TTS just *reads the LLM's text aloud*. So speaking Thai requires the **expert
LLM to produce Thai first** — otherwise JaiTTS would be handed English. Turning
on `TTS_LANG=th` therefore does **three** things automatically:

1. **Qwen answers in Thai** — a Thai output directive (`THAI_DIRECTIVE`) is
   appended to the expert/chat system prompt. The veterinary knowledge-base
   context stays English; Qwen3 is multilingual and translates as it answers.
2. **`clean_for_tts` preserves Thai** — the English path strips every non-ASCII
   character (which would erase Thai); the Thai path keeps the Thai Unicode
   block (`U+0E00–U+0E7F`) instead.
3. **Thai-aware sentence chunking** — Thai rarely uses `. ! ?`, so streaming
   chunks flush on newlines / length caps (`_iter_thai_chunks`).

## 🔧 Setup

```bash
# 1. extra deps (heavy, GPU recommended) + system ffmpeg
apt install -y ffmpeg
pip install -r backend/requirements-thai.txt

# 2. provide a short Thai reference voice clip (F5 clones its timbre)
#    a few seconds of clean Thai speech, mono .wav
mkdir -p backend/voices && cp my_thai_voice.wav backend/voices/th_ref.wav
```

## ▶️ Run

```bash
export TTS_LANG=th
export JAITTS_REF_VOICE=backend/voices/th_ref.wav   # required
export JAITTS_REF_TEXT="ข้อความถอดเสียงของคลิปอ้างอิง"  # optional, improves prosody
export USE_TRITON=false                               # Thai path is non-Triton
python backend/app.py
```

## ⚙️ Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TTS_LANG` | `en` | `th` = JaiTTS/F5 (Thai) · `en` = Kokoro (English) |
| `JAITTS_REF_VOICE` | `backend/voices/th_ref.wav` | Reference clip — repo ships a public Thai sample; swap for your own |
| `JAITTS_REF_TEXT` | _(empty)_ | Transcript of the reference clip (optional, better prosody) |
| `JAITTS_CHECKPOINT` | `hf://JTS-AI/JaiTTS-F5TTS/model.pt` | F5 checkpoint |
| `JAITTS_VOCAB` | `hf://JTS-AI/JaiTTS-F5TTS/vocab.txt` | vocab file |
| `JAITTS_VOCODER` | `vocos` | neural vocoder |
| `JAITTS_SPEED` | `0.8` | speech-rate multiplier — depends on the ref's pace (see below) |
| `JAITTS_NFE_STEP` | `32` | flow steps — ↓ = faster synth, slightly lower quality |
| `JAITTS_CFG` | `2.5` | classifier-free guidance strength |

## 🗣️ Speaking rate (avoid the "slow/dragged" voice)

F5-TTS **clones the reference clip's pace**, so the right `JAITTS_SPEED` depends
on how the reference was spoken. Same Thai sentence, two refs (target ≈ 4.5–6
chars/sec):

| `JAITTS_SPEED` | slow ref (pause-heavy) | normal-paced ref |
|---|---|---|
| 1.0 | 2.9 🐢 | **4.6 ✅** |
| 1.1 | 3.2 | **5.0 ✅** |
| 1.6 | **4.6 ✅** | 7.4 (too fast) |
| 2.0 | 5.8 | 9.3 (too fast) |

**The cleanest fix is a good reference clip** — a short (3–8 s), clearly-spoken
sample with minimal silence; then `JAITTS_SPEED=1.0–1.1` sounds natural. A
slow/pause-heavy ref needs ~1.6 to compensate. Tune with
`export JAITTS_SPEED=1.0`.

## ⭐ Use a **Thai** reference for natural prosody

JaiTTS is a **Thai** F5 model. If the reference clip is someone speaking
*another language* (e.g. English), the timbre is cloned but the **prosody and
pronunciation come out off** ("weird"), because the model forces Thai onto a
non-Thai speaking pattern. For the most natural result, use **5–8 s of clear
Thai speech** as the reference, with a matching `JAITTS_REF_TEXT`.

> Two musts for a good reference: (1) it is **Thai**, and (2) `JAITTS_REF_TEXT`
> **matches** what's actually said. A wrong/garbled ref text makes F5 collapse
> into mostly-silence garbage.

## 🔇 Silence handling

F5 pads each utterance with silence, so concatenated chunks otherwise sound
broken (long dead-air). `_tighten_wav_b64` trims leading/trailing silence and
caps internal gaps (default 0.3 s, set `JAITTS_MAX_GAP`) — this cut the demo
from 42 % → 26 % silence.

## 📝 Notes & limits

- **Reference voice is mandatory** — F5-TTS is voice-cloning; without
  `JAITTS_REF_VOICE` the app raises `FileNotFoundError` at startup.
- **Not via Triton** — the bundled `kokoro_tts` Triton model is English. With
  `USE_TRITON=true`, `TTS_LANG=th` is ignored (logged as a warning). Run the
  Thai path with `USE_TRITON=false`.
- **Latency** — F5 (flow-matching, `nfe_step=32`) is heavier than Kokoro and
  runs as a single locked instance, so per-sentence synthesis is slower. Tune
  `nfe_step` ↓ for speed or keep chunks short.
- Implementation: `backend/utils/thai_tts.py`; wiring in `backend/app.py`
  (`synth_audio_b64`, `clean_for_tts`, `iter_sentence_chunks`,
  `stream_expert_response`).

Model card: <https://huggingface.co/JTS-AI/JaiTTS-F5TTS> ·
Library: <https://github.com/biodatlab/thonburian-tts>

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
| `JAITTS_REF_VOICE` | `backend/voices/th_ref.wav` | **Required.** Reference voice clip to clone |
| `JAITTS_REF_TEXT` | _(empty)_ | Transcript of the reference clip (optional, better prosody) |
| `JAITTS_CHECKPOINT` | `hf://JTS-AI/JaiTTS-F5TTS/model.pt` | F5 checkpoint |
| `JAITTS_VOCAB` | `hf://JTS-AI/JaiTTS-F5TTS/vocab.txt` | vocab file |
| `JAITTS_VOCODER` | `vocos` | neural vocoder |

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

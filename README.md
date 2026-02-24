# Mini Speech-to-Speech — Mac M4

Fully local, real-time speech-to-speech pipeline.
No cloud APIs, no session timeouts, no internet required after first model download.
Runs entirely on your Mac using Apple Metal (MPS).

---

## Quick Start

```bash
bash setup.sh      # one-time install (5–10 min)
python3 app.py     # browser opens automatically at http://localhost:7860
```

---

## How It Works — Full Pipeline

```
Browser mic
    │
    │  webm blob (MediaRecorder)
    ▼
POST /upload
    │  ffmpeg: webm → wav 16kHz mono
    │  validate: size > 100 bytes
    │  store: _sessions[uuid] = {wav, tmp_dir}
    │  return: {session_id}
    ▼
GET /stream/<session_id>   ← Server-Sent Events (SSE)
    │
    ├─► Stage 1: Whisper ASR (CPU)
    │       wav → transcript text
    │       SSE: {type: "transcript", text, asr_time}
    │
    ├─► Stage 2: Qwen2.5 LLM (MPS / Apple Metal)
    │       TextIteratorStreamer → tokens arrive one by one
    │       iter_sentence_chunks() buffers tokens until [.!?] boundary
    │       yields complete sentence when min_words (8) reached
    │
    └─► Stage 3: Kokoro TTS (CPU)  ← runs per sentence, not per full response
            clean_for_tts() strips markdown + unicode
            sentence → WAV bytes (24kHz)
            SSE: {type: "chunk", index, b64, text}
            ← browser receives chunk 1, starts playing IMMEDIATELY
            ← chunk 2 still synthesizing on server while you listen
            ...
            SSE: {type: "done", full_text, timing}

Browser audio queue:
    chunk 1 plays → chunk 2 plays → chunk 3 plays ...
    (stop AI button drains queue and closes SSE stream)
```

### Why SSE instead of a single response

A single HTTP response would require waiting for:
`full LLM generation (all sentences) → full TTS (all chunks) → send → play`

With SSE, the first audio plays after only:
`ASR + LLM-first-sentence + TTS-chunk-1 ≈ 3–4s`

On a 6-sentence response the old approach took ~14s before any audio.
The new approach plays the first sentence at ~3–4s while the rest generates in background.

---

## Models

| Stage | Model | Params | Disk | Device | Typical latency |
|-------|-------|--------|------|--------|-----------------|
| ASR | openai/whisper-base | 74M | 145 MB | CPU | ~0.3s |
| LLM | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | 3.1 GB | MPS | ~2–4s total |
| TTS | hexgrad/Kokoro-82M | 82M | 330 MB | CPU | ~0.3s/sentence |
| **Total RAM** | | | **~2 GB** | | **first audio ~3–4s** |

Models download automatically on first run to:
- `~/.cache/whisper/` (Whisper)
- `~/.cache/huggingface/hub/` (Qwen + Kokoro)

### Device selection logic

```python
if torch.backends.mps.is_available():   # Apple Silicon Mac
    DEVICE = 'mps'
elif torch.cuda.is_available():          # NVIDIA GPU
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'                       # fallback
```

**Whisper is forced to CPU** regardless — Whisper has a known MPS bug on some
torch versions that produces garbage transcription output. CPU is ~0.3s for
a 3s clip on M4, which is fast enough.

**LLM uses MPS** with `torch_dtype=torch.float16` and `device_map='auto'`.
float16 halves memory vs float32 (1.5 GB instead of 3 GB).

**Kokoro TTS uses CPU** — it calls `espeak-ng` internally for phonemization
which is CPU-only.

---

## Technical Details

### 1. Token Streaming + Sentence Chunking

The LLM runs in a background thread using `TextIteratorStreamer`.
Tokens appear in the streamer queue as they are generated.
`iter_sentence_chunks()` consumes tokens and detects sentence boundaries:

```python
_BOUNDARY = re.compile(r'[.!?](?:\s|$)')

def iter_sentence_chunks(token_iter, min_words=8):
    buf = ''
    for token in token_iter:
        buf += token
        if _BOUNDARY.search(buf):
            m = None
            for m in _BOUNDARY.finditer(buf):
                pass                          # find LAST boundary in buf
            if m and len(buf[:m.end()].split()) >= min_words:
                chunk = buf[:m.end()].strip() # yield complete sentence
                buf   = buf[m.end():].strip() # keep remainder in buffer
                yield chunk
    if buf.strip():
        yield buf.strip()                     # flush final partial sentence
```

`min_words=8` prevents sending 2-word fragments like "Sure!" to TTS,
which would produce very choppy audio with audible gaps between clips.

### 2. TTS Sanitizer (`clean_for_tts`)

Kokoro's local phonemizer (`misaki` / `espeak-ng`) fails on certain characters
and falls back to a remote CAS (Character-to-Audio Service).
If that remote call times out you get:

```
Error: CAS service error: ReqwestMiddleware: Request failed after 5 retries
```

**Root cause:** Qwen loves to output em-dashes (`—`, Unicode U+2014)
even when instructed not to. `espeak-ng` cannot phonemize them.

The sanitizer runs on every chunk before Kokoro sees it:

```python
# Strip markdown (LLMs produce this despite system prompt instructions)
text = re.sub(r'```[\s\S]*?```', '', text)      # fenced code blocks → removed
text = re.sub(r'`[^`]*`', '', text)              # inline `code` → removed
text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** → bold
text = re.sub(r'\*([^*]+)\*', r'\1', text)       # *italic* → italic
text = re.sub(r'#{1,6}\s+', '', text)            # ## Header → removed
text = re.sub(r'^\s*[-*]\s+', '', text, ...)     # - bullets → removed
text = re.sub(r'^\s*\d+\.\s+', '', text, ...)    # 1. lists → removed
text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [link](url) → link

# Unicode punctuation → ASCII
'\u2014' → ', '    # em-dash  —  (most common Kokoro crash trigger)
'\u2013' → ' to '  # en-dash  –
'\u2026' → '...'   # ellipsis …
'\u2018' → "'"     # left single quote
'\u2019' → "'"     # right single quote / apostrophe
'\u201c' → '"'     # left double quote
'\u201d' → '"'     # right double quote
'\u00b2' → ' squared'
'\u00b0' → ' degrees'

# Abbreviation expansion (prevents phonemizer lookup failures)
PSI → P.S.I.   CO2 → C O 2   DNA → D.N.A.
AI  → A.I.     API → A.P.I.  LLM → L.L.M.

# Final: strip all remaining non-ASCII
re.sub(r'[^\x00-\x7F]+', ' ', text)
```

### 3. System Prompt Engineering (`_SPOKEN_RULE`)

Every persona appends `_SPOKEN_RULE` to prevent markdown and enforce
spoken-audio-appropriate responses:

```python
_SPOKEN_RULE = (
    ' This response will be read aloud by text-to-speech, so follow these rules strictly: '
    'No markdown whatsoever. No asterisks, backticks, hash signs, bullet points, '
    'numbered lists, or code blocks. No URLs. Speak naturally as if talking. '
    'Stop after a maximum of 4 sentences no matter what.'
)
```

Without this, Qwen defaults to markdown tutorial format — numbered steps,
code blocks, bold headers — which sounds terrible when spoken and causes
Kokoro to crash on backticks and asterisks.

`max_new_tokens=220` (~55 words at ~4 tokens/word) is enough for 4 natural
spoken sentences. 150 (the original value) caused mid-sentence cutoffs.

### 4. Session Lifecycle (The `finally` Bug Fix)

Two routes handle one request to allow SSE streaming:

```
POST /upload:
  1. Receive webm blob from browser
  2. Validate: len(data) > 100 bytes  ← empty blob guard
  3. Write to unique tmp_dir (tempfile.mkdtemp())
  4. ffmpeg: input.webm → output.wav 16kHz mono
  5. _sessions[uuid] = {'wav': path, 'tmp_dir': path}
  6. Return {session_id: uuid}
  NOTE: tmp_dir is NOT cleaned up here on success
        Only cleaned on exception

GET /stream/<session_id>:
  1. Pop session dict from _sessions
  2. Run stream_pipeline(wav) under pipeline_lock
  3. Yield SSE events as they arrive
  4. finally: shutil.rmtree(tmp_dir)  ← cleanup owned HERE
```

**Why two routes?** Flask cannot simultaneously receive an upload body
and stream a response in a single request. The separation also lets the
browser open the SSE connection immediately after the upload completes.

**The `finally` bug (now fixed):** The original code had `shutil.rmtree`
in the `/upload` `finally` block, which runs even on success. This deleted
the WAV file before `/stream` could read it, producing:

```
RuntimeError: Failed to load audio: Error opening input file output.wav.
Error opening input files: No such file or directory
```

Fix: cleanup responsibility transferred entirely to the `/stream` route's
`finally` block, which runs only after the full pipeline completes.

### 5. Audio Visualizer

Uses FFT frequency data (`getByteFrequencyData`, 256-point FFT = 128 bins).

**Why the old version was skewed left:** all 128 bins were drawn across the
canvas width. Human voice sits in ~80 Hz–3 kHz, which is roughly the bottom
60% of FFT bins at 44.1 kHz. The top 40% bins are always near zero,
creating empty bars on the right side — making the active bars appear
crowded to the left.

**Fix:** only draw the bottom 60% of bins (`usedBins = Math.floor(bufLen * 0.6)`).
Each bar is mirrored above and below the centre line for a symmetric look.

### 6. Conversation Memory

Last 12 messages (6 turns) are appended to every LLM prompt as chat history.
Stored in `state['history']` (server-side, in-memory).
Resets on: persona change, "Clear history" button, app restart.

### 7. Concurrent Request Protection

`state['pipeline_lock']` is a `threading.Lock()` that prevents two
simultaneous inference requests from racing on the shared LLM and TTS models.
If a second request arrives while inference is running, it blocks until the
first completes. Flask runs with `threaded=True` so the upload/SSE routes
themselves are non-blocking.

---

## Personas

| Key | Name | Voice | Style |
|-----|------|-------|-------|
| `wildlife_expert` | Dr. Maya Chen | af_heart | Calm, scientific authority |
| `friendly_teacher` | Professor Sam | am_michael | Warm analogies |
| `casual_chat` | Alex | af_bella | Natural conversation |
| `astronaut` | Captain Alex | am_adam | Urgent, terse |

### Available Voices (Kokoro)

```
American English female : af_heart  af_bella  af_nicole  af_sky
American English male   : am_adam   am_michael
British English female  : bf_emma   bf_isabella
British English male    : bm_george bm_lewis
```

### Adding a Custom Persona

In `app.py`, add to the `PERSONAS` dict:

```python
'chef_marco': {
    'name':   'Chef Marco',
    'system': ('You are Chef Marco, an enthusiastic Italian chef. '
               'Answer cooking questions with passion and warmth.' + _SPOKEN_RULE),
    'voice':  'am_michael',
},
```

Then add to the JavaScript `PERSONAS` object in the HTML section:

```javascript
chef_marco: { label: 'Chef Marco', voice: 'am_michael' },
```

---

## Upgrade Options

### Faster ASR: mlx-whisper (3–4× faster on M-series)

`mlx-whisper` uses Apple's MLX framework, which is optimised for the
Apple Neural Engine. For a 3s clip: ~0.08s vs ~0.3s for openai-whisper.

```bash
pip install mlx-whisper
```

Replace lines ~35–38 in `app.py`:

```python
# Remove:
import whisper
asr_model = whisper.load_model('base', device='cpu')

# Add:
import mlx_whisper
def transcribe(audio_path: str) -> str:
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo='mlx-community/whisper-base-mlx'
    )
    return result['text'].strip()
```

Also remove the `def transcribe` function below (since you just defined it above).

### Smarter LLM

Change `LLM_MODEL` on line ~43. All models fit comfortably in 16 GB unified memory:

| Model | Download | RAM (float16) | Quality | First-token latency |
|-------|----------|---------------|---------|---------------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 1 GB | ~0.5 GB | Basic | Very fast |
| `Qwen/Qwen2.5-1.5B-Instruct` | 3.1 GB | ~1.5 GB | Good | Fast ← **default** |
| `Qwen/Qwen2.5-3B-Instruct` | 6.2 GB | ~3 GB | Better | Medium |
| `Qwen/Qwen2.5-7B-Instruct` | 15 GB | ~7.5 GB | Great | Slower |

---

## Troubleshooting

### SSL certificate error on model download
```
ssl.SSLCertVerificationError: certificate verify failed: self-signed certificate
```
Fix:
```bash
# Option A — official Python installer
/Applications/Python\ 3.12/Install\ Certificates.command

# Option B — certifi
pip install certifi
SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") python3 app.py
```
`setup.sh` does this automatically.

### Microphone blocked in browser
```
Mic error: Permission denied
```
Go to: **System Settings → Privacy & Security → Microphone** → enable your browser.

### `CAS service error / ReqwestMiddleware` from Kokoro
The LLM put an un-phonemizable character in its response (almost always an em-dash `—`).
`clean_for_tts()` handles this automatically. If it still occurs, check the terminal
for which sentence triggered it and add that character to the replacement table in `app.py`.

### `No such file or directory` for wav file
```
RuntimeError: Failed to load audio: Error opening input file output.wav
```
This was the `finally`-block cleanup bug. Fixed in current version.
If you still see it, make sure you are running the latest `app.py`.

### LLM output is cut off mid-sentence
`max_new_tokens` is too low. Increase it in `app.py`:
```python
max_new_tokens=220,   # current default
max_new_tokens=320,   # increase if responses still truncate
```

### MPS crash / LLM gives garbage output
```python
# In app.py, change:
device_map='auto'
# To:
device_map='cpu'    # slower but completely stable
```

### Port already in use
```bash
lsof -i :7860        # find what's using it
# Then in app.py:
PORT = 8080          # use any free port
```

### App slows down after many turns
Conversation history accumulates. Click **Clear history** in the UI, or restart `app.py`.
History resets automatically when you switch persona.

---

## File Structure

```
.
├── app.py       Flask server + full pipeline (883 lines)
├── setup.sh     One-time dependency installer
└── README.md    This file
```

---

## Dependency Versions

| Package | Min version | Purpose |
|---------|-------------|---------|
| torch | 2.1 | MPS backend for Apple Silicon |
| transformers | 4.40 | LLM + TextIteratorStreamer |
| openai-whisper | any | ASR |
| kokoro | 0.9.4 | TTS |
| soundfile | any | WAV write |
| flask | any | Web server + SSE |
| numpy | <2.2 | Audio arrays (2.2+ breaks whisper) |
| ffmpeg | any | Audio format conversion (system) |
"""
Mini Speech-to-Speech - Mac M4  (v2: SSE streaming TTS)
=========================================================
Run:  python3 app.py
Open: http://localhost:7860

Pipeline:
  Mic -> Whisper -> LLM (token streaming) -> Kokoro TTS per sentence -> SSE -> Browser
  First audio chunk plays ~(ASR + LLM-first-sentence + TTS-first-chunk) seconds.
  Previously it waited for ALL sentences. Now each plays as soon as it is ready.
"""

import os, re, json, base64, tempfile, subprocess, threading, time, queue
from flask import Flask, request, jsonify, Response, stream_with_context
import numpy as np

# ── Device ────────────────────────────────────────────────────────
import torch

if torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("Device: Apple MPS (Metal)")
elif torch.cuda.is_available():
    DEVICE = 'cuda'
    print("Device: CUDA")
else:
    DEVICE = 'cpu'
    print("Device: CPU")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── Load models ───────────────────────────────────────────────────
print("\nLoading models (first run downloads ~2 GB)...")

print("  [1/3] Whisper-base...")
import whisper
asr_model = whisper.load_model('base', device='cpu')
print("        done")

print("  [2/3] Qwen2.5-1.5B-Instruct (with TextIteratorStreamer)...")
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

LLM_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
)
llm.eval()
print("        done")

print("  [3/3] Kokoro TTS...")
from kokoro import KPipeline
import soundfile as sf
tts_pipe = KPipeline(lang_code='a')
print("        done\n")
print("All models loaded!\n")

# ── Personas ──────────────────────────────────────────────────────
_SPOKEN_RULE = (
    ' This response will be read aloud by text-to-speech, so follow these rules strictly: '
    'No markdown whatsoever. No asterisks, backticks, hash signs, bullet points, '
    'numbered lists, or code blocks. No URLs. Speak naturally as if talking. '
    'Stop after a maximum of 4 sentences no matter what.'
)

PERSONAS = {
    'wildlife_expert': {
        'name':   'Dr. Maya Chen',
        'system': ('You are Dr. Maya Chen, a world-renowned wildlife biologist. '
                   'Answer with calm authority and scientific passion.' + _SPOKEN_RULE),
        'voice':  'af_heart',
    },
    'friendly_teacher': {
        'name':   'Professor Sam',
        'system': ('You are Professor Sam, a warm teacher who explains things with one vivid analogy. '
                   'Give the analogy and then explain it conversationally.' + _SPOKEN_RULE),
        'voice':  'am_michael',
    },
    'casual_chat': {
        'name':   'Alex',
        'system': ('You are Alex, a warm and friendly conversationalist. '
                   'Be natural and engaging. Maximum 2 sentences.' + _SPOKEN_RULE),
        'voice':  'af_bella',
    },
    'astronaut': {
        'name':   'Captain Alex',
        'system': ('You are Captain Alex, an astronaut on Mars with a reactor meltdown. '
                   'Calm but urgent. Write all numbers and abbreviations as full words. '
                   'Maximum 3 short sentences.' + _SPOKEN_RULE),
        'voice':  'am_adam',
    },
}

state = {
    'persona':       'wildlife_expert',
    'history':       [],
    'pipeline_lock': threading.Lock(),
}

# ── TTS sanitizer ─────────────────────────────────────────────────
_ABBREV = {
    r'\bPSI\b':  'P.S.I.', r'\bCO2\b':  'C O 2',  r'\bO2\b':   'O 2',
    r'\bDNA\b':  'D.N.A.', r'\bRNA\b':  'R.N.A.', r'\bAI\b':   'A.I.',
    r'\bLLM\b':  'L.L.M.', r'\bAPI\b':  'A.P.I.',
    r'\bNOW\b':  'now',    r'\bSTOP\b': 'stop',   r'\bALERT\b':'alert',
}

def clean_for_tts(text: str) -> str:
    # Strip markdown the LLM produces despite instructions
    text = re.sub(r'```[\s\S]*?```', '', text)          # fenced code blocks
    text = re.sub(r'`[^`]*`', '', text)                    # inline code
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)   # **bold**
    text = re.sub(r'\*([^*]+)\*',   r'\1', text)        # *italic*
    text = re.sub(r'#{1,6}\s+', '', text)                 # ## headers
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)    # bullets
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE) # 1. lists
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)       # [link](url)
    # Unicode -> ASCII
    text = text.replace('\u2014', ', ').replace('\u2013', ' to ')
    text = text.replace('\u2026', '...').replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u00b2', ' squared').replace('\u00b0', ' degrees')
    text = text.replace('\u03c0', 'pi').replace('\u221e', 'infinity')
    for pat, rep in _ABBREV.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'  +', ' ', text).strip()

# ── Sentence accumulator ──────────────────────────────────────────
# Detects sentence boundaries in a token stream.
# Yields complete sentence groups (min_words words) as they accumulate.
_BOUNDARY = re.compile(r'[.!?](?:\s|$)')

def iter_sentence_chunks(token_iter, min_words: int = 8):
    """
    Consumes a token iterator, yields text chunks at sentence boundaries.
    min_words: don't yield until we have at least this many words buffered.
    """
    buf = ''
    for token in token_iter:
        buf += token
        # Check if buf ends with a sentence boundary
        if _BOUNDARY.search(buf):
            # Split at last boundary
            m = None
            for m in _BOUNDARY.finditer(buf):
                pass
            if m and len(buf[:m.end()].split()) >= min_words:
                chunk = buf[:m.end()].strip()
                buf   = buf[m.end():].strip()
                yield chunk
    if buf.strip():
        yield buf.strip()

# ── Pipeline ──────────────────────────────────────────────────────
def transcribe(audio_path: str) -> str:
    result = asr_model.transcribe(audio_path, language='en', fp16=False)
    return result['text'].strip()

def synthesize_chunk(text: str) -> bytes:
    """Synthesize cleaned text chunk -> WAV bytes."""
    voice = PERSONAS[state['persona']]['voice']
    text  = clean_for_tts(text)
    if not text:
        return b''
    samples = []
    for _, _, audio in tts_pipe(text, voice=voice, speed=1.0):
        samples.append(audio)
    if not samples:
        return b''
    audio_np = np.concatenate(samples)
    tmp = tempfile.mktemp(suffix='.wav')
    sf.write(tmp, audio_np, 24000)
    with open(tmp, 'rb') as f:
        data = f.read()
    os.unlink(tmp)
    return data

def stream_pipeline(audio_path: str):
    """
    Generator that yields SSE events:

      data: {"type":"transcript","text":"..."}      <- ASR done
      data: {"type":"chunk","index":0,"b64":"...","text":"sent1"}  <- each TTS chunk
      data: {"type":"done","full_text":"...","timing":{...}}       <- all done

    The LLM runs token-by-token via TextIteratorStreamer.
    Each sentence is sent to TTS as soon as the LLM finishes it.
    The browser plays chunk 0 while chunk 1 is still being generated.
    """
    t0 = time.perf_counter()

    # ── Stage 1: ASR ────────────────────────────────────────────
    user_text = transcribe(audio_path)
    t1 = time.perf_counter()
    yield f'data: {json.dumps({"type":"transcript","text":user_text,"asr":round(t1-t0,2)})}\n\n'

    # ── Stage 2: LLM streaming ──────────────────────────────────
    persona  = PERSONAS[state['persona']]
    messages = [{'role': 'system', 'content': persona['system']}]
    messages += state['history'][-12:]
    messages.append({'role': 'user', 'content': user_text})

    prompt  = llm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = llm_tok([prompt], return_tensors='pt').to(DEVICE)

    streamer = TextIteratorStreamer(
        llm_tok, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=220,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=llm_tok.eos_token_id,
    )

    # LLM runs in background thread so we can consume tokens here
    gen_thread = threading.Thread(target=llm.generate, kwargs=gen_kwargs, daemon=True)
    gen_thread.start()

    t2_start = time.perf_counter()

    # ── Stage 3: TTS per sentence as tokens arrive ───────────────
    full_text  = ''
    chunk_idx  = 0
    tts_times  = []
    first_tts  = None

    for sentence in iter_sentence_chunks(streamer, min_words=8):
        full_text += (' ' if full_text else '') + sentence
        tc0 = time.perf_counter()
        wav = synthesize_chunk(sentence)
        tc1 = time.perf_counter()
        tts_times.append(round(tc1 - tc0, 2))

        if first_tts is None:
            first_tts = round(tc1 - t0, 2)

        if wav:
            b64 = base64.b64encode(wav).decode()
            yield f'data: {json.dumps({"type":"chunk","index":chunk_idx,"b64":b64,"text":sentence})}\n\n'
            chunk_idx += 1

    gen_thread.join()
    t3 = time.perf_counter()

    # Update history
    state['history'].append({'role': 'user',      'content': user_text})
    state['history'].append({'role': 'assistant', 'content': full_text.strip()})
    if len(state['history']) > 20:
        state['history'] = state['history'][-20:]

    timing = {
        'asr':        round(t1 - t0, 2),
        'llm':        round(t3 - t2_start, 2),
        'tts':        round(sum(tts_times), 2),
        'total':      round(t3 - t0, 2),
        'first_audio':first_tts,
        'tts_chunks': tts_times,
    }

    print(f"\nYou  : {user_text}")
    print(f"Agent: {full_text.strip()}")
    print(f"Times: ASR={timing['asr']}s  LLM={timing['llm']}s  "
          f"TTS={timing['tts']}s  total={timing['total']}s  "
          f"first_audio={timing['first_audio']}s")

    yield f'data: {json.dumps({"type":"done","full_text":full_text.strip(),"timing":timing})}\n\n'

# ── Flask ─────────────────────────────────────────────────────────
flask_app = Flask(__name__)
flask_app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Speech-to-Speech</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Courier New', monospace;
  background: #1e1e2e;
  color: #cdd6f4;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 30px 16px;
}
h1 { color: #cba6f7; margin-bottom: 6px; font-size: 22px; }
.subtitle { color: #6c7086; font-size: 13px; margin-bottom: 24px; }

.card {
  background: #313244;
  border-radius: 12px;
  padding: 20px;
  width: 100%;
  max-width: 560px;
  margin-bottom: 16px;
}
.card-title {
  color: #89b4fa;
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
  font-weight: bold;
  margin-bottom: 12px;
}

/* Persona */
.persona-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.persona-btn {
  padding: 10px 12px;
  border: 2px solid #45475a;
  border-radius: 8px;
  background: #1e1e2e;
  color: #cdd6f4;
  cursor: pointer;
  font-family: monospace;
  font-size: 13px;
  transition: all 0.15s;
  text-align: left;
}
.persona-btn:hover  { border-color: #cba6f7; }
.persona-btn.active { border-color: #a6e3a1; background: #1a2e22; color: #a6e3a1; }
.persona-name  { font-weight: bold; display: block; margin-bottom: 2px; }
.persona-voice { color: #6c7086; font-size: 11px; }

/* Visualizer — FFT bars, centered by symmetric yscale */
#visualizer {
  width: 100%;
  height: 64px;
  border-radius: 8px;
  background: #1e1e2e;
  display: none;
  margin-bottom: 12px;
}

/* Buttons row */
.controls { display: flex; gap: 8px; justify-content: center; margin-bottom: 12px; flex-wrap: wrap; }
.btn {
  padding: 11px 22px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: bold;
  cursor: pointer;
  font-family: monospace;
  transition: opacity 0.15s, transform 0.1s;
}
.btn:active   { transform: scale(0.96); }
.btn:disabled { opacity: 0.3; cursor: not-allowed; }
.btn-rec      { background: #a6e3a1; color: #1e1e2e; }
.btn-stop-rec { background: #f38ba8; color: #1e1e2e; }
/* Stop AI always visible but dimmed when not playing */
.btn-stop-ai  { background: #fab387; color: #1e1e2e; opacity: 0.35; pointer-events: none; }
.btn-stop-ai.active { opacity: 1; pointer-events: auto; }

/* Status */
#status {
  text-align: center;
  color: #89dceb;
  font-size: 13px;
  margin-bottom: 8px;
  min-height: 18px;
}

/* Timing */
#timing {
  display: none;
  font-size: 11px;
  text-align: center;
  margin-bottom: 10px;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: center;
}
#timing span { padding: 2px 8px; border-radius: 4px; background: #1e1e2e; }
.t-asr   { color: #89dceb; }
.t-llm   { color: #cba6f7; }
.t-tts   { color: #a6e3a1; }
.t-first { color: #f9e2af; }
.t-total { color: #f9e2af; font-weight: bold; }

/* Spinner */
#spinner {
  display: none;
  width: 20px; height: 20px;
  border: 3px solid #45475a;
  border-top-color: #cba6f7;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  margin: 0 auto 10px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Log */
#log {
  min-height: 80px;
  max-height: 380px;
  overflow-y: auto;
  font-size: 13px;
  line-height: 1.8;
}
.msg { margin-bottom: 8px; }
.msg-you   { color: #a6e3a1; }
.msg-agent { color: #89b4fa; }
.msg-err   { color: #f38ba8; }
.msg-info  { color: #6c7086; }

.clear-btn {
  margin-top: 10px;
  background: none;
  border: 1px solid #45475a;
  color: #6c7086;
  border-radius: 6px;
  padding: 4px 12px;
  font-size: 12px;
  cursor: pointer;
  font-family: monospace;
}
.clear-btn:hover { border-color: #f38ba8; color: #f38ba8; }
</style>
</head>
<body>

<h1>Speech-to-Speech</h1>
<div class="subtitle">Whisper + Qwen2.5 + Kokoro &mdash; Mac M4</div>

<!-- Persona -->
<div class="card">
  <div class="card-title">Persona</div>
  <div class="persona-grid" id="personaGrid"></div>
</div>

<!-- Recorder -->
<div class="card">
  <div class="card-title">Recorder</div>
  <canvas id="visualizer"></canvas>
  <div id="spinner"></div>
  <div id="status">Click Record and allow microphone access</div>
  <div id="timing">
    <span id="t-asr"   class="t-asr"></span>
    <span id="t-llm"   class="t-llm"></span>
    <span id="t-tts"   class="t-tts"></span>
    <span id="t-first" class="t-first"></span>
    <span id="t-total" class="t-total"></span>
  </div>
  <div class="controls">
    <button class="btn btn-rec"      id="recBtn"     onclick="startRec()">&#9679; Record</button>
    <button class="btn btn-stop-rec" id="stopRecBtn" onclick="stopRec()" disabled>&#9632; Stop &amp; Send</button>
    <button class="btn btn-stop-ai"  id="stopAiBtn"  onclick="stopAI()">&#9646;&#9646; Stop AI</button>
  </div>
</div>

<!-- Log -->
<div class="card">
  <div class="card-title">Conversation</div>
  <div id="log"><div class="msg msg-info">Waiting for input...</div></div>
  <button class="clear-btn" onclick="clearLog()">Clear history</button>
</div>

<script>
// ── Personas ─────────────────────────────────────────────────────
const PERSONAS = {
  wildlife_expert:  { label: 'Wildlife Expert', voice: 'af_heart'   },
  friendly_teacher: { label: 'Friendly Teacher', voice: 'am_michael' },
  casual_chat:      { label: 'Casual Chat',      voice: 'af_bella'   },
  astronaut:        { label: 'Astronaut',         voice: 'am_adam'    },
};
let currentPersona = 'wildlife_expert';

const grid = document.getElementById('personaGrid');
Object.entries(PERSONAS).forEach(([key, p]) => {
  const btn = document.createElement('button');
  btn.className = 'persona-btn' + (key === currentPersona ? ' active' : '');
  btn.innerHTML = `<span class="persona-name">${p.label}</span>
                   <span class="persona-voice">${p.voice}</span>`;
  btn.onclick = () => {
    document.querySelectorAll('.persona-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentPersona = key;
    fetch('/set_persona', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ persona: key })
    });
  };
  grid.appendChild(btn);
});

// ── Visualizer — FFT bars symmetric around centre ─────────────────
// Old FFT bars were better looking. Fix: mirror top+bottom around mid.
// Each bar drawn UP and DOWN from centre = symmetric, not left-skewed.
let animCtx   = null;
let animFrame = null;

function startVisualizer(stream) {
  const canvas  = document.getElementById('visualizer');
  const ctx     = canvas.getContext('2d');
  canvas.style.display = 'block';
  canvas.width  = canvas.offsetWidth;
  canvas.height = 64;

  const actx     = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = actx.createAnalyser();
  analyser.fftSize               = 256;
  analyser.smoothingTimeConstant = 0.75;
  actx.createMediaStreamSource(stream).connect(analyser);

  const bufLen = analyser.frequencyBinCount;  // 128 bins
  const data   = new Uint8Array(bufLen);
  const W = canvas.width;
  const H = canvas.height;
  const mid = H / 2;
  animCtx = actx;

  function draw() {
    animFrame = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(data);

    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(0, 0, W, H);

    // Use only the lower ~60% of bins (human voice range)
    // This removes the empty high-freq bins that caused left-skew
    const usedBins = Math.floor(bufLen * 0.6);
    const barW     = W / usedBins;

    for (let i = 0; i < usedBins; i++) {
      const norm = data[i] / 255;                  // 0..1
      const h    = norm * mid * 0.9;               // half-height
      const x    = i * barW;
      const hue  = 260 + norm * 60;                // purple -> pink

      ctx.fillStyle = `hsl(${hue},80%,65%)`;
      // Draw bar both above AND below centre line = symmetric
      ctx.fillRect(x, mid - h, barW - 1, h);      // top half
      ctx.fillRect(x, mid,     barW - 1, h);      // bottom half (mirror)
    }
  }
  draw();
}

function stopVisualizer() {
  if (animFrame) cancelAnimationFrame(animFrame);
  if (animCtx)   animCtx.close();
  animCtx = animFrame = null;
  document.getElementById('visualizer').style.display = 'none';
}

// ── Audio chunk queue (SSE streams chunks as they arrive) ─────────
let audioQueue   = [];
let isPlayingAI  = false;
let currentAudio = null;
let activeSSE    = null;   // EventSource reference so we can abort

function setStopAIActive(on) {
  const btn = document.getElementById('stopAiBtn');
  if (on) btn.classList.add('active');
  else    btn.classList.remove('active');
}

function stopAI() {
  // Stop SSE stream
  if (activeSSE) { activeSSE.close(); activeSSE = null; }
  // Stop current audio
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = '';
    currentAudio = null;
  }
  audioQueue  = [];
  isPlayingAI = false;
  setStopAIActive(false);
  document.getElementById('recBtn').disabled     = false;
  document.getElementById('stopRecBtn').disabled = true;
  setStatus('Stopped. Click Record for next turn.');
}

function enqueueChunk(b64) {
  audioQueue.push(b64);
  if (!isPlayingAI) playNext();
}

function playNext() {
  if (audioQueue.length === 0) {
    isPlayingAI = false;
    setStopAIActive(false);
    return;
  }
  isPlayingAI = true;
  setStopAIActive(true);

  const audio = new Audio('data:audio/wav;base64,' + audioQueue.shift());
  currentAudio = audio;
  audio.onended = () => { currentAudio = null; playNext(); };
  audio.onerror = () => { currentAudio = null; playNext(); };
  audio.play().catch(() => { currentAudio = null; playNext(); });
}

// ── Helpers ───────────────────────────────────────────────────────
function setStatus(msg) { document.getElementById('status').textContent = msg; }

function showTiming(t) {
  const el = document.getElementById('timing');
  el.style.display = 'flex';
  document.getElementById('t-asr').textContent   = 'ASR '   + t.asr   + 's';
  document.getElementById('t-llm').textContent   = 'LLM '   + t.llm   + 's';
  document.getElementById('t-tts').textContent   = 'TTS '   + t.tts   + 's x' + t.tts_chunks.length;
  document.getElementById('t-first').textContent = 'First '  + t.first_audio + 's';
  document.getElementById('t-total').textContent = 'Total '  + t.total + 's';
}

function addMsg(cls, label, text) {
  const log = document.getElementById('log');
  if (log.querySelector('.msg-info')) log.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'msg msg-' + cls;
  div.innerHTML = '<b>' + label + ':</b> ' + text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function clearLog() {
  document.getElementById('log').innerHTML = '<div class="msg msg-info">History cleared</div>';
  document.getElementById('timing').style.display = 'none';
  fetch('/clear_history', { method: 'POST' });
}

// ── Recording ─────────────────────────────────────────────────────
let mediaRecorder = null;
let recChunks     = [];

async function startRec() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];  // always reset before new recording
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                 ? 'audio/webm;codecs=opus' : 'audio/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    mediaRecorder.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };
    mediaRecorder.start(100);
    startVisualizer(stream);
    document.getElementById('recBtn').disabled      = true;
    document.getElementById('stopRecBtn').disabled  = false;
    document.getElementById('timing').style.display = 'none';
    setStatus('Recording... speak now');
  } catch(err) {
    setStatus('Mic error: ' + err.message);
  }
}

function stopRec() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  document.getElementById('stopRecBtn').disabled = true;
  stopVisualizer();
  stopAI();  // stop any previous playback
  setStatus('Transcribing...');
  document.getElementById('spinner').style.display = 'block';

  mediaRecorder.onstop = async () => {
    mediaRecorder.stream.getTracks().forEach(t => t.stop());

    // Guard: make sure we actually recorded something
    const totalBytes = recChunks.reduce((s, c) => s + c.size, 0);
    if (totalBytes < 100) {
      document.getElementById('spinner').style.display = 'none';
      setStatus('No audio captured. Try again.');
      document.getElementById('recBtn').disabled     = false;
      document.getElementById('stopRecBtn').disabled = true;
      return;
    }

    // Upload audio
    const blob = new Blob(recChunks, { type: 'audio/webm' });
    const form = new FormData();
    form.append('audio', blob, 'recording.webm');

    let agentName = PERSONAS[currentPersona]
                    ? (currentPersona === 'wildlife_expert'  ? 'Dr. Maya Chen'   :
                       currentPersona === 'friendly_teacher' ? 'Professor Sam'   :
                       currentPersona === 'casual_chat'      ? 'Alex'            :
                       'Captain Alex')
                    : 'Agent';

    // POST audio, get session ID back, then open SSE stream
    let sessionId;
    try {
      const r = await fetch('/upload', { method: 'POST', body: form });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      sessionId = d.session_id;
    } catch(err) {
      document.getElementById('spinner').style.display = 'none';
      addMsg('err', 'Error', err.message);
      setStatus('Upload error');
      document.getElementById('recBtn').disabled = false;
      return;
    }

    // Open SSE stream
    const sse = new EventSource('/stream/' + sessionId);
    activeSSE  = sse;
    let agentDiv = null;  // live-updated agent message div
    let fullText = '';

    sse.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.type === 'transcript') {
        document.getElementById('spinner').style.display = 'none';
        addMsg('you', 'You', msg.text);
        setStatus('Generating response...');

      } else if (msg.type === 'chunk') {
        // First chunk — create agent message div and start playing
        if (!agentDiv) {
          if (log.querySelector('.msg-info')) log.innerHTML = '';
          agentDiv = document.createElement('div');
          agentDiv.className = 'msg msg-agent';
          agentDiv.innerHTML = '<b>' + agentName + ':</b> ';
          document.getElementById('log').appendChild(agentDiv);
          setStatus('Playing response...');
        }
        // Append this sentence to the live text
        fullText += (fullText ? ' ' : '') + msg.text;
        agentDiv.innerHTML = '<b>' + agentName + ':</b> ' + fullText;
        document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
        // Play audio
        enqueueChunk(msg.b64);

      } else if (msg.type === 'done') {
        showTiming(msg.timing);
        if (!isPlayingAI) setStatus('Done! Click Record for next turn.');
        sse.close();
        activeSSE = null;
        document.getElementById('recBtn').disabled     = false;
        document.getElementById('stopRecBtn').disabled = true;
      }
    };

    sse.onerror = (e) => {
      document.getElementById('spinner').style.display = 'none';
      sse.close();
      activeSSE = null;
      document.getElementById('recBtn').disabled     = false;
      document.getElementById('stopRecBtn').disabled = true;
      // Only show error if we never got any text (otherwise it's normal SSE close)
      if (!fullText) {
        addMsg('err', 'Error', 'Stream failed. Check terminal for details.');
        setStatus('Error — try again');
      } else {
        setStatus('Done! Click Record for next turn.');
      }
    };
  };

  mediaRecorder.stop();
}
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────
import uuid

# Store uploaded audio paths keyed by session_id
_sessions = {}
_sessions_lock = threading.Lock()

@flask_app.route('/')
def index():
    return Response(HTML_PAGE, mimetype='text/html')

@flask_app.route('/set_persona', methods=['POST'])
def set_persona():
    data = request.get_json()
    key  = data.get('persona', 'wildlife_expert')
    if key in PERSONAS:
        state['persona'] = key
        state['history'] = []
        print(f"Persona: {PERSONAS[key]['name']}")
    return jsonify({'ok': True})

@flask_app.route('/clear_history', methods=['POST'])
def clear_history():
    state['history'] = []
    return jsonify({'ok': True})

@flask_app.route('/upload', methods=['POST'])
def upload():
    """Receive audio, convert to WAV, store for SSE stream."""
    if 'audio' not in request.files:
        return jsonify({'error': 'no audio'}), 400

    # Use unique temp dir per request to avoid filename collisions between sessions
    tmp_dir = tempfile.mkdtemp()
    webm = os.path.join(tmp_dir, 'input.webm')
    wav  = os.path.join(tmp_dir, 'output.wav')
    try:
        data_bytes = request.files['audio'].read()
        if len(data_bytes) < 100:
            return jsonify({'error': 'Audio too short or empty. Hold the button while speaking.'})
        with open(webm, 'wb') as f:
            f.write(data_bytes)
        print(f'Received audio: {len(data_bytes)/1024:.1f} KB')
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', webm, '-ar', '16000', '-ac', '1', wav],
            capture_output=True
        )
        if r.returncode != 0:
            # Skip ffmpeg version/config header (first ~8 lines) to show actual error
            stderr_lines = r.stderr.decode(errors='replace').splitlines()
            real_err = '\n'.join(l for l in stderr_lines if l and not l.startswith(('ffmpeg version', 'built with', 'configuration', 'lib', '  lib', 'Copyright')))
            real_err = real_err.strip()[-400:]   # last 400 chars = the actual error
            print(f'ffmpeg error:\n{real_err}')
            return jsonify({'error': 'ffmpeg: ' + real_err})

        sid = str(uuid.uuid4())
        with _sessions_lock:
            # Store wav path AND tmp_dir — stream route cleans up after use
            _sessions[sid] = {'wav': wav, 'tmp_dir': tmp_dir}
        return jsonify({'session_id': sid})
    except Exception as e:
        # Only clean up on failure — success path keeps tmp_dir for stream route
        import shutil
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({'error': str(e)})

@flask_app.route('/stream/<session_id>')
def stream(session_id):
    """SSE endpoint — runs full pipeline, streams chunks as they're ready."""
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if not session:
        return Response('data: {"type":"error","msg":"session not found"}\n\n',
                        mimetype='text/event-stream')

    wav     = session['wav']
    tmp_dir = session['tmp_dir']

    def generate():
        import shutil
        try:
            with state['pipeline_lock']:
                yield from stream_pipeline(wav)
        except Exception as e:
            import traceback; traceback.print_exc()
            yield f'data: {json.dumps({"type":"error","msg":str(e)})}\n\n'
        finally:
            # Clean up tmp_dir here — after pipeline is fully done
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )

# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    PORT = 7860
    print(f"Starting at http://localhost:{PORT}")
    print("Press Ctrl+C to quit\n")

    def open_browser():
        time.sleep(1.5)
        subprocess.run(['open', f'http://localhost:{PORT}'])
    threading.Thread(target=open_browser, daemon=True).start()

    flask_app.run(host='0.0.0.0', port=PORT,
                  debug=False, use_reloader=False, threaded=True)

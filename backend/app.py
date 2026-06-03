"""
ZoonoticSense — Voice-Driven Zoonotic Disease Surveillance
===========================================================
Pipeline:
  User Voice → Whisper ASR → NER Extraction → MoE Router
  → Domain RAG Retrieval → Expert LLM Agent → Risk Assessment
  → Kokoro TTS → Spoken verdict + UI risk card

Run:   uvicorn app:app --host 0.0.0.0 --port 7860 --reload
Open:  http://localhost:7860

Author: ZoonoticSense Team
"""

import os
# vLLM v1 on T4 (cc=7.5): disable FlashInfer, use XFormers attention
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_V1_ENABLED", "0")

import multiprocessing as _mp

import os, re, json, base64, tempfile, threading, time, queue, logging, subprocess, shutil, asyncio, uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
KB_DIR     = BASE_DIR / "knowledge_base"
MODELS_DIR = BASE_DIR / "models"

# ── Platform + backend detection ─────────────────────────────────────────────
import platform, sys
IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"

_use_mlx_env = os.getenv("USE_MLX", "auto")
if _use_mlx_env == "auto":
    USE_MLX = IS_MAC          # MLX ใช้ได้เฉพาะ Apple Silicon
elif _use_mlx_env == "true":
    USE_MLX = True
else:
    USE_MLX = False

# Triton — ใช้ได้เฉพาะ non-MLX path
USE_TRITON = (not USE_MLX) and os.getenv("USE_TRITON", "false").lower() == "true"
TRITON_URL = os.getenv("TRITON_URL", "triton:8000")

# CUDA detection — โหลด torch ตอนนี้เลยเฉพาะ non-MLX path
if not USE_MLX:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "mps"            # MLX จัดการ internally

# ── Config (platform-aware defaults) ─────────────────────────────────────────
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
PORT         = int(os.getenv("PORT", 7860))
DEBUG        = os.getenv("DEBUG", "false").lower() == "true"

if USE_MLX:
    _default_llm = "mlx-community/Qwen3-4B-4bit"
elif DEVICE == "cuda":
    _default_llm = "Qwen/Qwen3-4B"
else:
    _default_llm = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL = os.getenv("LLM_MODEL", _default_llm)

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(FFMPEG)
    if _ffmpeg_dir and _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

log.info(f"Platform  : {platform.system()} {platform.machine()} | Python {sys.version.split()[0]}")
log.info(f"Backend   : {'MLX (Apple Silicon)' if USE_MLX else f'transformers → {DEVICE.upper()}'}")
if not USE_MLX and DEVICE == "cuda":
    log.info(f"GPU       : {torch.cuda.get_device_name(0)} | VRAM {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
log.info(f"LLM model : {LLM_MODEL}")
log.info(f"Whisper   : {WHISPER_SIZE}")
log.info(f"ffmpeg    : {FFMPEG}")
log.info(f"Triton    : {'ON @ ' + TRITON_URL if USE_TRITON else 'OFF'}")

# ── Triton client + helpers ───────────────────────────────────────────────
if USE_TRITON:
    import tritonclient.http as _triton_http

    def _triton_client():
        """Create a fresh per-call client to avoid gevent cross-thread errors."""
        return _triton_http.InferenceServerClient(url=TRITON_URL)

    class TritonEmbedder:
        """Drop-in replacement for SentenceTransformer — calls Triton sentence_embedder."""
        def encode(self, texts, convert_to_numpy=True, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            texts_np = np.array([t.encode('utf-8') for t in texts], dtype=object)
            inp = _triton_http.InferInput('text', [len(texts)], 'BYTES')
            inp.set_data_from_numpy(texts_np)
            out = _triton_http.InferRequestedOutput('embedding')
            resp = _triton_client().infer('sentence_embedder', inputs=[inp], outputs=[out])
            return resp.as_numpy('embedding')

    def _transcribe_triton(wav_path: str) -> str:
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
        inp = _triton_http.InferInput('wav_bytes', [1], 'BYTES')
        inp.set_data_from_numpy(np.array([wav_bytes], dtype=object))
        out = _triton_http.InferRequestedOutput('transcript')
        resp = _triton_client().infer('whisper_asr', inputs=[inp], outputs=[out])
        return resp.as_numpy('transcript').flatten()[0].decode('utf-8')

    def _synth_triton(text: str, voice: str) -> str | None:
        text_inp = _triton_http.InferInput('text', [1], 'BYTES')
        text_inp.set_data_from_numpy(np.array([text.encode('utf-8')], dtype=object))
        voice_inp = _triton_http.InferInput('voice', [1], 'BYTES')
        voice_inp.set_data_from_numpy(np.array([voice.encode('utf-8')], dtype=object))
        out = _triton_http.InferRequestedOutput('audio_b64')
        resp = _triton_client().infer('kokoro_tts', inputs=[text_inp, voice_inp], outputs=[out])
        result = resp.as_numpy('audio_b64').flatten()[0].decode('utf-8')
        return result if result else None

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  ZoonoticSense — Loading Models")
print("="*60)

# ── 0. Wait for Triton ──────────────────────────────────────────────────────
if USE_TRITON:
    import time, urllib.request, urllib.error
    _triton_health = f"http://{TRITON_URL}/v2/health/ready"
    print(f"\n[0/5] Waiting for Triton at {_triton_health} ...")
    for _attempt in range(60):
        try:
            urllib.request.urlopen(_triton_health, timeout=5)
            print(f"      ✓ Triton ready (attempt {_attempt + 1})")
            break
        except Exception:
            print(f"      ... attempt {_attempt + 1}/60, retrying in 10s")
            time.sleep(10)
    else:
        raise RuntimeError("Triton not ready after 600s — aborting")

# ── 1. ASR (Whisper) ────────────────────────────────────────────────────────
print(f"\n[1/5] Whisper-{WHISPER_SIZE} (ASR)...")
if USE_TRITON:
    asr_model = None
    ASR_BACKEND = "triton"
elif USE_MLX:
    import mlx_whisper
    asr_model = None
    ASR_BACKEND = "mlx"
else:
    import whisper
    from transformers import pipeline as hf_pipeline
    asr_model = whisper.load_model(WHISPER_SIZE, device="cpu")
    asr_model_th = hf_pipeline(
        "automatic-speech-recognition",
        model="nectec/Pathumma-whisper-th-medium",
        device="cpu",
    )
    ASR_BACKEND = "openai-whisper + Pathumma-th"
print(f"      ✓ ASR ready [{ASR_BACKEND}]")

# ── 2. Sentence Embedder (Router + RAG) ─────────────────────────────────────
print("\n[2/5] Sentence embedder (all-MiniLM-L6-v2)...")
if USE_TRITON:
    embedder = TritonEmbedder()
    print("      ✓ Embedder ready [TRITON]")
else:
    from sentence_transformers import SentenceTransformer
    _embed_device = "cpu" if DEVICE == "cuda" else DEVICE
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=_embed_device)
    print(f"      ✓ Embedder ready [{_embed_device.upper()}]")

# ── 3. Router (MLP classifier) ──────────────────────────────────────────────
print("\n[3/5] MoE Router (NN classifier)...")
from models.router import ZoonoticRouter
router = ZoonoticRouter(embedder=embedder, model_dir=MODELS_DIR)
print(f"      ✓ Router ready | {router.n_domains} domains | backend: {router.backend}")

# ── 4. RAG (FAISS knowledge bases) ─────────────────────────────────────────
print("\n[4/5] RAG knowledge bases (FAISS)...")
from utils.rag import DomainRAG
rag = DomainRAG(kb_dir=KB_DIR, embedder=embedder)
print(f"      ✓ RAG ready | {rag.n_indexes} domain indexes loaded")

# ── 5. LLM ──────────────────────────────────────────────────────────────────
print(f"\n[5/5] LLM ({LLM_MODEL})...")
if USE_MLX:
    from mlx_lm import load as mlx_load, stream_generate
    llm, llm_tok = mlx_load(LLM_MODEL)
    LLM_BACKEND = "mlx-lm"
else:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams as VLLMSamplingParams
    from transformers import AutoTokenizer
    import asyncio as _asyncio
    import uuid as _uuid

    llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)

    if DEVICE == "cuda":
        _eargs = AsyncEngineArgs(
            model=LLM_MODEL,
            dtype="float16",
            max_model_len=1024,
            gpu_memory_utilization=0.85,
        )
        LLM_BACKEND = "vllm [CUDA+compressed-tensors]"
    else:
        _eargs = AsyncEngineArgs(
            model=LLM_MODEL,
            dtype="float32",
            max_model_len=2048,
        )
        LLM_BACKEND = "vllm [CPU]"

    if _mp.current_process().name == 'MainProcess':
        _vllm_loop = _asyncio.new_event_loop()
        threading.Thread(target=_vllm_loop.run_forever, daemon=True, name="vllm-loop").start()

        async def _create_engine():
            return AsyncLLMEngine.from_engine_args(_eargs)

        llm = _asyncio.run_coroutine_threadsafe(_create_engine(), _vllm_loop).result(timeout=600)

        def _run_async(coro):
            return _asyncio.run_coroutine_threadsafe(coro, _vllm_loop).result()

        async def _vllm_full(prompt: str, max_tokens: int) -> str:
            params = VLLMSamplingParams(max_tokens=max_tokens, temperature=0.0)
            out_text = ""
            async for req_out in llm.generate(prompt, params, request_id=str(_uuid.uuid4())):
                out_text = req_out.outputs[0].text
            return out_text
print(f"      ✓ LLM ready [{LLM_BACKEND}]")

# ── 6. TTS (Kokoro) ─────────────────────────────────────────────────────────
print("\n[6/6] Kokoro TTS...")
import soundfile as sf
import queue as _queue
if USE_TRITON:
    tts_pool = None
    print("      ✓ TTS ready [TRITON × 3 instances]")
else:
    from kokoro import KPipeline
    _tts_device = "cpu" if DEVICE == "cuda" else DEVICE
    _TTS_POOL_SIZE = 3
    tts_pool = _queue.Queue()
    for _ in range(_TTS_POOL_SIZE):
        tts_pool.put(KPipeline(lang_code='a', device=_tts_device))
    print(f"      ✓ TTS ready [{_tts_device.upper()}] × {_TTS_POOL_SIZE} instances")


print("\n" + "="*60)
print("  All systems nominal. Starting server...")
print("="*60 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
#  NER / EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """You are an epidemiological field data extractor.
Extract structured information from the report below.
Return ONLY a JSON object with these exact keys (use null if not mentioned):

{{
  "species": ["list of animal species mentioned"],
  "symptoms": ["list of symptoms or behaviors observed"],
  "mortality_count": <integer or null>,
  "affected_count": <integer or null>,
  "location": "<location string or null>",
  "timeframe": "<how long ago / duration or null>",
  "reporter_role": "<farmer | ranger | vet | researcher | public | unknown>",
  "raw_summary": "<one sentence summary of the report>"
}}

Report: {report}

JSON:"""


def extract_epi_fields(text: str) -> dict:
    if not USE_MLX and DEVICE == 'cpu':
        return {
            "species": [], "symptoms": [], "mortality_count": None,
            "affected_count": None, "location": None, "timeframe": None,
            "reporter_role": "unknown", "raw_summary": text[:200]
        }

    prompt = EXTRACT_PROMPT.format(report=text)

    if USE_MLX:
        messages = [{"role": "user", "content": prompt}]
        formatted = llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_out = ""
        for tok in stream_generate(llm, llm_tok, formatted, max_tokens=512):
            full_out += tok.text
    else:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = llm_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            full_out = _run_async(_vllm_full(formatted, 200))
        except Exception as e:
            log.error(f"NER generate failed: {e}")
            return {
                "species": [], "symptoms": [], "mortality_count": None,
                "affected_count": None, "location": None, "timeframe": None,
                "reporter_role": "unknown", "raw_summary": text[:200]
            }

    full_out = re.sub(r'<think>.*?</think>', '', full_out, flags=re.DOTALL).strip()

    try:
        json_match = re.search(r'\{.*\}', full_out, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return {
        "species": [], "symptoms": [], "mortality_count": None,
        "affected_count": None, "location": None, "timeframe": None,
        "reporter_role": "unknown", "raw_summary": text[:200]
    }


# ═══════════════════════════════════════════════════════════════════════════
#  EXPERT AGENT
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_EXPERTS = {
    "avian_flu":     {"name": "Avian Influenza Expert",    "voice": "af_heart"},
    "rabies":        {"name": "Rabies Surveillance Agent", "voice": "am_michael"},
    "fmd":           {"name": "FMD Field Specialist",      "voice": "am_adam"},
    "nipah_hendra":  {"name": "Nipah/Hendra Analyst",      "voice": "af_bella"},
    "leptospirosis": {"name": "Leptospirosis Advisor",     "voice": "am_michael"},
    "general":       {"name": "General Zoonotic Advisor",  "voice": "af_heart"},
}

EXPERT_SYSTEM = """You are {expert_name}, a specialist in zoonotic disease surveillance.
You advise farmers, rangers, and field workers in plain spoken language — like a calm expert on the phone.

CONTEXT FROM VETERINARY KNOWLEDGE BASE:
{rag_context}

EPIDEMIOLOGICAL REPORT:
Species: {species}
Symptoms: {symptoms}
Mortality: {mortality}
Location: {location}
Timeframe: {timeframe}

YOUR RESPONSE — cover ALL of these, in order, using plain natural sentences:
1. CLINICAL ASSESSMENT: What is likely happening based on species and symptoms. Top 1-2 disease candidates.
2. RISK LEVEL: State exactly one of: LOW / MEDIUM / HIGH / CRITICAL — and briefly explain why.
3. IMMEDIATE ACTIONS: What the farmer/ranger should do RIGHT NOW (isolation, PPE, stop movement).
4. TREATMENT / HANDLING: Any first-response treatment options, medications, or supportive care.
5. REPORTING: Whether this MUST BE REPORTED TO AUTHORITIES — yes or no, and why.
6. HUMAN SAFETY: Any risk to people on the farm and what precautions they should take.

Speak naturally. No markdown. No bullet points. Use flowing sentences as if on a phone call.
Aim for 4-6 sentences total — detailed but clear and not overwhelming."""

CHAT_SYSTEM = """You are a friendly and knowledgeable animal health advisor — think of yourself as a village doctor who cares deeply about both animals and the people who look after them.

Your role is to have a warm, helpful conversation. The person talking to you may be a farmer, a pet owner, or just a curious person.

If they are asking a general question or just chatting, respond in a friendly, natural way — like a doctor you can call on the phone.
If they mention any symptoms, illness, or concern about animals, gently ask a follow-up question to learn more.
If this appears to be a genuine disease concern, tell them you can do a proper assessment if they describe the species, symptoms, and how many animals are affected.

Keep responses SHORT — 2 to 3 sentences only. Be warm, not clinical. Speak plainly. No bullet points, no lists."""


def _strip_think_stream(token_iter):
    buf    = ""
    inside = False
    for tok in token_iter:
        buf += tok
        while True:
            if inside:
                end = buf.find("</think>")
                if end != -1:
                    buf    = buf[end + 8:]
                    inside = False
                else:
                    buf = ""
                    break
            else:
                start = buf.find("<think>")
                if start != -1:
                    yield buf[:start]
                    buf    = buf[start + 7:]
                    inside = True
                else:
                    safe = max(0, len(buf) - 8)
                    if safe:
                        yield buf[:safe]
                        buf = buf[safe:]
                    break
    if buf and not inside:
        yield buf


def stream_expert_response(domain: str, epi_fields: dict, rag_chunks: list):
    is_off_topic = epi_fields.get("_off_topic", False)

    if is_off_topic:
        user_msg  = epi_fields.get("raw_summary") or "Hello!"
        messages  = [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
    else:
        expert      = DOMAIN_EXPERTS.get(domain, DOMAIN_EXPERTS["general"])
        rag_context = "\n\n".join(rag_chunks) if rag_chunks else "No specific records retrieved."
        system_prompt = EXPERT_SYSTEM.format(
            expert_name=expert["name"],
            rag_context=rag_context[:1800],
            species=", ".join(epi_fields.get("species", []) or ["unknown"]),
            symptoms=", ".join(epi_fields.get("symptoms", []) or ["unspecified"]),
            mortality=epi_fields.get("mortality_count") or "unknown",
            location=epi_fields.get("location") or "unspecified",
            timeframe=epi_fields.get("timeframe") or "unspecified",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "Please assess this zoonotic event."},
        ]
    max_tok = 180 if is_off_topic else 650

    if USE_MLX:
        formatted = llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        raw_tokens = (tok.text for tok in stream_generate(llm, llm_tok, formatted, max_tokens=max_tok))
        yield from _strip_think_stream(raw_tokens)
    else:
        formatted = llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        tok_q = queue.Queue()

        async def _stream_vllm():
            params = VLLMSamplingParams(max_tokens=max_tok, temperature=0.0)
            prev_len = 0
            async for req_out in llm.generate(
                formatted, params, request_id=str(_uuid.uuid4())
            ):
                new_text = req_out.outputs[0].text[prev_len:]
                prev_len = len(req_out.outputs[0].text)
                if new_text:
                    tok_q.put(new_text)
            tok_q.put(None)

        _asyncio.run_coroutine_threadsafe(_stream_vllm(), _vllm_loop)

        def _token_iter():
            while True:
                tok = tok_q.get()
                if tok is None:
                    break
                yield tok

        yield from _strip_think_stream(_token_iter())


# ═══════════════════════════════════════════════════════════════════════════
#  RISK PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_risk_level(text: str, is_chat: bool = False) -> str:
    if is_chat:
        return "NONE"
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if re.search(rf'\b{level}\b', text, re.IGNORECASE):
            return level
    return "UNKNOWN"


def parse_report_flag(text: str) -> bool:
    text_lower = text.lower()
    report_words = ["report", "notify", "alert authoritie", "contact dld",
                    "contact the district", "inform the"]
    return any(w in text_lower for w in report_words)


# ═══════════════════════════════════════════════════════════════════════════
#  TTS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

_SENTENCE_BOUNDARY = re.compile(r'[.!?](?:\s|$)')
_ABBREV = {
    r'\bHPAI\b': 'H.P.A.I.', r'\bFMD\b': 'F.M.D.', r'\bASR\b': 'A.S.R.',
    r'\bTTS\b': 'T.T.S.',   r'\bRAG\b': 'R.A.G.', r'\bLLM\b': 'L.L.M.',
}


def clean_for_tts(text: str) -> str:
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*',   r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = text.replace('\u2014', ', ').replace('\u2013', ' to ')
    text = text.replace('\u2026', '...')
    for pat, rep in _ABBREV.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'  +', ' ', text).strip()


def iter_sentence_chunks(token_iter, min_words: int = 5):
    buf = ""
    for tok in token_iter:
        buf += tok
        if _SENTENCE_BOUNDARY.search(buf):
            m = None
            for m in _SENTENCE_BOUNDARY.finditer(buf):
                pass
            if m and len(buf[:m.end()].split()) >= min_words:
                chunk = buf[:m.end()].strip()
                buf   = buf[m.end():].strip()
                yield chunk
    if buf.strip():
        yield buf.strip()


def synth_audio_b64(text: str, voice: str = "af_heart") -> str | None:
    text = clean_for_tts(text)
    if not text:
        return None
    if USE_TRITON:
        return _synth_triton(text, voice)
    pipe = tts_pool.get()
    try:
        samples = []
        for _, _, audio in pipe(text, voice=voice, speed=1.0):
            samples.append(audio)
        if not samples:
            return None
        audio_np = np.concatenate(samples)
        tmp_path = tempfile.mktemp(suffix='.wav')
        sf.write(tmp_path, audio_np, 24000)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        os.unlink(tmp_path)
        return base64.b64encode(data).decode()
    except Exception as e:
        log.error(f"TTS error: {e}")
        return None
    finally:
        tts_pool.put(pipe)


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════

try:
    from db import create_tables, get_session
    from db_models import Report as ReportModel
    create_tables()
    log.info("Database connected and tables ready")
except Exception as _dbe:
    log.warning(f"Database unavailable — reports will not be saved: {_dbe}")
    get_session = lambda: None
    ReportModel = None

# ═══════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="ZoonoticSense", version="2.0.0")

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",   # any Vercel deployment URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
log.info(f"CORS origins : {_origins} + *.vercel.app (regex)")

# Serve static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

session_state = {
    "history":     [],
    "last_report": {},
}


# ── Pydantic request models ──────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    transcript: str

class StreamRequest(BaseModel):
    domain: str = "general"
    epi_fields: dict = {}
    rag_chunks: list = []
    transcript: str = ""
    timing: dict = {}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get('/favicon.ico')
@app.get('/favicon.svg')
async def favicon():
    return FileResponse(str(BASE_DIR / 'static' / 'favicon.svg'), media_type='image/svg+xml')


@app.get('/')
async def landing(request: Request):
    return templates.TemplateResponse('landing.html', {'request': request})


@app.get('/app')
async def index(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'domains': list(DOMAIN_EXPERTS.keys()),
        'version': '2.0.0',
    })


@app.post('/upload')
async def upload(audio: UploadFile = File(...)):
    """Receive audio, run ASR, return transcript."""
    orig_ext = Path(audio.filename).suffix.lower() if audio.filename else '.webm'
    if orig_ext not in {'.webm', '.m4a', '.mp3', '.wav', '.ogg', '.aac', '.flac'}:
        orig_ext = '.webm'

    with tempfile.NamedTemporaryFile(suffix=orig_ext, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    wav_path = tmp_path[:-len(orig_ext)] + '.wav'
    try:
        subprocess.run(
            [FFMPEG, '-y', '-i', tmp_path, '-ar', '16000', '-ac', '1', wav_path],
            capture_output=True, check=True
        )

        t0 = time.time()
        if ASR_BACKEND == "triton":
            transcript = _transcribe_triton(wav_path)
        elif ASR_BACKEND == "mlx":
            result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=f"mlx-community/whisper-{WHISPER_SIZE}-mlx")
            transcript = result["text"].strip()
        elif ASR_BACKEND == "openai-whisper + Pathumma-th":
            audio_np, _ = sf.read(wav_path, dtype='float32')
            audio_for_detect = whisper.pad_or_trim(audio_np)
            mel = whisper.log_mel_spectrogram(audio_for_detect).to(asr_model.device)
            _, lang_probs = asr_model.detect_language(mel)
            detected_lang = max(lang_probs, key=lang_probs.get)
            log.info(f"ASR detected language: {detected_lang} (conf={lang_probs[detected_lang]:.2f})")

            if detected_lang == "th":
                th_result = asr_model_th({"raw": audio_np, "sampling_rate": 16000})
                transcript = th_result["text"].strip()
            else:
                result = asr_model.transcribe(audio_np)
                transcript = result["text"].strip()
        else:
            audio_np, _ = sf.read(wav_path, dtype='float32')
            result = asr_model.transcribe(audio_np)
            transcript = result["text"].strip()
        asr_time = round(time.time() - t0, 2)

        # Hallucination guard
        words = transcript.lower().split()
        is_hallucination = False
        if len(words) > 10:
            max_run = 1; cur_run = 1
            for i in range(1, len(words)):
                if words[i] == words[i - 1]:
                    cur_run += 1; max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            if max_run > 6:
                is_hallucination = True
            if not is_hallucination:
                from collections import Counter
                top3 = sum(v for _, v in Counter(words).most_common(3))
                if top3 / len(words) > 0.80:
                    is_hallucination = True

        if is_hallucination:
            log.warning(f"ASR hallucination detected ({asr_time}s), discarding transcript.")
            raise HTTPException(status_code=422, detail="Audio unclear — please record again in a quieter environment.")

        log.info(f"ASR ({asr_time}s): {transcript}")
        return {"transcript": transcript, "asr_time": asr_time}

    except HTTPException:
        raise
    except subprocess.CalledProcessError as e:
        log.error(f"ffmpeg failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")
    except Exception as e:
        log.error(f"Upload error ({type(e).__name__}): {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [tmp_path, wav_path]:
            try: os.unlink(p)
            except: pass


@app.post('/analyze')
async def analyze(body: AnalyzeRequest):
    """Full pipeline: transcript → NER → Router → RAG."""
    transcript = body.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcript")

    try:
        t0 = time.time()
        epi_fields = await asyncio.get_event_loop().run_in_executor(None, extract_epi_fields, transcript)
        ner_time   = round(time.time() - t0, 2)
        log.info(f"NER ({ner_time}s): {epi_fields}")

        t1 = time.time()
        domain, confidence, all_scores = router.classify(transcript, epi_fields)
        route_time = round(time.time() - t1, 2)
        # Which cascade stage answered (1 = cheap text model, 2 = embedding+MLP)
        route_stage = getattr(router, "last_stage", None)
        log.info(f"Router ({route_time}s): domain={domain} conf={confidence:.2f}")

        has_species   = bool(epi_fields.get("species"))
        has_symptoms  = bool(epi_fields.get("symptoms"))
        has_mortality = bool(epi_fields.get("mortality_count"))

        _CHAT_PAT = re.compile(
            r"^\s*(hi|hello|hey|good\s+\w+|how\s+(do|to|can|does)|what\s+is|can\s+you|"
            r"teach\s+me|tell\s+me|explain|thanks|thank\s+you|what\s+should|is\s+it\s+safe)",
            re.IGNORECASE,
        )
        is_chat_query = bool(_CHAT_PAT.match(transcript)) and not has_mortality
        is_off_topic  = is_chat_query or (not has_species and not has_symptoms and confidence < 0.35)

        if is_off_topic:
            domain = "general"
            log.info(f"Off-topic/chat input detected (is_chat={is_chat_query}) — routing to friendly advisor")
            epi_fields["_off_topic"] = True

        t2 = time.time()
        rag_chunks = rag.retrieve(transcript, domain=domain, top_k=3)
        rag_time   = round(time.time() - t2, 2)
        log.info(f"RAG ({rag_time}s): {len(rag_chunks)} chunks from domain={domain}")

        session_state["last_report"] = {
            "transcript": transcript,
            "epi_fields": epi_fields,
            "domain":     domain,
        }

        return {
            "epi_fields":  epi_fields,
            "domain":      domain,
            "confidence":  round(confidence, 3),
            "all_scores":  {k: round(v, 3) for k, v in all_scores.items()},
            "rag_chunks":  rag_chunks,
            "route_stage":    route_stage,
            "router_backend": getattr(router, "backend", None),
            "timing": {
                "ner_s":   ner_time,
                "route_s": route_time,
                "rag_s":   rag_time,
            }
        }
    except Exception as e:
        log.error(f"Analyze error ({type(e).__name__}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/stream')
async def stream(body: StreamRequest):
    """Stream expert LLM response + TTS audio chunks via SSE."""
    domain     = body.domain
    epi_fields = body.epi_fields
    rag_chunks = body.rag_chunks
    transcript = body.transcript
    timing     = body.timing
    voice      = DOMAIN_EXPERTS.get(domain, DOMAIN_EXPERTS["general"])["voice"]

    def generate():
        full_text  = ""
        audio_sent = 0
        # ── latency instrumentation (Expert LLM + TTS stages) ──
        t_start  = time.time()
        ttft     = None          # time-to-first-token/sentence (LLM responsiveness)
        ttfa     = None          # time-to-first-audio (spoken responsiveness)
        tts_time = 0.0           # accumulated time inside speech synthesis

        token_gen = stream_expert_response(domain, epi_fields, rag_chunks)

        for sentence in iter_sentence_chunks(token_gen):
            if ttft is None:
                ttft = time.time() - t_start
            full_text += " " + sentence
            yield f"data: {json.dumps({'type': 'text', 'chunk': sentence})}\n\n"

            _t_tts = time.time()
            audio_b64 = synth_audio_b64(sentence, voice=voice)
            tts_time += time.time() - _t_tts
            if audio_b64:
                if ttfa is None:
                    ttfa = time.time() - t_start
                audio_sent += 1
                yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64, 'idx': audio_sent, 'sentence': sentence})}\n\n"

        _elapsed  = time.time() - t_start
        _llm_time = max(_elapsed - tts_time, 0.0)   # generation minus TTS blocking
        latency = {
            "total_s": round(_elapsed, 2),
            "ttft_s":  round(ttft, 3) if ttft is not None else None,
            "ttfa_s":  round(ttfa, 3) if ttfa is not None else None,
            "llm_s":   round(_llm_time, 2),
            "tts_s":   round(tts_time, 2),
            "audio_chunks": audio_sent,
        }

        is_chat     = epi_fields.get("_off_topic", False)
        risk_level  = parse_risk_level(full_text, is_chat=is_chat)
        report_flag = parse_report_flag(full_text) if not is_chat else False
        expert_name = DOMAIN_EXPERTS.get(domain, DOMAIN_EXPERTS["general"])["name"]

        risk_card = {
            "risk_level":    risk_level,
            "report_flag":   report_flag,
            "domain":        domain,
            "expert_name":   expert_name,
            "full_response": full_text.strip(),
            "is_chat":       bool(is_chat),
        }

        session_state["history"].append({
            "role": "assistant",
            "domain": domain,
            "risk_level": risk_level,
            "text": full_text.strip(),
        })

        # ── Save report to database ──────────────────────────────────────
        try:
            db = get_session()
            if db and ReportModel:
                row = ReportModel(
                    id              = str(uuid.uuid4()),
                    transcript      = transcript or session_state.get("last_report", {}).get("transcript", ""),
                    domain          = domain,
                    confidence      = session_state.get("last_report", {}).get("confidence"),
                    risk_level      = risk_level,
                    report_flag     = str(report_flag),
                    llm_response    = full_text.strip(),
                    species         = epi_fields.get("species", []),
                    symptoms        = epi_fields.get("symptoms", []),
                    mortality_count = epi_fields.get("mortality_count"),
                    affected_count  = epi_fields.get("affected_count"),
                    location        = epi_fields.get("location"),
                    reporter_role   = epi_fields.get("reporter_role"),
                    asr_time_s      = timing.get("asr_s"),
                    ner_time_s      = timing.get("ner_s"),
                )
                db.add(row)
                db.commit()
                db.close()
                log.info(f"Report saved: id={row.id} domain={domain} risk={risk_level}")
        except Exception as _dbe:
            log.error(f"Failed to save report: {_dbe}")

        yield f"data: {json.dumps({'type': 'risk_card', 'card': risk_card})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'latency': latency})}\n\n"

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.get('/history')
async def history():
    return session_state["history"]


@app.post('/reset')
async def reset():
    session_state["history"].clear()
    session_state["last_report"] = {}
    return {"ok": True}


@app.get('/health')
async def health():
    return {
        "status":         "ok",
        "llm_backend":    LLM_BACKEND,
        "asr_backend":    ASR_BACKEND,
        "router_backend": router.backend,
        "rag_indexes":    rag.n_indexes,
        "domains":        list(DOMAIN_EXPERTS.keys()),
    }


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import uvicorn
    log.info(f"Starting ZoonoticSense on http://localhost:{PORT}")
    uvicorn.run("app:app", host='0.0.0.0', port=PORT, reload=DEBUG)

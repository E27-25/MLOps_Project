"""
ZoonoticSense — Voice-Driven Zoonotic Disease Surveillance
===========================================================
Pipeline:
  User Voice → Whisper ASR → NER Extraction → MoE Router
  → Domain RAG Retrieval → Expert LLM Agent → Risk Assessment
  → Kokoro TTS → Spoken verdict + UI risk card

Run:   python app.py
Open:  http://localhost:7860

Author: ZoonoticSense Team
"""

import os, re, json, base64, tempfile, threading, time, queue, logging, subprocess
from pathlib import Path
from flask import Flask, request, jsonify, Response, render_template, stream_with_context
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
KB_DIR       = BASE_DIR / "knowledge_base"
MODELS_DIR   = BASE_DIR / "models"
USE_MLX      = os.getenv("USE_MLX", "auto")   # "auto" | "true" | "false"
LLM_MODEL    = os.getenv("LLM_MODEL", "mlx-community/Qwen3-4B-4bit")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")   # base | small | medium
PORT         = int(os.getenv("PORT", 7860))
DEBUG        = os.getenv("DEBUG", "false").lower() == "true"

# ── Device detection ────────────────────────────────────────────────────────
import platform
IS_MAC = platform.system() == "Darwin"

if USE_MLX == "auto":
    USE_MLX = IS_MAC
elif USE_MLX == "true":
    USE_MLX = True
else:
    USE_MLX = False

log.info(f"Platform: {platform.system()} | MLX: {USE_MLX} | LLM: {LLM_MODEL}")

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  ZoonoticSense — Loading Models")
print("="*60)

# ── 1. ASR (Whisper) ────────────────────────────────────────────────────────
print(f"\n[1/5] Whisper-{WHISPER_SIZE} (ASR)...")
if USE_MLX:
    import mlx_whisper
    asr_model = None   # mlx_whisper uses functional API
    ASR_BACKEND = "mlx"
else:
    import whisper
    asr_model = whisper.load_model(WHISPER_SIZE, device='cpu')
    ASR_BACKEND = "openai-whisper"
print(f"      ✓ ASR ready [{ASR_BACKEND}]")

# ── 2. Sentence Embedder (Router + RAG) ─────────────────────────────────────
print("\n[2/5] Sentence embedder (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("      ✓ Embedder ready")

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
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    _fallback = "Qwen/Qwen2.5-1.5B-Instruct"
    log.info(f"Non-MLX: loading {_fallback} on {DEVICE}")
    llm_tok = AutoTokenizer.from_pretrained(_fallback)
    llm = AutoModelForCausalLM.from_pretrained(
        _fallback,
        torch_dtype=torch.float16 if DEVICE != 'cpu' else torch.float32,
        device_map='auto',
    )
    llm.eval()
    LLM_BACKEND = "transformers"
print(f"      ✓ LLM ready [{LLM_BACKEND}]")

# ── 6. TTS (Kokoro) ─────────────────────────────────────────────────────────
print("\n[6/?] Kokoro TTS...")
from kokoro import KPipeline
import soundfile as sf
tts_pipe = KPipeline(lang_code='a')
print("      ✓ TTS ready")

print("\n" + "="*60)
print("  All systems nominal. Starting server...")
print("="*60 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
#  NER / EXTRACTION  (structured epi fields from raw speech)
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
    """Extract structured epidemiological fields from raw speech transcript."""
    prompt = EXTRACT_PROMPT.format(report=text)

    if USE_MLX:
        messages = [{"role": "user", "content": prompt}]
        formatted = llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,   # Qwen3: disable chain-of-thought
        )
        full_out = ""
        for tok in stream_generate(llm, llm_tok, formatted, max_tokens=512):
            full_out += tok.text
    else:
        inputs = llm_tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = llm.generate(**inputs, max_new_tokens=512, do_sample=False)
        full_out = llm_tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Strip any <think>...</think> blocks (Qwen3 chain-of-thought)
    full_out = re.sub(r'<think>.*?</think>', '', full_out, flags=re.DOTALL).strip()

    # Parse JSON from output
    try:
        json_match = re.search(r'\{.*\}', full_out, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    # Fallback: minimal structure
    return {
        "species": [], "symptoms": [], "mortality_count": None,
        "affected_count": None, "location": None, "timeframe": None,
        "reporter_role": "unknown", "raw_summary": text[:200]
    }


# ═══════════════════════════════════════════════════════════════════════════
#  EXPERT AGENT  (LLM with RAG context + structured output)
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
    """Generator: transparently remove <think>...</think> blocks from a token stream."""
    buf    = ""
    inside = False
    for tok in token_iter:
        buf += tok
        while True:
            if inside:
                end = buf.find("</think>")
                if end != -1:
                    buf    = buf[end + 8:]   # discard everything up to and including </think>
                    inside = False
                else:
                    buf = ""                 # still inside think block — discard buffer
                    break
            else:
                start = buf.find("<think>")
                if start != -1:
                    yield buf[:start]        # yield text before <think>
                    buf    = buf[start + 7:]
                    inside = True
                else:
                    # yield safely, keeping last 8 chars as lookahead for split tags
                    safe = max(0, len(buf) - 8)
                    if safe:
                        yield buf[:safe]
                        buf = buf[safe:]
                    break
    if buf and not inside:
        yield buf


def stream_expert_response(domain: str, epi_fields: dict, rag_chunks: list):
    """Stream tokens from the expert LLM agent (think-blocks stripped)."""
    is_off_topic = epi_fields.get("_off_topic", False)

    if is_off_topic:
        # Use friendly chat prompt for general conversation
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
            enable_thinking=False,   # Qwen3: disable chain-of-thought
        )
        raw_tokens = (tok.text for tok in stream_generate(llm, llm_tok, formatted, max_tokens=max_tok))
        yield from _strip_think_stream(raw_tokens)
    else:
        import torch
        from transformers import TextIteratorStreamer
        formatted = llm_tok.apply_chat_template(messages, return_tensors='pt').to(DEVICE)
        streamer  = TextIteratorStreamer(llm_tok, skip_prompt=True, skip_special_tokens=True)
        thread    = threading.Thread(
            target=llm.generate,
            kwargs=dict(inputs=formatted, max_new_tokens=max_tok, streamer=streamer, do_sample=False)
        )
        thread.start()
        yield from _strip_think_stream(streamer)


# ═══════════════════════════════════════════════════════════════════════════
#  RISK PARSER  (extract structured risk card from LLM output)
# ═══════════════════════════════════════════════════════════════════════════

def parse_risk_level(text: str) -> str:
    """Extract risk level keyword from LLM response."""
    text_upper = text.upper()
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if level in text_upper:
            return level
    return "UNKNOWN"


def parse_report_flag(text: str) -> bool:
    """Detect if LLM recommends reporting to authorities."""
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
    """Strip markdown and normalise text so Kokoro TTS reads it cleanly."""
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
    """Consumes a token iterator and yields complete sentences at boundaries.
    Uses word-count (min_words=5 default) — short friendly sentences still get TTS'd.
    """
    buf = ""
    for tok in token_iter:
        buf += tok
        if _SENTENCE_BOUNDARY.search(buf):
            # Find the LAST sentence boundary
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
    """Synthesize speech and return base64-encoded WAV (all chunks concatenated)."""
    text = clean_for_tts(text)
    if not text:
        return None
    try:
        samples = []
        for _, _, audio in tts_pipe(text, voice=voice, speed=1.0):
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


# ═══════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder='templates', static_folder='static')

pipeline_lock = threading.Lock()
session_state = {
    "history":     [],   # conversation turns
    "last_report": {},   # last epi report for follow-up context
}


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/app')
def index():
    return render_template('index.html',
                           domains=list(DOMAIN_EXPERTS.keys()),
                           version="1.0.0")


@app.route('/upload', methods=['POST'])
def upload():
    """Receive audio, run ASR, return transcript + epi fields."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Convert to WAV for Whisper
        wav_path = tmp_path.replace('.webm', '.wav')
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_path, '-ar', '16000', '-ac', '1', wav_path],
            capture_output=True, check=True
        )

        # ASR
        t0 = time.time()
        if ASR_BACKEND == "mlx":
            result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=f"mlx-community/whisper-{WHISPER_SIZE}-mlx")
        else:
            result = asr_model.transcribe(wav_path)
        transcript = result["text"].strip()
        asr_time   = round(time.time() - t0, 2)
        log.info(f"ASR ({asr_time}s): {transcript}")

        return jsonify({
            "transcript": transcript,
            "asr_time":   asr_time,
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Audio conversion failed: {e}"}), 500
    finally:
        for p in [tmp_path, tmp_path.replace('.webm', '.wav')]:
            try: os.unlink(p)
            except: pass


@app.route('/analyze', methods=['POST'])
def analyze():
    """Full pipeline: transcript → NER → Router → RAG → Expert LLM → TTS stream."""
    data       = request.get_json()
    transcript = data.get("transcript", "").strip()
    if not transcript:
        return jsonify({"error": "Empty transcript"}), 400

    # Step 1 — NER extraction
    t0 = time.time()
    epi_fields = extract_epi_fields(transcript)
    ner_time   = round(time.time() - t0, 2)
    log.info(f"NER ({ner_time}s): {epi_fields}")

    # Step 2 — Router classification
    t1 = time.time()
    domain, confidence, all_scores = router.classify(transcript, epi_fields)
    route_time = round(time.time() - t1, 2)
    log.info(f"Router ({route_time}s): domain={domain} conf={confidence:.2f}")

    # Off-topic guard — two conditions trigger chat mode:
    # 1) No epi signal at all AND low router confidence
    # 2) Starts with a clear greeting / general question pattern with no mortality
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

    # Step 3 — RAG retrieval
    t2 = time.time()
    rag_chunks = rag.retrieve(transcript, domain=domain, top_k=3)
    rag_time   = round(time.time() - t2, 2)
    log.info(f"RAG ({rag_time}s): {len(rag_chunks)} chunks from domain={domain}")

    # Update session
    session_state["last_report"] = {
        "transcript": transcript,
        "epi_fields": epi_fields,
        "domain":     domain,
    }

    return jsonify({
        "epi_fields":   epi_fields,
        "domain":       domain,
        "confidence":   round(confidence, 3),
        "all_scores":   {k: round(v, 3) for k, v in all_scores.items()},
        "rag_chunks":   rag_chunks,
        "timing": {
            "ner_s":   ner_time,
            "route_s": route_time,
            "rag_s":   rag_time,
        }
    })


@app.route('/stream', methods=['POST'])
def stream():
    """Stream expert LLM response + TTS audio chunks via SSE."""
    data       = request.get_json()
    domain     = data.get("domain", "general")
    epi_fields = data.get("epi_fields", {})
    rag_chunks = data.get("rag_chunks", [])
    voice      = DOMAIN_EXPERTS.get(domain, DOMAIN_EXPERTS["general"])["voice"]

    def generate():
        full_text  = ""
        audio_sent = 0

        with pipeline_lock:
            token_gen = stream_expert_response(domain, epi_fields, rag_chunks)

            for sentence in iter_sentence_chunks(token_gen):
                full_text += " " + sentence

                # Stream text token
                yield f"data: {json.dumps({'type': 'text', 'chunk': sentence})}\n\n"

                # Synthesize + stream audio
                audio_b64 = synth_audio_b64(sentence, voice=voice)
                if audio_b64:
                    audio_sent += 1
                    yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64, 'idx': audio_sent, 'sentence': sentence})}\n\n"

        # After full generation — parse risk card
        risk_level   = parse_risk_level(full_text)
        report_flag  = parse_report_flag(full_text)
        expert_name  = DOMAIN_EXPERTS.get(domain, DOMAIN_EXPERTS["general"])["name"]

        risk_card = {
            "risk_level":    risk_level,
            "report_flag":   report_flag,
            "domain":        domain,
            "expert_name":   expert_name,
            "full_response": full_text.strip(),
        }

        # Append to session history
        session_state["history"].append({
            "role": "assistant",
            "domain": domain,
            "risk_level": risk_level,
            "text": full_text.strip(),
        })

        yield f"data: {json.dumps({'type': 'risk_card', 'card': risk_card})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':  'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/history', methods=['GET'])
def history():
    return jsonify(session_state["history"])


@app.route('/reset', methods=['POST'])
def reset():
    session_state["history"].clear()
    session_state["last_report"] = {}
    return jsonify({"ok": True})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":        "ok",
        "llm_backend":   LLM_BACKEND,
        "asr_backend":   ASR_BACKEND,
        "router_backend": router.backend,
        "rag_indexes":   rag.n_indexes,
        "domains":       list(DOMAIN_EXPERTS.keys()),
    })


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    log.info(f"Starting ZoonoticSense on http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG, threaded=True)

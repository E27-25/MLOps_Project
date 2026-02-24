/**
 * ZoonoticSense — Frontend Pipeline Controller
 * Handles: recording → ASR → analyze → SSE stream → TTS playback → UI updates
 */

// ── Collapsible toggle ─────────────────────────────────────────────────────
function toggleSection(bodyId, btn) {
  const body = document.getElementById(bodyId);
  if (!body) return;
  const isCollapsed = body.classList.contains('collapsed');
  body.classList.toggle('collapsed', !isCollapsed);
  btn.setAttribute('aria-expanded', isCollapsed ? 'true' : 'false');
  if (isCollapsed) {
    btn.removeAttribute('data-collapsed');
  } else {
    btn.setAttribute('data-collapsed', '');
  }
  // Chevron handled by CSS via data-collapsed attribute
  const chevron = btn.querySelector('.toggle-chevron');
  if (chevron) chevron.textContent = isCollapsed ? '▾' : '▸';
}

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  isRecording: false,
  mediaRecorder: null,
  audioChunks: [],
  audioQueue: [],   // {audio, btn, blobUrl} entries
  isPlaying: false,
  chunkIdx: 0,
  runId: 0,         // incremented each run to discard stale onended callbacks
};

// ── Utility: set pipeline step status ──────────────────────────────────────
function stepStatus(id, status, detail = '', time = '') {
  const el = document.getElementById(`step-${id}`);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (status === 'active') el.classList.add('active');
  if (status === 'done') el.classList.add('done');

  const detailEl = document.getElementById(`step-${id}-detail`);
  if (detailEl && detail) detailEl.textContent = detail;

  const timeEl = document.getElementById(`step-${id}-time`);
  if (timeEl && time) timeEl.textContent = time;
}

function setStatus(text, mode = 'ready') {
  document.getElementById('statusText').textContent = text;
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot';
  if (mode === 'busy') dot.classList.add('busy');
  if (mode === 'error') dot.classList.add('error');
}

// ── Canvas waveform ─────────────────────────────────────────────────────────
const wave = (() => {
  let animId = null;
  let analyser = null;
  let dataArr = null;
  let phase = 0;

  const canvas = () => document.getElementById('waveCanvas');
  const W = 220, H = 48;

  function drawIdle() {
    const c = canvas(); if (!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const bars = 28;
    for (let i = 0; i < bars; i++) {
      const x = (i / bars) * W + W / bars / 2;
      const h = 6 + Math.sin(phase + i * 0.55) * 5 + Math.sin(phase * 1.3 + i * 0.9) * 3;
      const alpha = 0.25 + 0.15 * Math.sin(phase + i * 0.4);
      ctx.fillStyle = `rgba(203,166,247,${alpha})`;
      ctx.beginPath();
      ctx.roundRect(x - 2, (H - h) / 2, 3, h, 2);
      ctx.fill();
    }
    phase += 0.04;
    animId = requestAnimationFrame(drawIdle);
  }

  function drawRecording() {
    const c = canvas(); if (!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    analyser.getByteFrequencyData(dataArr);
    const bars = 28;
    const step = Math.floor(dataArr.length / bars);
    for (let i = 0; i < bars; i++) {
      const v = dataArr[i * step] / 255;
      const h = 4 + v * (H - 8);
      const g = ctx.createLinearGradient(0, (H - h) / 2, 0, (H + h) / 2);
      g.addColorStop(0, 'rgba(166,227,161,0.9)');
      g.addColorStop(1, 'rgba(166,227,161,0.3)');
      ctx.fillStyle = g;
      const x = (i / bars) * W + W / bars / 2;
      ctx.beginPath();
      ctx.roundRect(x - 2, (H - h) / 2, 3, h, 2);
      ctx.fill();
    }
    animId = requestAnimationFrame(drawRecording);
  }

  return {
    startIdle() {
      if (animId) cancelAnimationFrame(animId);
      drawIdle();
    },
    async startRecording(stream) {
      if (animId) cancelAnimationFrame(animId);
      const ctx2 = new AudioContext();
      const src = ctx2.createMediaStreamSource(stream);
      analyser = ctx2.createAnalyser();
      analyser.fftSize = 256;
      dataArr = new Uint8Array(analyser.frequencyBinCount);
      src.connect(analyser);
      drawRecording();
    },
    stopRecording() {
      if (animId) cancelAnimationFrame(animId);
      analyser = null; dataArr = null;
      this.startIdle();
    },
  };
})();

// Start idle wave on load
document.addEventListener('DOMContentLoaded', () => wave.startIdle());

// ── Stop TTS ────────────────────────────────────────────────────────────────
function stopTTS() {
  state.runId++;            // invalidates all in-flight onended callbacks
  state.audioQueue = [];
  state.isPlaying = false;
  document.getElementById('audioLabel').textContent = 'Stopped.';
}

// ── Panel summary helpers ───────────────────────────────────────────────────
function setEpiSummary(fields) {
  const el = document.getElementById('epiSummary');
  if (!el) return;
  const sp = (fields.species || []).slice(0, 2).join(', ') || '—';
  const sy = (fields.symptoms || []).slice(0, 2).join(', ') || '—';
  const mt = fields.mortality_count ? ` · ${fields.mortality_count} dead` : '';
  el.textContent = `${sp} · ${sy}${mt}`;
}

function setRouterSummary(domain, confidence) {
  const el = document.getElementById('routerSummary');
  if (!el) return;
  el.textContent = `${domain.replace(/_/g, ' ')} · ${(confidence * 100).toFixed(0)}%`;
}

function resetPipeline() {
  ['asr', 'ner', 'router', 'rag', 'llm', 'tts'].forEach(s => stepStatus(s, 'idle', 'waiting', ''));
}

function resetResults() {
  // Bump runId so any in-flight onended callbacks from old run are ignored
  state.runId++;
  state.audioQueue = [];
  state.isPlaying = false;

  // Hide previous results
  document.getElementById('assessmentSection').classList.add('hidden');
  document.getElementById('detailsSection').classList.add('hidden');
  document.getElementById('riskText').textContent = '';
  document.getElementById('riskBadge').textContent = '—';
  document.getElementById('riskBadge').className = 'risk-badge UNKNOWN';
  const flagsEl = document.getElementById('riskFlags');
  if (flagsEl) { flagsEl.innerHTML = ''; flagsEl.classList.add('hidden'); }
  document.getElementById('audioQueue').innerHTML = '';
}

// ── Recording ──────────────────────────────────────────────────────────────
async function toggleRecording() {
  if (state.isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    state.audioChunks = [];

    state.mediaRecorder.ondataavailable = e => state.audioChunks.push(e.data);
    state.mediaRecorder.onstop = () => {
      const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
      processAudio(blob);
      stream.getTracks().forEach(t => t.stop());
    };

    state.mediaRecorder.start(100);
    state.isRecording = true;

    document.getElementById('micBtn').classList.add('recording');
    document.getElementById('micIcon').textContent = '⏹';
    document.getElementById('micLabel').textContent = 'Recording...';
    setStatus('Recording...', 'busy');

    // Start real-time waveform
    wave.startRecording(stream);

  } catch (err) {
    alert('Microphone access denied. Please use the text input below.');
    console.error(err);
  }
}

function stopRecording() {
  if (state.mediaRecorder) {
    state.mediaRecorder.stop();
    state.isRecording = false;
    document.getElementById('micBtn').classList.remove('recording');
    document.getElementById('micIcon').textContent = '🎤';
    document.getElementById('micLabel').textContent = 'Processing…';
    wave.stopRecording(); // back to idle animation
  }
}

// ── Process audio blob → ASR ───────────────────────────────────────────────
async function processAudio(blob) {
  setStatus('Transcribing...', 'busy');
  stepStatus('asr', 'active', 'Transcribing audio...');

  const formData = new FormData();
  formData.append('audio', blob, 'recording.webm');

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    stepStatus('asr', 'done',
      `"${data.transcript.slice(0, 55)}${data.transcript.length > 55 ? '…' : ''}"`,
      `${data.asr_time}s`
    );

    document.getElementById('textInput').value = data.transcript;
    document.getElementById('micLabel').textContent = 'Tap to Record';

    // Auto-proceed to analysis
    await runAnalysis(data.transcript);

  } catch (err) {
    stepStatus('asr', 'idle', `Error: ${err.message}`);
    setStatus('ASR error', 'error');
    document.getElementById('micLabel').textContent = 'Hold to Record';
  }
}

// ── Analyze from text input ────────────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) return;
  await runAnalysis(text, false); // fromAudio=false — ASR is text input
}

// ── Main analysis pipeline ─────────────────────────────────────────────────
async function runAnalysis(transcript, fromAudio = true) {
  setStatus('Analyzing…', 'busy');
  document.getElementById('analyzeBtn').disabled = true;

  // Reset pipeline + old results
  resetPipeline();
  resetResults();

  // Show pipeline section
  document.getElementById('pipelineSection').classList.remove('hidden');

  // Set ASR step immediately after reset (so it sticks)
  if (fromAudio) {
    // ASR was already shown as 'done' by stopRecording — restore it
    stepStatus('asr', 'done', 'audio transcribed', '');
  } else {
    stepStatus('asr', 'done', 'text input  — no audio', '0.0s');
  }

  // Activate NER step
  stepStatus('ner', 'active', 'extracting fields…');
  stepStatus('router', 'idle', 'waiting');
  stepStatus('rag', 'idle', 'waiting');

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // NER done
    const fields = data.epi_fields;
    stepStatus('ner', 'done',
      `${(fields.species || []).join(', ') || '—'} · ${(fields.symptoms || []).slice(0, 2).join(', ') || '—'}`,
      `${data.timing.ner_s}s`
    );
    renderEpiFields(fields);
    setEpiSummary(fields); // update collapsed header summary

    // Router done
    stepStatus('router', 'done',
      `${data.domain} · ${(data.confidence * 100).toFixed(0)}%`,
      `${data.timing.route_s}s`
    );
    renderRouterBars(data.domain, data.all_scores);
    setRouterSummary(data.domain, data.confidence); // update collapsed header summary

    // RAG done
    stepStatus('rag', 'done',
      `${data.rag_chunks.length} chunks`,
      `${data.timing.rag_s}s`
    );
    renderRagChunks(data.rag_chunks);

    // Show details section
    document.getElementById('detailsSection').classList.remove('hidden');

    // Now stream the LLM response
    await streamResponse(data.domain, data.epi_fields, data.rag_chunks);

  } catch (err) {
    setStatus('Analysis error', 'error');
    stepStatus('ner', 'idle', `Error: ${err.message}`);
    console.error(err);
  } finally {
    document.getElementById('analyzeBtn').disabled = false;
  }
}

// ── SSE Stream: LLM + TTS ──────────────────────────────────────────────────
async function streamResponse(domain, epiFields, ragChunks) {
  const isChatMode = !!epiFields._off_topic;

  stepStatus('llm', 'active', isChatMode ? 'chatting…' : 'generating…');

  // Show assessment section
  document.getElementById('assessmentSection').classList.remove('hidden');
  document.getElementById('riskText').innerHTML = '<span class="cursor"></span>';
  document.getElementById('riskExpert').textContent = isChatMode ? 'ZoonoticSense Advisor' : domainToExpert(domain);
  document.getElementById('riskDomain').textContent = isChatMode ? 'chat' : domain.replace(/_/g, ' ');

  // Chat-mode tag
  const chatTag = document.getElementById('chatTag');
  chatTag.classList.toggle('hidden', !isChatMode);

  // Badge: hide for chat, reset for clinical
  const badge = document.getElementById('riskBadge');
  if (isChatMode) {
    badge.textContent = ''; badge.className = 'risk-badge hidden';
  } else {
    badge.textContent = '—'; badge.className = 'risk-badge UNKNOWN';
  }

  const flagsEl = document.getElementById('riskFlags');
  flagsEl.innerHTML = ''; flagsEl.classList.add('hidden');

  // TTS script list
  const ttsScriptList = document.getElementById('ttsScriptList');
  ttsScriptList.innerHTML = '';
  const ttsBody = document.getElementById('ttsScriptBody');
  ttsBody.classList.add('collapsed');
  const ttsToggleBtn = ttsBody.previousElementSibling;
  if (ttsToggleBtn) {
    ttsToggleBtn.setAttribute('data-collapsed', '');
    ttsToggleBtn.setAttribute('aria-expanded', 'false');
    const ch = ttsToggleBtn.querySelector('.toggle-chevron');
    if (ch) ch.textContent = '▸';
  }

  document.getElementById('audioQueue').innerHTML = '';
  document.getElementById('audioLabel').textContent = 'Synthesizing speech…';
  state.audioQueue = [];
  state.chunkIdx = 0;

  const t0 = Date.now();
  let fullText = '';

  const res = await fetch('/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ domain, epi_fields: epiFields, rag_chunks: ragChunks }),
  });

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let sseBuf = '';  // accumulate across TCP chunks — audio b64 is large!

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    sseBuf += dec.decode(value, { stream: true });

    // SSE events are delimited by double-newline
    let boundary;
    while ((boundary = sseBuf.indexOf('\n\n')) !== -1) {
      const raw = sseBuf.slice(0, boundary);
      sseBuf = sseBuf.slice(boundary + 2);

      for (const line of raw.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          const evt = JSON.parse(line.slice(6));

          if (evt.type === 'text') {
            fullText += ' ' + evt.chunk;
            document.getElementById('riskText').innerHTML =
              fullText.trim() + '<span class="cursor"></span>';
          }

          if (evt.type === 'audio') {
            const chunkN = evt.idx;
            stepStatus('tts', 'active', `chunk ${chunkN}`, '');
            document.getElementById('audioLabel').textContent = `Speaking chunk ${chunkN}…`;
            queueAudio(evt.data, chunkN);

            // Append to TTS script
            if (evt.sentence) {
              const li = document.createElement('li');
              li.className = 'tts-line';
              li.innerHTML = `<span class="tts-idx">${chunkN}</span><span class="tts-sent">${evt.sentence}</span>`;
              ttsScriptList.appendChild(li);
              // Auto-open script on first chunk
              if (chunkN === 1) {
                ttsBody.classList.remove('collapsed');
                ttsToggleBtn.removeAttribute('data-collapsed');
                ttsToggleBtn.setAttribute('aria-expanded', 'true');
                const ch = ttsToggleBtn.querySelector('.toggle-chevron');
                if (ch) ch.textContent = '▾';
              }
            }
          }

          if (evt.type === 'risk_card') {
            const card = evt.card;
            const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

            stepStatus('llm', 'done', isChatMode ? 'replied' : 'assessment complete', `${elapsed}s`);
            stepStatus('tts', 'done', 'all chunks done', '');

            document.getElementById('riskText').innerHTML = card.full_response;
            document.getElementById('audioLabel').textContent = 'Playback complete.';

            if (!isChatMode) {
              // Risk badge
              badge.className = `risk-badge ${card.risk_level}`;
              badge.textContent = card.risk_level;

              // Flags
              flagsEl.classList.remove('hidden');
              flagsEl.innerHTML = card.report_flag
                ? `<span class="risk-flag report">⚠ Report to Authorities</span>`
                : `<span class="risk-flag monitor">✓ Monitor</span>`;
            }

            addToHistory(card, fullText.trim());
            setStatus('Ready', 'ready');

            // Scroll to assessment
            document.getElementById('assessmentSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
          }

          if (evt.type === 'done') {
            // nothing
          }

        } catch (e) { /* ignore parse errors */ }
      }
    }
  }
} // end streamResponse

// ── Audio queue + sequential playback ─────────────────────────────────────
function queueAudio(base64Data, idx) {
  const blobUrl = URL.createObjectURL(base64ToBlob(base64Data, 'audio/wav'));
  const audioEl = new Audio(blobUrl);

  const btn = document.createElement('button');
  btn.className = 'audio-chunk';
  btn.textContent = idx;
  btn.id = `ac-${idx}`;
  document.getElementById('audioQueue').appendChild(btn);

  state.audioQueue.push({ audio: audioEl, btn, blobUrl });

  if (!state.isPlaying) playNextAudio();
}

function playNextAudio() {
  if (state.audioQueue.length === 0) {
    state.isPlaying = false;
    return;
  }
  state.isPlaying = true;

  const { audio, btn, blobUrl } = state.audioQueue.shift();
  const myRunId = state.runId; // close over current run ID
  btn.classList.add('playing');

  const advance = () => {
    btn.classList.remove('playing');
    btn.classList.add('done');
    URL.revokeObjectURL(blobUrl);
    if (state.runId !== myRunId) { state.isPlaying = false; return; }
    playNextAudio();
  };

  audio.onended = advance;

  const p = audio.play();
  if (p !== undefined) {
    p.catch(err => {
      console.warn('[TTS] play blocked:', err);
      advance(); // advance so queue never deadlocks
    });
  }
}

function base64ToBlob(base64, mimeType) {
  const bytes = atob(base64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mimeType });
}

// ── Render helpers ─────────────────────────────────────────────────────────
function renderEpiFields(fields) {
  const card = document.getElementById('epiCard');
  card.classList.remove('hidden');
  const grid = document.getElementById('epiGrid');
  grid.innerHTML = '';

  const rows = [
    ['species', 'Species', (fields.species || []).join(', ')],
    ['symptoms', 'Symptoms', (fields.symptoms || []).slice(0, 4).join(', ')],
    ['mortality', 'Mortality', fields.mortality_count],
    ['affected', 'Affected Count', fields.affected_count],
    ['location', 'Location', fields.location],
    ['timeframe', 'Timeframe', fields.timeframe],
    ['reporter', 'Reporter Role', fields.reporter_role],
    ['summary', 'Summary', fields.raw_summary],
  ];

  rows.forEach(([, label, value]) => {
    const isEmpty = !value || (Array.isArray(value) && value.length === 0);
    grid.innerHTML += `
      <div class="epi-field">
        <div class="epi-key">${label}</div>
        <div class="epi-val ${isEmpty ? 'empty' : ''}">${isEmpty ? 'not mentioned' : value}</div>
      </div>`;
  });
}

function renderRouterBars(selectedDomain, allScores) {
  const card = document.getElementById('routerCard');
  card.classList.remove('hidden');
  const container = document.getElementById('routerBars');
  container.innerHTML = '';

  const sorted = Object.entries(allScores).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([domain, score]) => {
    const isSelected = domain === selectedDomain;
    const pct = (score * 100).toFixed(1);
    container.innerHTML += `
      <div class="router-bar-row">
        <div class="router-domain ${isSelected ? 'selected' : ''}">${domain.replace('_', ' ')}</div>
        <div class="router-track">
          <div class="router-fill ${isSelected ? 'selected' : ''}" style="width: ${pct}%"></div>
        </div>
        <div class="router-score ${isSelected ? 'selected' : ''}">${pct}%</div>
      </div>`;
  });
}

function renderRagChunks(chunks) {
  const card = document.getElementById('ragCard');
  card.classList.remove('hidden');
  const container = document.getElementById('ragChunks');
  container.innerHTML = '';

  if (!chunks.length) {
    container.innerHTML = '<p style="color:var(--muted);font-size:12px">No chunks retrieved.</p>';
    return;
  }

  chunks.forEach((chunk, i) => {
    container.innerHTML += `
      <div class="rag-chunk">
        <div class="rag-chunk-num">SOURCE ${i + 1}</div>
        <div class="rag-chunk-text">${chunk.slice(0, 280)}${chunk.length > 280 ? '...' : ''}</div>
      </div>`;
  });
}

function addToHistory(card, text) {
  const list = document.getElementById('historyList');
  const empty = list.querySelector('.history-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `
    <div class="history-meta">
      <span class="history-domain">${card.domain.replace('_', ' ')}</span>
      <span class="history-risk ${card.risk_level}">${card.risk_level}</span>
    </div>
    <div class="history-text">${text.slice(0, 120)}${text.length > 120 ? '...' : ''}</div>`;
  list.insertBefore(item, list.firstChild);
}

function domainToExpert(domain) {
  const map = {
    avian_flu: 'Avian Influenza Expert',
    rabies: 'Rabies Surveillance Agent',
    fmd: 'FMD Field Specialist',
    nipah_hendra: 'Nipah/Hendra Analyst',
    leptospirosis: 'Leptospirosis Advisor',
    general: 'General Zoonotic Advisor',
  };
  return map[domain] || 'Expert Agent';
}

// ── Session reset ──────────────────────────────────────────────────────────
async function resetSession() {
  await fetch('/reset', { method: 'POST' });
  document.getElementById('historyList').innerHTML = '<div class="history-empty">No reports yet. Record or type a field report above.</div>';
  document.getElementById('pipelineSection').classList.add('hidden');
  document.getElementById('assessmentSection').classList.add('hidden');
  document.getElementById('detailsSection').classList.add('hidden');
  document.getElementById('epiCard').classList.add('hidden');
  document.getElementById('routerCard').classList.add('hidden');
  document.getElementById('ragCard').classList.add('hidden');
  document.getElementById('textInput').value = '';
  resetPipeline();
  setStatus('Ready', 'ready');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Enter key on textarea ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('textInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) analyzeText();
  });
});

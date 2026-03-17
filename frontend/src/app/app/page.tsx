'use client'
import { useState, useCallback, useRef } from 'react'
import { useRouter } from 'next/navigation'
import MicButton from '@/components/MicButton'
import { BACKEND } from '@/lib/config'
import PipelineBar, { PipelineStep } from '@/components/PipelineBar'
import AssessmentCard from '@/components/AssessmentCard'
import EpiFields from '@/components/EpiFields'
import RouterScores from '@/components/RouterScores'
import RagSources from '@/components/RagSources'
import styles from './app.module.css'

const INITIAL_STEPS: PipelineStep[] = [
  { id: 'asr',    label: 'ASR',    detail: 'waiting', time: '', status: 'idle' },
  { id: 'ner',    label: 'NER',    detail: 'waiting', time: '', status: 'idle' },
  { id: 'router', label: 'Router', detail: 'waiting', time: '', status: 'idle' },
  { id: 'rag',    label: 'RAG',    detail: 'waiting', time: '', status: 'idle' },
  { id: 'llm',    label: 'Expert', detail: 'waiting', time: '', status: 'idle' },
  { id: 'tts',    label: 'TTS',    detail: 'waiting', time: '', status: 'idle' },
]

interface AnalyzeResult {
  epi_fields: Record<string, unknown>
  domain: string
  confidence: number
  all_scores: Record<string, number>
  rag_chunks: string[]
  timing: { ner_s: number; route_s: number; rag_s: number }
}

interface HistoryItem {
  domain: string
  risk_level: string
  text: string
}

function updateStep(steps: PipelineStep[], id: string, status: PipelineStep['status'], detail?: string, time?: string): PipelineStep[] {
  return steps.map(s => s.id === id ? { ...s, status, detail: detail ?? s.detail, time: time ?? s.time } : s)
}

export default function AppPage() {
  const router = useRouter()
  const [transcript, setTranscript] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dropLabel, setDropLabel] = useState('Drop audio / Click')
  const [showText, setShowText] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [status, setStatus] = useState<'ready' | 'busy' | 'error'>('ready')
  const [statusText, setStatusText] = useState('Ready')
  const [steps, setSteps] = useState<PipelineStep[]>(INITIAL_STEPS)
  const [running, setRunning] = useState(false)
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResult | null>(null)
  const [showAssessment, setShowAssessment] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [busy, setBusy] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])

  function resetAll() {
    setSteps(INITIAL_STEPS)
    setAnalyzeResult(null)
    setShowAssessment(false)
    setShowDetails(false)
    setRunning(false)
  }

  function setStep(id: string, status: PipelineStep['status'], detail?: string, time?: string) {
    setSteps(prev => updateStep(prev, id, status, detail, time))
  }

  async function runAnalysis(text: string, fromAudio = false) {
    if (!text.trim()) return
    setBusy(true)
    setStatus('busy')
    setStatusText('Analyzing…')
    resetAll()
    setRunning(true)

    if (fromAudio) {
      setStep('asr', 'done', 'audio transcribed')
    } else {
      setStep('asr', 'done', 'text input — no audio', '0.0s')
    }
    setStep('ner', 'active', 'extracting fields…')

    try {
      const res = await fetch(`${BACKEND}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: text }),
      })
      const data: AnalyzeResult & { error?: string } = await res.json()
      if (data.error) throw new Error(data.error)

      const fields = data.epi_fields
      const isOffTopic = !!(fields._off_topic)

      setStep('ner', 'done',
        `${(fields.species as string[] || []).join(', ') || '—'} · ${(fields.symptoms as string[] || []).slice(0,2).join(', ') || '—'}`,
        `${data.timing.ner_s}s`
      )

      if (isOffTopic) {
        setStep('router', 'done', 'chat mode', `${data.timing.route_s}s`)
      } else {
        setStep('router', 'done',
          `${data.domain} · ${(data.confidence * 100).toFixed(0)}%`,
          `${data.timing.route_s}s`
        )
      }

      setStep('rag', 'done', `${data.rag_chunks.length} chunks`, `${data.timing.rag_s}s`)
      setStep('llm', 'active', isOffTopic ? 'chatting…' : 'generating…')

      setAnalyzeResult(data)
      setShowDetails(true)
      setShowAssessment(true)

    } catch (e: unknown) {
      setStatus('error')
      setStatusText('Analysis error')
      setStep('ner', 'idle', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    } finally {
      setBusy(false)
    }
  }

  const handleTranscript = useCallback((text: string, asrTime: number) => {
    setTranscript(text)
    setShowText(true)
    setStep('asr', 'done', `"${text.slice(0, 50)}${text.length > 50 ? '…' : ''}"`, `${asrTime}s`)
    runAnalysis(text, true)
  }, [])

  const handleDone = useCallback(() => {
    setStatus('ready')
    setStatusText('Ready')
    setRunning(false)
  }, [])

  const handleStepUpdate = useCallback((id: string, st: string, detail?: string, time?: string) => {
    setStep(id, st as PipelineStep['status'], detail, time)
  }, [])

  async function uploadFile(file: File) {
    setUploading(true)
    setDropLabel(file.name.slice(0, 22) + (file.name.length > 22 ? '…' : ''))
    setStep('asr', 'active', 'Transcribing audio...')
    setRunning(true)
    const formData = new FormData()
    formData.append('audio', file, file.name)
    try {
      const res = await fetch(`${BACKEND}/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setTranscript(data.transcript)
      setShowText(true)
      setStep('asr', 'done', `"${data.transcript.slice(0, 50)}${data.transcript.length > 50 ? '…' : ''}"`, `${data.asr_time}s`)
      await runAnalysis(data.transcript, true)
    } catch (e: unknown) {
      setStatus('error')
      setStatusText(e instanceof Error ? e.message : 'Upload failed')
      setStep('asr', 'idle', 'error')
    } finally {
      setUploading(false)
      setDropLabel('Drop audio / Click')
    }
  }

  async function handleReset() {
    await fetch(`${BACKEND}/reset`, { method: 'POST' })
    setTranscript('')
    setHistory([])
    resetAll()
    setStatus('ready')
    setStatusText('Ready')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={styles.logo}>Zx</span>
          <div>
            <div className={styles.appName}>ZoonoMoE</div>
            <div className={styles.appSub}>Voice Zoonotic Surveillance · v2.0</div>
          </div>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.statusRow}>
            <span className={`${styles.dot} ${styles[status]}`} />
            <span className={styles.statusText}>{statusText}</span>
          </div>
          <button className={styles.navBtn} onClick={() => router.push('/')}>Home</button>
          <button className={styles.navBtn} onClick={handleReset}>New Session</button>
        </div>
      </header>

      <main className={styles.main}>
        {/* Input section */}
        <section className={styles.hero}>
          <MicButton
            onTranscript={handleTranscript}
            onError={msg => { setStatus('error'); setStatusText(msg) }}
            disabled={busy}
          />
          <div
            className={`${styles.dropZone} ${dragOver ? styles.dragOver : ''} ${uploading ? styles.uploading : ''}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) uploadFile(f) }}
            onClick={() => fileInputRef.current?.click()}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span>{dropLabel}</span>
            <input ref={fileInputRef} type="file" accept=".m4a,.wav,.mp3,.webm,.ogg,audio/*" style={{display:'none'}} onChange={e => { const f = e.target.files?.[0]; e.target.value=''; if (f) uploadFile(f) }} />
          </div>
          <button
            className={`${styles.textToggle} ${showText ? styles.textToggleOpen : ''}`}
            onClick={() => setShowText(v => !v)}
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points={showText ? '18 15 12 9 6 15' : '6 9 12 15 18 9'} />
            </svg>
            <span>{showText ? 'Hide text' : 'Type / View transcript'}</span>
          </button>

          {showText && (
            <div className={styles.textRow}>
              <textarea
                className={styles.textarea}
                value={transcript}
                onChange={e => setTranscript(e.target.value)}
                placeholder="Or type a field report here…&#10;e.g. 30 chickens died this morning with purple combs and twisted necks"
                rows={3}
                onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) runAnalysis(transcript) }}
                autoFocus
              />
              <button
                className={styles.analyzeBtn}
                onClick={() => runAnalysis(transcript)}
                disabled={busy || !transcript.trim()}
              >
                Analyze
              </button>
            </div>
          )}
        </section>

        {/* Pipeline */}
        {(running || steps.some(s => s.status !== 'idle')) && (
          <section className={styles.section}>
            <PipelineBar steps={steps} running={running} />
          </section>
        )}

        {/* Assessment */}
        {showAssessment && analyzeResult && (
          <section className={styles.section}>
            <div className={styles.sectionLabel}>Assessment</div>
            <AssessmentCard
              domain={analyzeResult.domain}
              epiFields={analyzeResult.epi_fields}
              ragChunks={analyzeResult.rag_chunks}
              transcript={transcript}
              timing={analyzeResult.timing}
              onStepUpdate={handleStepUpdate}
              onDone={handleDone}
            />
          </section>
        )}

        {/* Detail panels */}
        {showDetails && analyzeResult && (
          <section className={styles.section}>
            <div className={styles.detailGrid}>
              {/* EPI Fields */}
              <details className={styles.panel} open>
                <summary className={styles.panelSummary}>
                  🧬 Extracted Fields
                  <span className={styles.panelMeta}>
                    {(analyzeResult.epi_fields.species as string[] || []).slice(0,2).join(', ') || '—'}
                  </span>
                </summary>
                <div className={styles.panelBody}>
                  <EpiFields fields={analyzeResult.epi_fields} />
                </div>
              </details>

              {/* Router scores */}
              {!(analyzeResult.epi_fields._off_topic) && (
                <details className={styles.panel} open>
                  <summary className={styles.panelSummary}>
                    🚦 Domain Confidence
                    <span className={styles.panelMeta}>
                      {analyzeResult.domain.replace(/_/g,' ')} · {(analyzeResult.confidence*100).toFixed(0)}%
                    </span>
                  </summary>
                  <div className={styles.panelBody}>
                    <RouterScores domain={analyzeResult.domain} scores={analyzeResult.all_scores} />
                  </div>
                </details>
              )}

              {/* RAG Sources */}
              <details className={styles.panel}>
                <summary className={styles.panelSummary}>
                  📚 Knowledge Sources
                  <span className={styles.panelMeta}>{analyzeResult.rag_chunks.length} chunks</span>
                </summary>
                <div className={styles.panelBody}>
                  <RagSources chunks={analyzeResult.rag_chunks} />
                </div>
              </details>
            </div>
          </section>
        )}

        {/* History */}
        <section className={styles.section}>
          <details className={styles.panel}>
            <summary className={styles.panelSummary}>📋 Session History</summary>
            <div className={styles.panelBody}>
              {history.length === 0
                ? <p className={styles.historyEmpty}>No reports yet.</p>
                : history.map((h, i) => (
                  <div key={i} className={styles.historyItem}>
                    <div className={styles.historyMeta}>
                      <span className={styles.historyDomain}>{h.domain.replace('_',' ')}</span>
                      <span className={`risk-badge ${h.risk_level}`}>{h.risk_level}</span>
                    </div>
                    <div className={styles.historyText}>{h.text.slice(0,120)}{h.text.length > 120 ? '…' : ''}</div>
                  </div>
                ))
              }
            </div>
          </details>
        </section>

        {/* Emergency footer */}
        <footer className={styles.footer}>
          <span>🚨 Emergency: DLD <strong>02-653-4444</strong></span>
          <span>DDC Hotline <strong>1422</strong></span>
        </footer>
      </main>
    </div>
  )
}

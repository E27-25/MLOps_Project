'use client'
import { useEffect, useRef, useState } from 'react'
import styles from './AssessmentCard.module.css'
import { BACKEND } from '@/lib/config'

export interface RiskCard {
  risk_level: string
  report_flag: boolean
  domain: string
  expert_name: string
  full_response: string
  is_chat: boolean
}

interface Props {
  domain: string
  epiFields: Record<string, unknown>
  ragChunks: string[]
  transcript?: string
  timing?: Record<string, number>
  onStepUpdate: (step: string, status: string, detail?: string, time?: string) => void
  onDone: () => void
}

function mdToHtml(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/^[-•]\s+(.+)/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, ' ')
}

export default function AssessmentCard({ domain, epiFields, ragChunks, transcript, timing, onStepUpdate, onDone }: Props) {
  const [text, setText] = useState('')
  const [streaming, setStreaming] = useState(true)
  const [riskCard, setRiskCard] = useState<RiskCard | null>(null)
  const [audioLabel, setAudioLabel] = useState('Synthesizing speech…')
  const [paused, setPaused] = useState(false)
  const [ttsLines, setTtsLines] = useState<{ idx: number; sentence: string }[]>([])

  const audioQueueRef = useRef<{ audio: HTMLAudioElement; idx: number; blobUrl: string }[]>([])
  const isPlayingRef = useRef(false)
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)
  const runIdRef = useRef(0)
  const isChatMode = !!(epiFields._off_topic)

  function base64ToBlob(b64: string, mime: string) {
    const bytes = atob(b64)
    const arr = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i)
    return new Blob([arr], { type: mime })
  }

  function playNext() {
    if (audioQueueRef.current.length === 0) { isPlayingRef.current = false; return }
    isPlayingRef.current = true
    const { audio, blobUrl } = audioQueueRef.current.shift()!
    const myRunId = runIdRef.current
    currentAudioRef.current = audio
    audio.onended = () => {
      URL.revokeObjectURL(blobUrl)
      if (runIdRef.current !== myRunId) { isPlayingRef.current = false; return }
      playNext()
    }
    audio.play().catch(() => playNext())
  }

  function queueAudio(b64: string, idx: number) {
    const blobUrl = URL.createObjectURL(base64ToBlob(b64, 'audio/wav'))
    const audio = new Audio(blobUrl)
    audioQueueRef.current.push({ audio, idx, blobUrl })
    if (!isPlayingRef.current) playNext()
  }

  function toggleTTS() {
    const audio = currentAudioRef.current
    if (!audio) return
    if (audio.paused) {
      audio.play()
      isPlayingRef.current = true
      setPaused(false)
      setAudioLabel('Playing…')
    } else {
      audio.pause()
      isPlayingRef.current = false
      setPaused(true)
      setAudioLabel('Paused.')
    }
  }

  useEffect(() => {
    runIdRef.current++
    let fullText = ''
    let sseBuf = ''

    async function stream() {
      const res = await fetch(`${BACKEND}/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain, epi_fields: epiFields, rag_chunks: ragChunks, transcript: transcript ?? '', timing: timing ?? {} }),
      })
      const reader = res.body!.getReader()
      const dec = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        sseBuf += dec.decode(value, { stream: true })
        let boundary: number
        while ((boundary = sseBuf.indexOf('\n\n')) !== -1) {
          const raw = sseBuf.slice(0, boundary)
          sseBuf = sseBuf.slice(boundary + 2)
          for (const line of raw.split('\n')) {
            if (!line.startsWith('data: ')) continue
            try {
              const evt = JSON.parse(line.slice(6))
              if (evt.type === 'text') {
                fullText += ' ' + evt.chunk
                setText(fullText.trim())
              }
              if (evt.type === 'audio') {
                onStepUpdate('tts', 'active', `chunk ${evt.idx}`)
                setAudioLabel(`Speaking chunk ${evt.idx}…`)
                queueAudio(evt.data, evt.idx)
                if (evt.sentence) {
                  setTtsLines(prev =>
                    prev.some(x => x.idx === evt.idx) ? prev : [...prev, { idx: evt.idx, sentence: evt.sentence }]
                  )
                }
              }
              if (evt.type === 'risk_card') {
                setRiskCard(evt.card)
                setStreaming(false)
                setAudioLabel('Playback complete.')
                onStepUpdate('llm', 'done', isChatMode ? 'replied' : 'assessment complete')
                onStepUpdate('tts', 'done', 'all chunks done')
                onDone()
              }
            } catch { /* ignore */ }
          }
        }
      }
    }
    stream().catch(console.error)
  }, [])

  const riskLevel = riskCard?.risk_level ?? 'UNKNOWN'
  const showBadge = riskCard && !riskCard.is_chat && riskLevel !== 'NONE'

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div>
          <div className={styles.expert}>
            {isChatMode ? 'ZoonoticSense Advisor' : riskCard?.expert_name ?? domainToExpert(domain)}
          </div>
          <div className={styles.domainRow}>
            <span className={styles.domain}>{isChatMode ? 'chat' : domain.replace(/_/g, ' ')}</span>
            {isChatMode && <span className={styles.chatTag}>💬 chat mode</span>}
          </div>
        </div>
        {showBadge && (
          <span className={`risk-badge ${riskLevel}`}>{riskLevel}</span>
        )}
      </div>

      <div
        className={styles.text}
        dangerouslySetInnerHTML={{
          __html: text
            ? '<p>' + mdToHtml(text) + '</p>' + (streaming ? '<span class="cursor"></span>' : '')
            : streaming ? '<span class="cursor"></span>' : ''
        }}
      />

      {riskCard && !riskCard.is_chat && riskCard.report_flag !== undefined && (
        <div className={styles.flag}>
          {riskCard.report_flag
            ? <span className={styles.flagReport}>⚠ Report to Authorities</span>
            : <span className={styles.flagMonitor}>✓ Monitor</span>}
        </div>
      )}

      {/* TTS Controls */}
      <div className={styles.ttsBar}>
        <span className={styles.ttsLabel}>{audioLabel}</span>
        <button className={styles.ttsBtn} onClick={toggleTTS}>
          {paused ? '▶ Resume' : '⏸ Pause'}
        </button>
      </div>

      {/* TTS Script */}
      {ttsLines.length > 0 && (
        <details className={styles.ttsScript}>
          <summary className={styles.ttsSummary}>TTS Script ({ttsLines.length} sentences)</summary>
          <ol className={styles.ttsLines}>
            {ttsLines.map(l => (
              <li key={l.idx} className={styles.ttsLine}>
                <span className={styles.ttsIdx}>{l.idx}</span>
                <span>{l.sentence}</span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  )
}

function domainToExpert(domain: string) {
  const map: Record<string, string> = {
    avian_flu: 'Avian Influenza Expert',
    rabies: 'Rabies Surveillance Agent',
    fmd: 'FMD Field Specialist',
    nipah_hendra: 'Nipah/Hendra Analyst',
    leptospirosis: 'Leptospirosis Advisor',
    general: 'General Zoonotic Advisor',
  }
  return map[domain] ?? 'Expert Agent'
}

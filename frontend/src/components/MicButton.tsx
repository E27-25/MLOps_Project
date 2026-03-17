'use client'
import { useEffect, useRef, useState } from 'react'
import styles from './MicButton.module.css'
import { BACKEND } from '@/lib/config'

interface Props {
  onTranscript: (text: string, asrTime: number) => void
  onError: (msg: string) => void
  disabled: boolean
}

export default function MicButton({ onTranscript, onError, disabled }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [recording, setRecording] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const animRef = useRef<number>(0)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const dataArrRef = useRef<Uint8Array<ArrayBuffer> | null>(null)
  const phaseRef = useRef(0)

  const W = 220, H = 48

  function drawIdle() {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, W, H)
    const bars = 28
    for (let i = 0; i < bars; i++) {
      const x = (i / bars) * W + W / bars / 2
      const h = 6 + Math.sin(phaseRef.current + i * 0.55) * 5 + Math.sin(phaseRef.current * 1.3 + i * 0.9) * 3
      const alpha = 0.25 + 0.15 * Math.sin(phaseRef.current + i * 0.4)
      ctx.fillStyle = `rgba(203,166,247,${alpha})`
      ctx.beginPath()
      ctx.roundRect(x - 2, (H - h) / 2, 3, h, 2)
      ctx.fill()
    }
    phaseRef.current += 0.04
    animRef.current = requestAnimationFrame(drawIdle)
  }

  function drawRecording() {
    const c = canvasRef.current
    if (!c || !analyserRef.current || !dataArrRef.current) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, W, H)
    analyserRef.current.getByteFrequencyData(dataArrRef.current)
    const bars = 28
    const step = Math.floor(dataArrRef.current.length / bars)
    for (let i = 0; i < bars; i++) {
      const v = dataArrRef.current[i * step] / 255
      const h = 4 + v * (H - 8)
      const g = ctx.createLinearGradient(0, (H - h) / 2, 0, (H + h) / 2)
      g.addColorStop(0, 'rgba(166,227,161,0.9)')
      g.addColorStop(1, 'rgba(166,227,161,0.3)')
      ctx.fillStyle = g
      const x = (i / bars) * W + W / bars / 2
      ctx.beginPath()
      ctx.roundRect(x - 2, (H - h) / 2, 3, h, 2)
      ctx.fill()
    }
    animRef.current = requestAnimationFrame(drawRecording)
  }

  useEffect(() => {
    animRef.current = requestAnimationFrame(drawIdle)
    return () => cancelAnimationFrame(animRef.current)
  }, [])

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []
      mr.ondataavailable = e => chunksRef.current.push(e.data)
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(t => t.stop())
        processAudio(blob)
      }
      mr.start(100)
      mediaRecorderRef.current = mr
      setRecording(true)

      // Waveform
      const actx = new AudioContext()
      const src = actx.createMediaStreamSource(stream)
      const analyser = actx.createAnalyser()
      analyser.fftSize = 256
      analyserRef.current = analyser
      dataArrRef.current = new Uint8Array(analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>
      src.connect(analyser)
      cancelAnimationFrame(animRef.current)
      animRef.current = requestAnimationFrame(drawRecording)
    } catch {
      onError('Microphone access denied. Please use the text input below.')
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop()
    setRecording(false)
    analyserRef.current = null
    dataArrRef.current = null
    cancelAnimationFrame(animRef.current)
    animRef.current = requestAnimationFrame(drawIdle)
  }

  async function processAudio(blob: Blob) {
    const formData = new FormData()
    formData.append('audio', blob, 'recording.webm')
    try {
      const res = await fetch(`${BACKEND}/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      onTranscript(data.transcript, data.asr_time)
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : 'Upload failed')
    }
  }

  function toggle() {
    if (recording) stopRecording()
    else startRecording()
  }

  return (
    <div className={styles.wrap}>
      <button
        className={`${styles.micBtn} ${recording ? styles.recording : ''}`}
        onClick={toggle}
        disabled={disabled}
        aria-label={recording ? 'Stop recording' : 'Start recording'}
      >
        {recording ? (
          <svg viewBox="0 0 24 24" width="24" height="24" fill="white">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="white" strokeWidth="2">
            <rect x="9" y="2" width="6" height="12" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0" />
            <line x1="12" y1="19" x2="12" y2="22" />
            <line x1="8" y1="22" x2="16" y2="22" />
          </svg>
        )}
      </button>
      <canvas ref={canvasRef} width={W} height={H} className={styles.wave} />
      <span className={styles.label}>
        {recording ? 'Recording…' : 'Tap to Record'}
      </span>
    </div>
  )
}

'use client'
import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import styles from './landing.module.css'
import ParticleBackground, { type MouseState } from '../components/ParticleBackground'

const PIPELINE = [
  { name: 'ASR',    icon: '🎤', desc: 'Whisper speech recognition',   color: 'sapph' },
  { name: 'NER',    icon: '🧬', desc: 'Clinical entity extraction',    color: 'mauve' },
  { name: 'Router', icon: '🧭', desc: 'Domain classification',         color: 'yellow' },
  { name: 'RAG',    icon: '📚', desc: 'Vector knowledge retrieval',    color: 'peach' },
  { name: 'TTS',    icon: '🔊', desc: 'Streaming voice synthesis',     color: 'green' },
]

const PIPELINE_INFO: Record<string, string> = {
  ASR:    'Whisper (MLX) — converts your voice to text in under 2 seconds.',
  NER:    'Qwen3-4B-4bit — extracts species, symptoms, mortality and location with a structured JSON prompt.',
  Router: 'Sentence-BERT fine-tuned on 200+ zoonotic examples — routes to the correct disease domain.',
  RAG:    'FAISS vector search over per-domain vet documents — 3 top passages retrieved and ranked.',
  TTS:    "Kokoro TTS — streams audio sentence-by-sentence so you hear the result as it's generated.",
}

const FEATURES = [
  { icon: '🎤', title: 'Voice-First Input',  badge: 'MLX WHISPER',          desc: 'Record directly in noisy field conditions. MLX Whisper transcribes in real time even with accents and background noise.' },
  { icon: '🧬', title: 'Clinical NER',       badge: 'QWEN3 4B-INT4',        desc: 'Extracts species, symptoms, mortality count, location and timeframe from natural language field reports.' },
  { icon: '🧭', title: 'Smart Routing',      badge: 'F1 0.83 · 6 DOMAINS',  desc: 'Avian flu, FMD, Nipah, Rabies, Leptospirosis, General — the right specialist knowledge is selected automatically.' },
  { icon: '📚', title: 'RAG Knowledge',      badge: 'FAISS · SENTENCE-BERT', desc: 'Per-domain FAISS vector stores ground every LLM response in verified veterinary documentation.' },
  { icon: '🔊', title: 'Streaming TTS',      badge: 'KOKORO TTS',            desc: 'Assessment is synthesised and spoken sentence-by-sentence as the model generates it — no waiting for the full response.' },
  { icon: '🔒', title: 'Fully Local',        badge: 'APPLE MLX · ON-DEVICE', desc: 'Every model runs on-device with Apple MLX acceleration on Apple Silicon. No API keys. No data leaves the machine.' },
]

const STATS: [string, string][] = [
  ['6', 'Disease Domains'],
  ['F1\u00a00.83', 'Router Accuracy'],
  ['<20s', 'Full Pipeline'],
  ['100%', 'On-Device Local'],
]

export default function LandingPage() {
  const router    = useRouter()
  const blastRef  = useRef<HTMLCanvasElement>(null)
  const orbARef   = useRef<HTMLDivElement>(null)
  const orbBRef   = useRef<HTMLDivElement>(null)
  const orbCRef   = useRef<HTMLDivElement>(null)
  const [navScrolled, setNavScrolled] = useState(false)
  const [typeText,    setTypeText]    = useState('')
  const [tooltip,     setTooltip]    = useState<string | null>(null)

  // Shared mouse state — read by ParticleBackground (GPU) + onFrame callback
  const mouseRef  = useRef<MouseState>({ x: -9999, y: -9999, down: false, clicked: false, cx: -9999, cy: -9999 })
  // Blast state — mutated by click handler, read by onFrame (no re-renders)
  type BlastRing  = { speed: number; color: string; lw: number; life: number; r: number }
  type BlastSpark = { x: number; y: number; vx: number; vy: number; life: number; len: number; col: [number,number,number] }
  type BlastObj   = { x: number; y: number; rings: BlastRing[]; sparks: BlastSpark[]; flash: { r: number; life: number } }
  const blastsRef = useRef<BlastObj[]>([])

  // ── Mouse tracking + blast spawn (NO RAF loop here) ─────
  useEffect(() => {
    if (!blastRef.current) return
    const el = blastRef.current
    const resize = () => { el.width = window.innerWidth; el.height = window.innerHeight }
    resize()
    window.addEventListener('resize', resize, { passive: true })

    const onMM = (e: MouseEvent) => { mouseRef.current.x = e.clientX; mouseRef.current.y = e.clientY }
    const onMD = (e: MouseEvent) => {
      mouseRef.current.down = true; mouseRef.current.clicked = true
      mouseRef.current.cx = e.clientX; mouseRef.current.cy = e.clientY
      mouseRef.current.x  = e.clientX; mouseRef.current.y  = e.clientY
    }
    const onMU = () => { mouseRef.current.down = false }
    const onTM = (e: TouchEvent) => { if (e.touches[0]) { mouseRef.current.x = e.touches[0].clientX; mouseRef.current.y = e.touches[0].clientY } }
    const onTS = (e: TouchEvent) => { if (e.touches[0]) { mouseRef.current.x = e.touches[0].clientX; mouseRef.current.y = e.touches[0].clientY; mouseRef.current.clicked = true } }
    document.addEventListener('mousemove',  onMM)
    document.addEventListener('mousedown',  onMD)
    document.addEventListener('mouseup',    onMU)
    document.addEventListener('touchmove',  onTM, { passive: true })
    document.addEventListener('touchstart', onTS, { passive: true })

    function rnd(a: number, b: number) { return a + Math.random() * (b - a) }
    const onClickBlast = (e: MouseEvent) => {
      const rings: BlastRing[] = [{ speed:4.5,color:'203,166,247',lw:1.5,life:1.0,r:0 },{ speed:3.2,color:'116,199,236',lw:1.0,life:0.8,r:0 },{ speed:2.0,color:'137,180,250',lw:0.6,life:0.6,r:0 }]
      const COLS: [number,number,number][] = [[203,166,247],[137,180,250],[116,199,236],[166,227,161]]
      const sparks: BlastSpark[] = Array.from({ length: 16 }, (_, i) => {
        const ang = (i / 16) * Math.PI * 2 + rnd(-0.2, 0.2), spd = rnd(2.5, 7), col = COLS[Math.floor(Math.random() * COLS.length)]
        return { x: e.clientX, y: e.clientY, vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd, life: 1, len: rnd(6, 18), col }
      })
      blastsRef.current.push({ x: e.clientX, y: e.clientY, rings, sparks, flash: { r: 0, life: 1 } })
    }
    document.addEventListener('click', onClickBlast)

    return () => {
      window.removeEventListener('resize', resize)
      document.removeEventListener('mousemove', onMM); document.removeEventListener('mousedown', onMD)
      document.removeEventListener('mouseup',   onMU); document.removeEventListener('touchmove', onTM)
      document.removeEventListener('touchstart', onTS); document.removeEventListener('click', onClickBlast)
    }
  }, [])

  // ── onFrame: draws cursor glow + blasts in R3F's RAF ────
  const onFrameRef = useRef<() => void>(() => {})
  useEffect(() => {
    onFrameRef.current = () => {
      const el  = blastRef.current; if (!el) return
      const ctx = el.getContext('2d'); if (!ctx) return
      ctx.clearRect(0, 0, el.width, el.height)
      const mx = mouseRef.current.x, my = mouseRef.current.y
      if (mx > 0) {
        const g = ctx.createRadialGradient(mx, my, 0, mx, my, 80)
        g.addColorStop(0, 'rgba(203,166,247,0.10)'); g.addColorStop(1, 'rgba(203,166,247,0)')
        ctx.beginPath(); ctx.arc(mx, my, 80, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill()
      }
      const blasts = blastsRef.current
      for (let b = blasts.length - 1; b >= 0; b--) {
        const { x, y, rings, sparks, flash } = blasts[b]; let alive = false
        if (flash.life > 0) {
          flash.r += 8; flash.life -= 0.12
          if (flash.life > 0) {
            const fg = ctx.createRadialGradient(x, y, 0, x, y, flash.r)
            fg.addColorStop(0, `rgba(255,255,255,${flash.life * 0.5})`); fg.addColorStop(0.4, `rgba(203,166,247,${flash.life * 0.3})`); fg.addColorStop(1, 'rgba(203,166,247,0)')
            ctx.beginPath(); ctx.arc(x, y, flash.r, 0, Math.PI * 2); ctx.fillStyle = fg; ctx.fill(); alive = true
          }
        }
        rings.forEach(rg => {
          if (rg.life <= 0) return; rg.r += rg.speed; rg.life -= 0.028
          if (rg.life > 0) { ctx.beginPath(); ctx.arc(x, y, rg.r, 0, Math.PI * 2); ctx.strokeStyle = `rgba(${rg.color},${rg.life * 0.7})`; ctx.lineWidth = rg.lw; ctx.stroke(); alive = true }
        })
        sparks.forEach(sp => {
          if (sp.life <= 0) return; sp.vx *= 0.93; sp.vy *= 0.93; sp.vy += 0.12; sp.x += sp.vx; sp.y += sp.vy; sp.life -= 0.025
          if (sp.life > 0) {
            const [r2,g2,b2] = sp.col, tx2 = sp.x - sp.vx * sp.len * 0.5, ty2 = sp.y - sp.vy * sp.len * 0.5
            const sg = ctx.createLinearGradient(tx2, ty2, sp.x, sp.y)
            sg.addColorStop(0, `rgba(${r2},${g2},${b2},0)`); sg.addColorStop(1, `rgba(${r2},${g2},${b2},${sp.life})`)
            ctx.beginPath(); ctx.moveTo(tx2, ty2); ctx.lineTo(sp.x, sp.y); ctx.strokeStyle = sg; ctx.lineWidth = 1.5; ctx.stroke()
            ctx.beginPath(); ctx.arc(sp.x, sp.y, 1.5, 0, Math.PI * 2); ctx.fillStyle = `rgba(255,255,255,${sp.life * 0.9})`; ctx.fill(); alive = true
          }
        })
        if (!alive) blasts.splice(b, 1)
      }
    }
  }, [])

  // ── Aurora orbs lerp ────────────────────────────────────
  useEffect(() => {
    if (!orbARef.current || !orbBRef.current || !orbCRef.current) return
    const orbA = orbARef.current, orbB = orbBRef.current, orbC = orbCRef.current
    let mx = window.innerWidth / 2, my = window.innerHeight / 2, lx = mx, ly = my
    const onMove = (e: MouseEvent) => { mx = e.clientX; my = e.clientY }
    document.addEventListener('mousemove', onMove)
    let id: number
    function animate() {
      lx += (mx - lx) * 0.04; ly += (my - ly) * 0.04
      const ox = (lx / window.innerWidth - 0.5) * 100, oy = (ly / window.innerHeight - 0.5) * 70
      orbA.style.transform = `translate(${ox * 0.5}px,${oy * 0.5}px)`
      orbB.style.transform = `translate(${-ox * 0.4}px,${-oy * 0.35}px)`
      orbC.style.transform = `translate(calc(-50% + ${ox * 0.25}px),calc(-50% + ${oy * 0.25}px))`
      id = requestAnimationFrame(animate)
    }
    animate()
    return () => { cancelAnimationFrame(id); document.removeEventListener('mousemove', onMove) }
  }, [])

  // ── Nav scroll ──────────────────────────────────────────
  useEffect(() => {
    const fn = () => setNavScrolled(window.scrollY > 40)
    window.addEventListener('scroll', fn, { passive: true })
    return () => window.removeEventListener('scroll', fn)
  }, [])

  // ── Scroll reveal ───────────────────────────────────────
  useEffect(() => {
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('landing-visible'); io.unobserve(e.target) } })
    }, { threshold: 0.1 })
    document.querySelectorAll('.landing-reveal').forEach(el => io.observe(el))
    return () => io.disconnect()
  }, [])

  // ── Typewriter ──────────────────────────────────────────
  useEffect(() => {
    const PHRASES = [
      'Frictionless zoonotic surveillance, routed at the edge.',
      'Voice to risk assessment — on-device, under 20 seconds.',
      'MoE routing sends each field report to the right domain expert.',
      'No cloud. No latency. Full zoonotic intelligence on your hardware.',
    ]
    let pi = 0, ci = 0, tid: ReturnType<typeof setTimeout>
    function type() {
      if (ci <= PHRASES[pi].length) { setTypeText(PHRASES[pi].slice(0, ci++)); tid = setTimeout(type, ci === 1 ? 480 : 30) }
      else { tid = setTimeout(erase, 3200) }
    }
    function erase() {
      if (ci > 0) { setTypeText(PHRASES[pi].slice(0, --ci)); tid = setTimeout(erase, 12) }
      else { pi = (pi + 1) % PHRASES.length; tid = setTimeout(type, 550) }
    }
    tid = setTimeout(type, 900)
    return () => clearTimeout(tid)
  }, [])

  // ── 3D card tilt ────────────────────────────────────────
  function onCardMove(e: React.MouseEvent<HTMLDivElement>) {
    const c = e.currentTarget, r = c.getBoundingClientRect()
    const dx = (e.clientX - (r.left + r.width / 2)) / (r.width / 2)
    const dy = (e.clientY - (r.top  + r.height / 2)) / (r.height / 2)
    c.style.transform = `perspective(700px) rotateX(${-dy * 5}deg) rotateY(${dx * 5}deg) translateZ(4px)`
    c.style.setProperty('--mx', `${((e.clientX - r.left) / r.width * 100).toFixed(1)}%`)
    c.style.setProperty('--my', `${((e.clientY - r.top)  / r.height * 100).toFixed(1)}%`)
  }
  function onCardLeave(e: React.MouseEvent<HTMLDivElement>) { e.currentTarget.style.transform = '' }

  return (
    <div className={styles.landing}>
      {/* GPU particle system — blast/cursor drawn in same RAF */}
      <ParticleBackground mouseRef={mouseRef} onFrame={() => onFrameRef.current()} />
      {/* Blast rings + cursor glow — lightweight 2D overlay */}
      <canvas ref={blastRef} className={styles.physCanvas} />
      <div className={styles.aurora}>
        <div ref={orbARef} className={`${styles.orb} ${styles.orbA}`} />
        <div ref={orbBRef} className={`${styles.orb} ${styles.orbB}`} />
        <div ref={orbCRef} className={`${styles.orb} ${styles.orbC}`} />
      </div>

      <div className={styles.page}>
        {/* NAV */}
        <nav className={`${styles.nav} ${navScrolled ? styles.navScrolled : ''}`}>
          <div className={styles.navLogo}>
            <div className={styles.navLogoIcon}>Zx</div>
            <span className={styles.navLogoName}>ZoonoMoE</span>
          </div>
          <div className={styles.navRight}>
            <span className={styles.navPill}>● ONLINE</span>
            <a href="https://github.com/E27-25/MLOps_Project" target="_blank" rel="noreferrer" className={styles.navLink}>Docs</a>
            <button className={styles.btnPrimary} onClick={() => router.push('/app')} style={{ padding:'9px 22px', fontSize:'13px', borderRadius:'10px' }}>
              Launch →
            </button>
          </div>
        </nav>

        {/* HERO */}
        <section className={styles.hero}>
          <span className={styles.heroTag}>
            <span className={styles.heroTagDot} />
            AI ZOONOTIC SURVEILLANCE SYSTEM
          </span>
          <h1 className={styles.heroTitle}>
            Hear the symptoms.<br />
            <span className={styles.heroGrad}>Assess locally.<br />Contain globally.</span>
          </h1>
          <p className={styles.heroSub}>
            {typeText}<span className={styles.twCursor}>|</span>
          </p>
          <div className={styles.ctaRow}>
            <button className={styles.btnPrimary} onClick={() => router.push('/app')}>Launch ZoonoMoE →</button>
            <a href="https://github.com/E27-25/MLOps_Project" target="_blank" rel="noreferrer" className={styles.btnOutline}>View on GitHub</a>
          </div>
          <div className={styles.statsStrip}>
            {STATS.flatMap(([num, lbl], i, arr) => {
              const items = [
                <div key={num} className={styles.statItem}>
                  <span className={styles.statNum}>{num}</span>
                  <div className={styles.statLbl}>{lbl}</div>
                </div>
              ]
              if (i < arr.length - 1) items.push(<div key={`d${i}`} className={styles.statDivider} />)
              return items
            })}
          </div>
        </section>

        {/* PIPELINE */}
        <section className={styles.pipelineSection}>
          <div className={styles.pipelineWrap}>
            <div className={`${styles.pipelineHeader} landing-reveal`}>
              <div className={styles.sectionLabel}>How it works</div>
              <h2 className={styles.sectionTitle}>One voice.<br />Five stages.</h2>
              <p className={styles.sectionSub}>Each layer is purpose-built — from raw audio waveform to a spoken risk assessment.</p>
            </div>
            <div className={`${styles.pipelineFlow} landing-reveal`} style={{ transitionDelay: '0.15s' }}>
              {PIPELINE.flatMap((step, i, arr) => {
                const items = [
                  <div
                    key={step.name}
                    className={styles.plStep}
                    data-color={step.color}
                    onClick={() => setTooltip(tooltip === step.name ? null : step.name)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className={styles.plCircle}>
                      {step.icon}
                      {tooltip === step.name && (
                        <div className={styles.plTip}>{PIPELINE_INFO[step.name]}</div>
                      )}
                    </div>
                    <div className={styles.plName}>{step.name}</div>
                    <div className={styles.plDesc}>{step.desc}</div>
                  </div>
                ]
                if (i < arr.length - 1) items.push(
                  <div key={`c${i}`} className={styles.plConnector}>
                    <div className={styles.plConnLine}><div className={styles.plConnPulse} /></div>
                  </div>
                )
                return items
              })}
            </div>
          </div>
        </section>

        {/* FEATURES */}
        <section className={styles.section} style={{ paddingTop: 0 }}>
          <div className={`${styles.sectionLabel} landing-reveal`}>Capabilities</div>
          <h2 className={`${styles.sectionTitle} landing-reveal`}>Built for the field.</h2>
          <p className={`${styles.sectionSub} landing-reveal`} style={{ marginBottom: '52px', transitionDelay: '0.1s' }}>
            From a farmer calling in a sick flock to a district vet filing an alert — every voice report is handled.
          </p>
          <div className={styles.featureGrid}>
            {FEATURES.map((f, i) => (
              <div
                key={f.title}
                className={`${styles.featCard} landing-reveal`}
                style={{ transitionDelay: `${0.05 + i * 0.07}s` }}
                onMouseMove={onCardMove}
                onMouseLeave={onCardLeave}
              >
                <div className={styles.featIconWrap}>{f.icon}</div>
                <div className={styles.featTitle}>{f.title}</div>
                <div className={styles.featDesc}>{f.desc}</div>
                <span className={styles.featBadge}>{f.badge}</span>
              </div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section className={`${styles.ctaBanner} landing-reveal`}>
          <div className={styles.ctaGlow} />
          <div className={styles.ctaBannerInner}>
            <h2 className={styles.ctaTitle}>
              Ready to start your<br />
              <span style={{ background:'linear-gradient(135deg,#cba6f7,#89b4fa)', WebkitBackgroundClip:'text', backgroundClip:'text', WebkitTextFillColor:'transparent' }}>
                assessment?
              </span>
            </h2>
            <p className={styles.ctaSub}>
              Speak a field report. ZoonoMoE transcribes it, extracts clinical signals, routes it to the right domain model, and delivers a spoken risk assessment — all on-device.
            </p>
            <div className={styles.ctaRow}>
              <button className={styles.btnPrimary} onClick={() => router.push('/app')}>Launch ZoonoMoE →</button>
              <a href="https://github.com/E27-25/MLOps_Project" target="_blank" rel="noreferrer" className={styles.btnOutline}>View Docs</a>
            </div>
          </div>
        </section>

        <footer className={styles.footer}>
          <span>ZoonoMoE v2.0 · MLX · Qwen3 · Kokoro · FAISS</span>
          <span>GitHub · MLOps Project</span>
        </footer>
      </div>
    </div>
  )
}

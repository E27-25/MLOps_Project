'use client'
import { useRef, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

const N             = 70
const MAX_LINES     = (N * (N - 1)) / 2   // 2415
const CONNECT_DIST  = 140

// ── GLSL shaders ─────────────────────────────────────────────
const PARTICLE_VERT = `
attribute float aSize;
attribute float aOpacity;
attribute vec3  aColor;
varying   float vOpacity;
varying   vec3  vColor;
void main() {
  vColor   = aColor;
  vOpacity = aOpacity;
  vec4 mv  = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize;
  gl_Position  = projectionMatrix * mv;
}
`
const PARTICLE_FRAG = `
varying float vOpacity;
varying vec3  vColor;
void main() {
  vec2  uv = gl_PointCoord - 0.5;
  float d  = length(uv);
  if (d > 0.5) discard;
  float g  = pow(1.0 - d * 2.0, 1.8);
  gl_FragColor = vec4(vColor, vOpacity * g);
}
`
const LINE_VERT = `
attribute vec3  aLineColor;
attribute float aLineAlpha;
varying   vec3  vLineColor;
varying   float vLineAlpha;
void main() {
  vLineColor = aLineColor;
  vLineAlpha = aLineAlpha;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`
const LINE_FRAG = `
varying vec3  vLineColor;
varying float vLineAlpha;
void main() {
  gl_FragColor = vec4(vLineColor, vLineAlpha);
}
`

// ── Particle physics state ───────────────────────────────────
interface P {
  cx: number; cy: number
  ra: number; rb: number
  tilt: number; spd: number; theta: number
  x: number; y: number
  vx: number; vy: number; disrupted: number
  base: number; op: number
  angle: number; spin: number
  scale: number; tScale: number
  twinkle: number; twinkleSpeed: number
  r: number; col: [number, number, number]
}

const TYPES: { col: [number, number, number]; r: number }[] = [
  { col: [203 / 255, 166 / 255, 247 / 255], r: 3.5 },
  { col: [137 / 255, 180 / 255, 250 / 255], r: 2.5 },
  { col: [116 / 255, 199 / 255, 236 / 255], r: 5.0 },
  { col: [203 / 255, 166 / 255, 247 / 255], r: 4.0 },
  { col: [166 / 255, 227 / 255, 161 / 255], r: 2.0 },
]
function rnd(a: number, b: number) { return a + Math.random() * (b - a) }

function mkParticle(W: number, H: number): P {
  const t    = TYPES[Math.floor(Math.random() * TYPES.length)]
  const r    = t.r * rnd(0.7, 1.6)
  const cx   = rnd(W * 0.1, W * 0.9) - W / 2
  const cy   = H / 2 - rnd(H * 0.1, H * 0.9)
  const ra   = rnd(30, Math.min(W, H) * 0.22)
  const rb   = ra * rnd(0.35, 0.85)
  const tilt = rnd(0, Math.PI * 2)
  const spd  = rnd(0.003, 0.012) * (Math.random() < 0.5 ? 1 : -1)
  const th   = rnd(0, Math.PI * 2)
  const x    = cx + Math.cos(tilt) * ra * Math.cos(th) - Math.sin(tilt) * rb * Math.sin(th)
  const y    = cy + Math.sin(tilt) * ra * Math.cos(th) + Math.cos(tilt) * rb * Math.sin(th)
  return {
    cx, cy, ra, rb, tilt, spd, theta: th, x, y,
    vx: 0, vy: 0, disrupted: 0,
    base: rnd(0.2, 0.7), op: rnd(0.2, 0.7),
    angle: rnd(0, Math.PI * 2), spin: rnd(-0.012, 0.012),
    scale: 1, tScale: 1,
    twinkle: rnd(0, Math.PI * 2), twinkleSpeed: rnd(0.008, 0.025),
    r, col: t.col,
  }
}

// ── Public types ─────────────────────────────────────────────
export interface MouseState {
  x: number; y: number
  down: boolean; clicked: boolean
  cx: number; cy: number
}

// ── Inner R3F component ───────────────────────────────────────
function System({
  mouseRef,
  onFrame,
}: {
  mouseRef: React.RefObject<MouseState>
  onFrame?: () => void
}) {
  const { size } = useThree()
  const W = size.width, H = size.height

  const particles = useRef<P[]>([])
  const pointsRef = useRef<THREE.Points>(null)
  const linesRef  = useRef<THREE.LineSegments>(null)

  // Typed arrays — allocated once, never recreated
  const posArr  = useRef(new Float32Array(N * 3))
  const colArr  = useRef(new Float32Array(N * 3))   // set once, not per-frame
  const sizeArr = useRef(new Float32Array(N))
  const opArr   = useRef(new Float32Array(N))
  const lPosArr = useRef(new Float32Array(MAX_LINES * 6))
  const lColArr = useRef(new Float32Array(MAX_LINES * 6))
  const lAlpArr = useRef(new Float32Array(MAX_LINES * 2))
  const colDirty = useRef(true)   // upload colors only when particles are (re)created

  // Initialise particles + upload static color data
  useEffect(() => {
    const ps = Array.from({ length: N }, () => mkParticle(W, H))
    particles.current = ps
    for (let i = 0; i < N; i++) {
      const i3 = i * 3
      colArr.current[i3]     = ps[i].col[0]
      colArr.current[i3 + 1] = ps[i].col[1]
      colArr.current[i3 + 2] = ps[i].col[2]
    }
    colDirty.current = true
  }, [W, H])

  useFrame(() => {
    const ps    = particles.current
    const mouse = mouseRef.current
    if (!ps.length || !mouse) return

    const mx  = mouse.x  - W / 2
    const my  = H / 2 - mouse.y
    const mcx = mouse.cx - W / 2
    const mcy = H / 2 - mouse.cy

    // ── Physics ───────────────────────────────────────────
    for (let i = 0; i < N; i++) {
      const p = ps[i]
      p.theta += p.spd

      const tx = p.cx + Math.cos(p.tilt) * p.ra * Math.cos(p.theta) - Math.sin(p.tilt) * p.rb * Math.sin(p.theta)
      const ty = p.cy + Math.sin(p.tilt) * p.ra * Math.cos(p.theta) + Math.cos(p.tilt) * p.rb * Math.sin(p.theta)

      const dx   = p.x - mx, dy = p.y - my
      const dist = Math.sqrt(dx * dx + dy * dy) || 1
      const RR   = mouse.down ? 180 : 110, RS = mouse.down ? 9 : 3.5
      if (dist < RR) {
        const str = (1 - dist / RR) * RS
        p.vx += (dx / dist) * str; p.vy += (dy / dist) * str
        p.disrupted = Math.min(1, p.disrupted + 0.15)
      }
      if (mouse.clicked) {
        const cdx = p.x - mcx, cdy = p.y - mcy
        const cd  = Math.sqrt(cdx * cdx + cdy * cdy) || 1
        if (cd < 320) { p.vx += (cdx / cd) * (1 - cd / 320) * 28; p.vy += (cdy / cd) * (1 - cd / 320) * 28; p.disrupted = 1 }
      }
      p.disrupted *= 0.96
      const ob  = 1 - p.disrupted
      p.vx += (tx - p.x) * (0.04 + ob * 0.08); p.vy += (ty - p.y) * (0.04 + ob * 0.08)
      const damp = 0.78 + ob * 0.14; p.vx *= damp; p.vy *= damp
      p.x += p.vx; p.y += p.vy; p.angle += p.spin
      if (p.disrupted > 0.1) {
        if (p.x < -W / 2 + p.r) { p.x = -W / 2 + p.r; p.vx =  Math.abs(p.vx) * 0.35 }
        if (p.x >  W / 2 - p.r) { p.x =  W / 2 - p.r; p.vx = -Math.abs(p.vx) * 0.35 }
        if (p.y < -H / 2 + p.r) { p.y = -H / 2 + p.r; p.vy =  Math.abs(p.vy) * 0.35 }
        if (p.y >  H / 2 - p.r) { p.y =  H / 2 - p.r; p.vy = -Math.abs(p.vy) * 0.35 }
      }
      p.twinkle += p.twinkleSpeed
      p.op     += (p.base * (0.6 + (Math.sin(p.twinkle) * 0.5 + 0.5) * 0.55) - p.op) * 0.07
      p.tScale  = dist < 70 ? 1 + (1 - dist / 70) * 0.9 : 1
      p.scale  += (p.tScale - p.scale) * 0.14

      const i3 = i * 3
      posArr.current[i3]     = p.x
      posArr.current[i3 + 1] = p.y
      posArr.current[i3 + 2] = 0
      sizeArr.current[i] = p.r * p.scale * 9
      opArr.current[i]   = p.op
    }

    // ── Lines — use setDrawRange instead of zeroing ────────
    let li = 0
    for (let i = 0; i < N && li < MAX_LINES; i++) {
      for (let j = i + 1; j < N && li < MAX_LINES; j++) {
        const dx = ps[i].x - ps[j].x, dy = ps[i].y - ps[j].y
        const d  = Math.sqrt(dx * dx + dy * dy)
        if (d < CONNECT_DIST) {
          const a  = (1 - d / CONNECT_DIST) * 0.18
          const b6 = li * 6, b2 = li * 2
          lPosArr.current[b6]     = ps[i].x; lPosArr.current[b6 + 1] = ps[i].y; lPosArr.current[b6 + 2] = 0
          lPosArr.current[b6 + 3] = ps[j].x; lPosArr.current[b6 + 4] = ps[j].y; lPosArr.current[b6 + 5] = 0
          lColArr.current[b6]     = ps[i].col[0]; lColArr.current[b6 + 1] = ps[i].col[1]; lColArr.current[b6 + 2] = ps[i].col[2]
          lColArr.current[b6 + 3] = ps[j].col[0]; lColArr.current[b6 + 4] = ps[j].col[1]; lColArr.current[b6 + 5] = ps[j].col[2]
          lAlpArr.current[b2]     = a; lAlpArr.current[b2 + 1] = a
          li++
        }
      }
    }

    // ── GPU upload — minimal needsUpdate ──────────────────
    if (pointsRef.current) {
      const g = pointsRef.current.geometry
      ;(g.attributes.position as THREE.BufferAttribute).needsUpdate = true
      ;(g.attributes.aSize    as THREE.BufferAttribute).needsUpdate = true
      ;(g.attributes.aOpacity as THREE.BufferAttribute).needsUpdate = true
      if (colDirty.current) {
        ;(g.attributes.aColor as THREE.BufferAttribute).needsUpdate = true
        colDirty.current = false
      }
    }
    if (linesRef.current) {
      const g = linesRef.current.geometry
      ;(g.attributes.position   as THREE.BufferAttribute).needsUpdate = true
      ;(g.attributes.aLineColor as THREE.BufferAttribute).needsUpdate = true
      ;(g.attributes.aLineAlpha as THREE.BufferAttribute).needsUpdate = true
      g.setDrawRange(0, li * 2)   // ← only draw active lines, no zeroing needed
    }

    mouse.clicked = false

    // ── 2D overlay (blast/cursor) drawn in SAME frame ─────
    onFrame?.()
  })

  return (
    <>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[posArr.current, 3]} />
          <bufferAttribute attach="attributes-aColor"   args={[colArr.current, 3]} />
          <bufferAttribute attach="attributes-aSize"    args={[sizeArr.current, 1]} />
          <bufferAttribute attach="attributes-aOpacity" args={[opArr.current, 1]} />
        </bufferGeometry>
        <shaderMaterial
          vertexShader={PARTICLE_VERT}
          fragmentShader={PARTICLE_FRAG}
          transparent depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>

      <lineSegments ref={linesRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position"   args={[lPosArr.current, 3]} />
          <bufferAttribute attach="attributes-aLineColor" args={[lColArr.current, 3]} />
          <bufferAttribute attach="attributes-aLineAlpha" args={[lAlpArr.current, 1]} />
        </bufferGeometry>
        <shaderMaterial
          vertexShader={LINE_VERT}
          fragmentShader={LINE_FRAG}
          transparent depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </lineSegments>
    </>
  )
}

// ── Public component ─────────────────────────────────────────
export default function ParticleBackground({
  mouseRef,
  onFrame,
}: {
  mouseRef: React.RefObject<MouseState>
  onFrame?: () => void
}) {
  return (
    <Canvas
      orthographic
      camera={{ zoom: 1, position: [0, 0, 100] }}
      style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}
      gl={{ alpha: true, antialias: false }}
      dpr={1}            // fixed — no runtime DPR variance
      frameloop="always" // explicit 60fps loop
    >
      <System mouseRef={mouseRef} onFrame={onFrame} />
    </Canvas>
  )
}

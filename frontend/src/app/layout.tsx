import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ZoonoMoE — Frictionless Zoonotic Surveillance',
  description: 'Voice-driven zoonotic disease surveillance. On-device, real-time, fully private.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

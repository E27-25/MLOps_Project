import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  env: {
    // Expose backend URL to browser bundle at build time
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:7860',
  },
}

export default nextConfig

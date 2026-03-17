/**
 * Shared backend URL helper.
 * Set NEXT_PUBLIC_BACKEND_URL in .env.local or Vercel dashboard.
 * Falls back to localhost:7860 for local dev.
 */
export const BACKEND = (
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:7860"
).replace(/\/$/, "");

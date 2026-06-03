# ZoonoMoE — Deployment Guide

## GPU Backend (your machine) + Vercel Frontend (public URL)

---

## Architecture

```
Browser → Vercel (Next.js pages)         free Hobby plan
Browser → Cloudflare Tunnel → :7860      FastAPI + vLLM on YOUR GPU
                            → :8000      NVIDIA Triton
Vercel  → Neon PostgreSQL               /api/reports + /dashboard
```

---

## Step 1 — Start backend on your GPU machine

```bash
# Clone / pull latest
cd mlop-project

# Set your Vercel domain (you'll get this in Step 3)
# For now, start with localhost only:
docker compose up --build
```

Backend will be up at `http://localhost:7860` and Triton at `http://localhost:8000`.

---

## Step 2 — Expose backend with Cloudflare Tunnel (free, no account needed)

```bash
# Install cloudflared (Windows)
winget install --id Cloudflare.cloudflared

# OR download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

# Start a temporary tunnel (gives you a public HTTPS URL instantly)
cloudflared tunnel --url http://localhost:7860
```

You'll see output like:

```
Your quick Tunnel has been created! Visit it at:
https://abc123-xyz.trycloudflare.com
```

> **Keep this terminal running** — the tunnel is live as long as this process runs.
> For a permanent URL, use a named tunnel with a Cloudflare account (recommended for production).

---

## Step 3 — Deploy frontend to Vercel

```bash
# Install Vercel CLI (one time)
npm install -g vercel

# Go to frontend folder
cd frontend
npm install

# Deploy
vercel --prod
```

Vercel will ask you to log in and link a project. After deploy, you'll get a URL like:
`https://zoonmoe.vercel.app`

---

## Step 4 — Set environment variables on Vercel

In your [Vercel dashboard](https://vercel.com/dashboard) → your project → **Settings → Environment Variables**, add:

| Name                      | Value                                                                |
| ------------------------- | -------------------------------------------------------------------- |
| `NEXT_PUBLIC_BACKEND_URL` | `https://abc123-xyz.trycloudflare.com` (your tunnel URL from Step 2) |
| `DATABASE_URL`            | Your Neon PostgreSQL connection string                               |

Then **redeploy**:

```bash
vercel --prod
```

---

## Step 5 — Update CORS on backend to allow Vercel domain

Edit your `.env` or re-run docker compose with the Vercel domain added:

```bash
ALLOWED_ORIGINS="http://localhost:3000,https://zoonmoe.vercel.app" docker compose up -d backend
```

Or add it permanently to `docker-compose.yml`:

```yaml
- ALLOWED_ORIGINS=http://localhost:3000,https://zoonmoe.vercel.app
```

---

## Running for a week (stability tips)

| Concern                       | Solution                                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| Tunnel URL changes on restart | Use a **named Cloudflare Tunnel** with a stable domain (free with Cloudflare account)        |
| Backend crashes               | `restart: unless-stopped` is already set in docker-compose                                   |
| GPU machine reboots           | Enable Docker to start on boot: `docker compose up -d` in Windows Task Scheduler / autostart |
| Tunnel dies                   | Run cloudflared as a Windows service: `cloudflared service install`                          |

### Run cloudflared as a Windows service (permanent tunnel)

```bash
# After running `cloudflared tunnel --url http://localhost:7860` at least once:
cloudflared service install
net start cloudflared
```

---

## Local development (no Vercel)

```bash
# Terminal 1 — GPU backend
docker compose up

# Terminal 2 — Frontend dev server
cd frontend
cp .env.example .env.local
# Edit .env.local and set NEXT_PUBLIC_BACKEND_URL=http://localhost:7860
npm run dev
```

Open http://localhost:3000

---

## Summary of what each env var does

| Variable                  | Where                           | Purpose                                              |
| ------------------------- | ------------------------------- | ---------------------------------------------------- |
| `ALLOWED_ORIGINS`         | backend docker-compose          | Comma-separated CORS origins allowed to call FastAPI |
| `NEXT_PUBLIC_BACKEND_URL` | Vercel dashboard                | GPU backend URL that the browser fetches directly    |
| `DATABASE_URL`            | backend docker-compose + Vercel | Neon PostgreSQL for report storage + dashboard       |

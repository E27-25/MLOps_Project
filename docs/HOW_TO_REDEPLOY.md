# 🚀 How to Re-Deploy ZoonoMoE (Quickstart)

If your computer restarts, Docker crashes, or you need to launch the project again from scratch, just follow these exact steps.

*(Note: We fixed the CORS issue in the backend, so you **no longer** need to manually whitelist Vercel URLs! Any `*.vercel.app` domain is automatically allowed to connect to your GPU.)*

---

## Step 1: Start your GPU Backend

1. **Open Docker Desktop** and make sure it is running (the whale icon in your system tray should be stable/green).
2. Open PowerShell and go to your project folder:
   ```powershell
   cd C:\Users\usEr\Desktop\mlop-project
   ```
3. Start the AI stack:
   ```powershell
   docker compose up -d
   ```
4. **Wait for models to load.** This takes a few minutes. You can check if it's completely ready by running:
   ```powershell
   curl -s http://localhost:7860/health
   ```
   *(Wait until you see `{"status":"ok"...}` before moving to the next step).*

---

## Step 2: Open the Cloudflare Tunnel

Your Vercel frontend needs a way to securely talk to your local GPU. We use a free Cloudflare Tunnel for this.

1. Open a **new, separate PowerShell window**.
2. Run this exact command to start the tunnel:
   ```powershell
   & "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:7860
   ```
3. Look closely at the output text. You need to find a line that looks exactly like this:
   ```
   Your quick Tunnel has been created! Visit it at:
   https://some-random-words.trycloudflare.com
   ```
4. **Copy that `trycloudflare.com` URL.**
5. **DO NOT close this window.** Keep the tunnel running in the background.

---

## Step 3: Connect Vercel to your new Tunnel URL

Every time you run Step 2, you get a randomly generated Cloudflare URL. You have to give Vercel this new URL so the frontend knows where to send requests.

1. Go back to your main PowerShell window.
2. Navigate into your frontend folder:
   ```powershell
   cd frontend
   ```
3. **Remove the old, broken URL** from Vercel:
   ```powershell
   vercel env rm NEXT_PUBLIC_BACKEND_URL production --yes
   ```
4. **Add the NEW URL** you copied in Step 2: (Replace the URL below with your actual `trycloudflare.com` URL)
   ```powershell
   echo "https://your-new-url.trycloudflare.com" | vercel env add NEXT_PUBLIC_BACKEND_URL production
   ```
5. **Redeploy the frontend** to lock in the new URL:
   ```powershell
   vercel --prod --yes
   ```

---

## Step 4: You're Done! 🎉

Wait for the deployment to finish (1-2 minutes). The Vercel CLI will give you a **Production** URL like:
`✅ Production: https://frontend-xxxxx-projects.vercel.app`

Click that link — your app is now fully live and connected entirely to your GPU backend!

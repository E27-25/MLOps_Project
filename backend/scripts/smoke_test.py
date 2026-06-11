#!/usr/bin/env python3
"""
ZoonoMoE — end-to-end pipeline smoke test.

Boots the real FastAPI app (loading every stage) and drives the full pipeline
through its HTTP routes with an in-process TestClient — no server/port needed:

    /upload  → ASR (Whisper + Pathumma-th)          [stage 1]
    /analyze → NER + Router(+cascade) + RAG          [stages 2-4]
    /stream  → Expert LLM + TTS (Kokoro or JaiTTS)   [stages 5-6]

Each stage is reported PASS/FAIL with timing and a snippet of output, so a
partial environment still yields a useful per-stage verdict. Exit code is the
number of failed stages (0 = all green).

Usage:
    python scripts/smoke_test.py --wav voices/th_ref.wav --tts-lang th
"""
import os, sys, json, time, argparse, traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(BACKEND / "voices" / "th_ref.wav"))
    ap.add_argument("--tts-lang", default=os.getenv("TTS_LANG", "en"), choices=["en", "th"])
    args = ap.parse_args()

    # ── force the local, GPU, non-Triton config before importing app ──────────
    os.environ.setdefault("USE_MLX", "false")
    os.environ.setdefault("USE_TRITON", "false")
    os.environ["TTS_LANG"] = args.tts_lang
    os.environ.setdefault("JAITTS_REF_VOICE", str(BACKEND / "voices" / "th_ref.wav"))
    os.environ.setdefault("LLM_MODEL", "Qwen/Qwen3-14B")   # already in HF cache
    os.environ.setdefault("WHISPER_SIZE", "base")
    os.chdir(BACKEND)
    sys.path.insert(0, str(BACKEND))

    results: list[tuple[str, bool, str]] = []

    def stage(name, fn):
        t0 = time.time()
        try:
            info = fn()
            dt = time.time() - t0
            print(f"  ✓ {name:28} {dt:6.2f}s  {info}")
            results.append((name, True, info))
            return True
        except Exception as e:
            dt = time.time() - t0
            print(f"  ✗ {name:28} {dt:6.2f}s  {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))
            return False

    print("=" * 72)
    print(f"ZoonoMoE smoke test | wav={args.wav} | TTS_LANG={args.tts_lang} | LLM={os.environ['LLM_MODEL']}")
    print("=" * 72)

    # ── 0. import app (this loads ASR, LLM, embedder, router, RAG, TTS) ───────
    ctx: dict = {}

    def _boot():
        import app as appmod
        from fastapi.testclient import TestClient
        ctx["app"] = appmod
        ctx["client"] = TestClient(appmod.app)
        r = ctx["client"].get("/health")
        return f"/health -> {r.status_code} {str(r.json())[:80]}"

    if not stage("0. boot app + load models", _boot):
        print("\n‼ App failed to boot — cannot continue.")
        return _summary(results)

    client = ctx["client"]

    # ── 1. ASR (/upload) ──────────────────────────────────────────────────────
    def _asr():
        with open(args.wav, "rb") as f:
            r = client.post("/upload", files={"audio": (Path(args.wav).name, f, "audio/wav")})
        r.raise_for_status()
        ctx["transcript"] = r.json()["transcript"]
        return f"transcript[{len(ctx['transcript'])} chars]: {ctx['transcript'][:60]!r}"
    asr_ok = stage("1. ASR /upload", _asr)
    if not asr_ok:
        ctx["transcript"] = "มีไก่ตายสามตัวเมื่อคืน หงอนเขียวคล้ำ ตัวหนึ่งชัก"  # fallback for later stages

    # ── 2-4. NER + Router + RAG (/analyze) ────────────────────────────────────
    def _analyze():
        r = client.post("/analyze", json={"transcript": ctx["transcript"]})
        r.raise_for_status()
        d = r.json()
        ctx["analyze"] = d
        return (f"domain={d['domain']} conf={d['confidence']} "
                f"stage={d.get('route_stage')} rag_chunks={len(d.get('rag_chunks', []))} "
                f"epi_keys={list(d.get('epi_fields', {}).keys())}")
    analyze_ok = stage("2-4. NER+Router+RAG /analyze", _analyze)

    # ── 5-6. Expert LLM + TTS (/stream) ───────────────────────────────────────
    def _stream():
        a = ctx.get("analyze", {})
        payload = {
            "domain":     a.get("domain", "general"),
            "epi_fields": a.get("epi_fields", {}),
            "rag_chunks": a.get("rag_chunks", []),
            "transcript": ctx["transcript"],
        }
        text_chunks, audio_chunks, first_audio = 0, 0, None
        with client.stream("POST", "/stream", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                evt = json.loads(line[6:])
                if evt.get("type") == "text":
                    text_chunks += 1
                elif evt.get("type") == "audio":
                    audio_chunks += 1
                    if first_audio is None:
                        first_audio = evt.get("data", "")
        if first_audio:
            import base64
            out = BACKEND / "scripts" / "eval_out" / f"smoke_tts_{args.tts_lang}.wav"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(base64.b64decode(first_audio))
            ctx["audio_out"] = str(out)
        return f"text_chunks={text_chunks} audio_chunks={audio_chunks} saved={ctx.get('audio_out', 'none')}"
    stage("5-6. Expert LLM + TTS /stream", _stream)

    return _summary(results)


def _summary(results) -> int:
    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for name, ok, _ in results:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {passed}/{len(results)} stages passed, {failed} failed")
    print("=" * 72)
    return failed


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Generate input/output demo samples for ZoonoMoE and verify chunking.

Drives the real pipeline (in-process TestClient) on a spoken field report:
    /upload  -> ASR transcript
    /analyze -> NER + Router + RAG
    /stream  -> Expert LLM (text chunks) + TTS (audio chunks)

Saves the FULL spoken response (all audio chunks concatenated) plus a text
breakdown showing every (text chunk -> audio seconds) pair — so you can confirm
the streaming sentence-chunker actually split + synthesised piece by piece.

    python scripts/make_samples.py --wav samples/input_report_en.wav --tts-lang th
"""
import os, sys, io, json, base64, argparse
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(BACKEND / "samples" / "input_report_en.wav"))
    ap.add_argument("--tts-lang", default=os.getenv("TTS_LANG", "en"), choices=["en", "th"])
    ap.add_argument("--out-dir", default=str(BACKEND / "samples"))
    args = ap.parse_args()

    os.environ.setdefault("USE_MLX", "false")
    os.environ.setdefault("USE_TRITON", "false")
    os.environ["TTS_LANG"] = args.tts_lang
    os.chdir(BACKEND); sys.path.insert(0, str(BACKEND))
    import numpy as np, soundfile as sf
    import app as appmod
    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)
    lang = args.tts_lang
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    # 1) ASR
    with open(args.wav, "rb") as f:
        r = client.post("/upload", files={"audio": (Path(args.wav).name, f, "audio/wav")})
    r.raise_for_status()
    transcript = r.json()["transcript"]
    print(f"[ASR] transcript: {transcript!r}")

    # 2) NER + Router + RAG
    a = client.post("/analyze", json={"transcript": transcript}).json()
    print(f"[ANALYZE] domain={a['domain']} conf={a['confidence']} "
          f"stage={a.get('route_stage')} rag={len(a.get('rag_chunks', []))}")

    # 3) Expert LLM + TTS (stream), collect text + audio per chunk
    payload = {"domain": a["domain"], "epi_fields": a["epi_fields"],
               "rag_chunks": a["rag_chunks"], "transcript": transcript}
    rows, all_audio, sr = [], [], None
    with client.stream("POST", "/stream", json=payload) as resp:
        resp.raise_for_status()
        pending_text = None
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            evt = json.loads(line[6:])
            if evt.get("type") == "text":
                pending_text = evt["chunk"]
            elif evt.get("type") == "audio":
                wav = base64.b64decode(evt["data"])
                data, sr = sf.read(io.BytesIO(wav))
                all_audio.append(data)
                rows.append((evt.get("sentence", pending_text) or "", len(data) / sr))

    if not all_audio:
        print("‼ NO AUDIO CHUNKS — TTS produced nothing"); return 1

    full = np.concatenate(all_audio)
    wav_path = out / f"output_{lang}.wav"
    sf.write(wav_path, full, sr)

    # text breakdown (proves chunking)
    txt_path = out / f"output_{lang}.txt"
    with open(txt_path, "w") as fh:
        fh.write(f"# ZoonoMoE response ({lang}) — {len(rows)} chunks, "
                 f"{full.shape[0]/sr:.1f}s total @ {sr}Hz\n")
        fh.write(f"# input transcript: {transcript}\n")
        fh.write(f"# routed domain: {a['domain']} (conf {a['confidence']})\n\n")
        for i, (t, dur) in enumerate(rows, 1):
            fh.write(f"[chunk {i}] {dur:4.1f}s | {t.strip()}\n")

    print(f"\n[TTS] {len(rows)} chunks synthesised, {full.shape[0]/sr:.1f}s total @ {sr}Hz")
    for i, (t, dur) in enumerate(rows, 1):
        print(f"  chunk {i}: {dur:4.1f}s | {t.strip()[:70]}")
    print(f"\n✓ saved {wav_path}")
    print(f"✓ saved {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare Thai TTS quality with an English ref vs a Thai ref (+ silence trim).

Synthesises the same Thai paragraph with two reference voices and reports
duration, chars/sec, and silence %, so you can A/B listen and confirm whether
the 'weird' is the cross-lingual reference (English voice -> Thai model).
"""
import os, io, sys, base64
from pathlib import Path
BACKEND = Path(__file__).resolve().parents[1]
os.chdir(BACKEND); sys.path.insert(0, str(BACKEND))
import numpy as np, soundfile as sf
from utils.thai_tts import ThaiTTS

PARA = [
    "จากอาการที่อธิบาย น่าจะเป็นโรคไข้หวัดนกชนิดรุนแรง",
    "ความเสี่ยงอยู่ในระดับสูง เพราะแพร่กระจายเร็วและมีอัตราการตายสูง",
    "ควรแยกสัตว์ที่ติดเชื้อทันที สวมอุปกรณ์ป้องกัน และรายงานหน่วยงานที่เกี่ยวข้อง",
]
nchars = sum(len(s.replace(" ", "")) for s in PARA)
out = BACKEND / "samples" / "compare"; out.mkdir(parents=True, exist_ok=True)

def silence_pct(data, sr, thr=0.012):
    fl = int(0.05 * sr); n = len(data)//fl
    rms = np.sqrt((data[:n*fl].reshape(n, fl)**2).mean(axis=1))
    return 100*np.mean(rms < thr)

refs = [
    ("kokoro_en", BACKEND/"voices"/"kokoro_ref_en.wav", BACKEND/"voices"/"kokoro_ref_en.txt"),
    ("thairef",   BACKEND/"voices"/"th_ref_thai.wav",   BACKEND/"voices"/"th_ref_thai.txt"),
]
for name, rv, rt in refs:
    if not rv.exists():
        print(f"[{name}] ref missing, skip"); continue
    tts = ThaiTTS(device="cuda", ref_voice=str(rv), ref_text=rt.read_text().strip())
    tts.speed = float(os.getenv("JAITTS_SPEED", "1.1"))
    chunks = []
    for s in PARA:
        b = tts.synth_b64(s)
        if b:
            d, sr = sf.read(io.BytesIO(base64.b64decode(b))); chunks.append(d)
    if not chunks:
        print(f"[{name}] FAILED"); continue
    full = np.concatenate(chunks)
    sf.write(out / f"th_{name}.wav", full, sr)
    dur = len(full)/sr
    print(f"[{name}] {dur:5.1f}s | {nchars/dur:4.1f} chars/sec | silence {silence_pct(full,sr):4.1f}% | "
          f"saved samples/compare/th_{name}.wav")
print("\nDONE")

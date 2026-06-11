#!/usr/bin/env python3
"""Synthesise one Thai sentence at several speeds to pick a natural pace.

Saves samples/speedtest/th_speed_<x>.wav and prints duration + chars/sec so you
can compare (natural Thai TTS ≈ 4.5–6 chars/sec; the default ref clip drags
~2.6 because F5 clones the reference's slow pace).
"""
import os, io, sys, base64
from pathlib import Path
BACKEND = Path(__file__).resolve().parents[1]
os.chdir(BACKEND); sys.path.insert(0, str(BACKEND))
import soundfile as sf

ref_voice = os.getenv("JAITTS_REF_VOICE", str(BACKEND / "voices" / "th_ref.wav"))
ref_text  = os.getenv("JAITTS_REF_TEXT") or (BACKEND / "voices" / "th_ref.txt").read_text().strip()
from utils.thai_tts import ThaiTTS
tts = ThaiTTS(device="cuda", ref_voice=ref_voice, ref_text=ref_text)

SENT = ("จากอาการที่อธิบาย น่าจะเป็นโรคไข้หวัดนกชนิดรุนแรง "
        "ความเสี่ยงอยู่ในระดับสูง ต้องแยกสัตว์ที่ติดเชื้อทันที และรายงานหน่วยงานที่เกี่ยวข้อง")
nchars = len(SENT.replace(" ", ""))
out = BACKEND / "samples" / "speedtest"; out.mkdir(parents=True, exist_ok=True)

print(f"text: {nchars} Thai chars\n")
for s in [1.0, 1.3, 1.6, 2.0]:
    tts.speed = s
    b64 = tts.synth_b64(SENT)
    if not b64:
        print(f"speed {s}: FAILED"); continue
    data, sr = sf.read(io.BytesIO(base64.b64decode(b64)))
    dur = len(data) / sr
    (out / f"th_speed_{s}.wav").write_bytes(base64.b64decode(b64))
    print(f"speed {s:>4}: {dur:6.1f}s  ->  {nchars/dur:4.1f} chars/sec")
print(f"\nsaved clips in {out}")

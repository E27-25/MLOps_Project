"""
utils/thai_tts.py — Thai text-to-speech for the ZoonoMoE TTS stage.

Wraps **JaiTTS (Thonburian-TTS / F5-TTS)** so the pipeline can speak Thai
instead of English Kokoro. Selected at runtime by ``TTS_LANG=th`` (see app.py).

JaiTTS is an F5-TTS (flow-matching) voice-cloning model: it needs a short
**reference voice** clip (``ref_voice``) whose timbre it imitates, and ideally
the transcript of that clip (``ref_text``) for best prosody.

Model card:  https://huggingface.co/JTS-AI/JaiTTS-F5TTS
Library:     https://github.com/biodatlab/thonburian-tts  (``flowtts``)

Install (heavy, GPU recommended)::

    pip install -r requirements-thai.txt      # f5-tts, cached-path, vocos, librosa
    # plus a system ffmpeg:  apt install -y ffmpeg

Because F5-TTS is GPU-heavy and not re-entrant, we keep a single pipeline
instance guarded by a lock (unlike the 3-way Kokoro pool).
"""
from __future__ import annotations

import os
import sys
import base64
import logging
import tempfile
import threading

log = logging.getLogger("zoonomoe.thai_tts")

# ── Defaults (override via env) ──────────────────────────────────────────────
_DEF_CHECKPOINT = os.getenv("JAITTS_CHECKPOINT", "hf://JTS-AI/JaiTTS-F5TTS/model.pt")
_DEF_VOCAB      = os.getenv("JAITTS_VOCAB",      "hf://JTS-AI/JaiTTS-F5TTS/vocab.txt")
_DEF_VOCODER    = os.getenv("JAITTS_VOCODER",    "vocos")


def _tighten_wav_b64(path: str, thresh: float = 0.012,
                     pad: float = 0.10, max_gap: float = 0.30) -> str:
    """Trim leading/trailing silence and cap long internal gaps → base64 WAV.

    F5 tends to pad each utterance with long silence; concatenated chunks then
    sound broken (dead air). This keeps speech continuous. Tunable via
    ``JAITTS_MAX_GAP`` (seconds). On any error, returns the file untouched.
    """
    import io
    import numpy as np
    import soundfile as sf
    try:
        max_gap = float(os.getenv("JAITTS_MAX_GAP", max_gap))
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        fl = max(1, int(0.02 * sr))                       # 20 ms frames
        n = len(data) // fl
        if n < 2:
            raise ValueError("clip too short")
        rms = np.sqrt((data[:n * fl].reshape(n, fl) ** 2).mean(axis=1))
        voiced = rms > thresh
        if not voiced.any():
            raise ValueError("all silence")
        keep = voiced.copy()
        gap_f = int(round(max_gap / 0.02))
        i = 0
        while i < n:
            if voiced[i]:
                i += 1
                continue
            j = i
            while j < n and not voiced[j]:
                j += 1
            if i == 0 or j == n:                          # leading / trailing
                keep[i:j] = False
            else:                                         # internal gap → cap it
                keep[i:i + min(j - i, gap_f)] = True
                keep[i + min(j - i, gap_f):j] = False
            i = j
        pad_f = int(round(pad / 0.02))                    # re-add a little pad
        first, last = np.argmax(voiced), n - 1 - np.argmax(voiced[::-1])
        for k in range(max(0, first - pad_f), first):
            keep[k] = True
        for k in range(last + 1, min(n, last + 1 + pad_f)):
            keep[k] = True
        idx = np.repeat(keep, fl)
        out = data[:n * fl][idx]
        buf = io.BytesIO()
        sf.write(buf, out, sr, format="WAV")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:                                     # fall back to raw file
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()


class ThaiTTS:
    """JaiTTS / F5-TTS synthesiser exposing a Kokoro-compatible ``synth_b64``."""

    def __init__(
        self,
        device: str = "cuda",
        ref_voice: str | None = None,
        ref_text: str | None = None,
        checkpoint: str = _DEF_CHECKPOINT,
        vocab_file: str = _DEF_VOCAB,
        vocoder: str = _DEF_VOCODER,
        cfg_strength: float = 2.5,
        nfe_step: int = 32,
        speed: float = 0.8,         # paired with the shipped (brisk) Thai ref
                                    # `voices/th_ref.wav`. F5 clones the ref's
                                    # pace; for a slower/normal ref use ~1.0.
                                    # Tune via JAITTS_SPEED.
        silence_threshold: int = -45,
    ):
        # `flowtts` lives in the thonburian-tts repo as a namespace package, not
        # a pip module — add its checkout to sys.path (set THONBURIAN_TTS_DIR).
        _src = os.getenv("THONBURIAN_TTS_DIR")
        if _src and _src not in sys.path:
            sys.path.insert(0, _src)
        # Imported lazily so the English path never needs flowtts installed.
        from flowtts.inference import FlowTTSPipeline, ModelConfig, AudioConfig

        if not ref_voice or not os.path.exists(ref_voice):
            raise FileNotFoundError(
                f"JaiTTS needs a reference voice clip. Set JAITTS_REF_VOICE to a "
                f"short Thai .wav (got: {ref_voice!r})."
            )

        self.ref_voice = ref_voice
        self.ref_text  = (ref_text or "").strip() or None
        self._lock     = threading.Lock()

        model_config = ModelConfig(
            language="th",
            model_type="F5",
            checkpoint=checkpoint,
            vocab_file=vocab_file,
            vocoder=vocoder,
            device=device,
        )
        # env overrides for live tuning without code changes
        self.speed = float(os.getenv("JAITTS_SPEED", speed))
        audio_config = AudioConfig(
            silence_threshold=int(os.getenv("JAITTS_SILENCE_DB", silence_threshold)),
            cfg_strength=float(os.getenv("JAITTS_CFG", cfg_strength)),
            nfe_step=int(os.getenv("JAITTS_NFE_STEP", nfe_step)),
            speed=self.speed,
        )
        self.pipeline = FlowTTSPipeline(
            model_config=model_config,
            audio_config=audio_config,
        )
        log.info(
            "JaiTTS (F5) ready on %s | ref_voice=%s | ref_text=%s",
            device, os.path.basename(ref_voice), "set" if self.ref_text else "auto",
        )

    def synth_b64(self, text: str) -> str | None:
        """Synthesise Thai ``text`` → base64-encoded WAV (or None on failure)."""
        text = (text or "").strip()
        if not text:
            return None
        out_path = tempfile.mktemp(suffix=".wav")
        try:
            # NOTE: flowtts passes the *call-time* speed to inference (not the
            # AudioConfig), so it must be set here to take effect.
            kwargs = dict(ref_voice=self.ref_voice, text=text,
                          output_file=out_path, speed=self.speed)
            if self.ref_text:
                kwargs["ref_text"] = self.ref_text
            with self._lock:                    # F5 pipeline is not re-entrant
                self.pipeline(**kwargs)
            return _tighten_wav_b64(out_path)
        except Exception as e:                  # noqa: BLE001 — never break the SSE stream
            log.error("JaiTTS synth error: %s", e)
            return None
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

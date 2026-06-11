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
        speed: float = 1.1,         # F5 clones the ref's pace: with a normally-
                                    # spoken reference, ~1.0–1.1 ≈ natural Thai
                                    # (4.6–5 chars/sec). A slow ref needs ~1.6.
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
            with open(out_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except Exception as e:                  # noqa: BLE001 — never break the SSE stream
            log.error("JaiTTS synth error: %s", e)
            return None
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

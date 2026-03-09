"""
scripts/discord_logger.py
==========================
ZoonoMoE pipeline logger via discordflow (Forum channel mode).

Each pipeline run creates its own Discord thread — one thread per
inference call. Stages ASR → NER → Router → RAG → LLM → TTS are
logged as incremental metric/param embeds inside that thread.

Usage
-----
1. Set env var:
   export DISCORD_WEBHOOK=https://discord.com/api/webhooks/...

2. Wrap your pipeline call (see `log_pipeline_run` below).

3. For router retraining, call `log_router_retrain`.

Install
-------
pip install "discordflow[system]"

Integration points
------------------
In app.py, import and call:

    from scripts.discord_logger import log_pipeline_run, log_router_retrain

Then call log_pipeline_run(...) at the end of the /analyze + /stream flow.
"""

import os
import time
import datetime
from pathlib import Path

# ── Env ────────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
DRY_RUN         = not DISCORD_WEBHOOK               # prints to stdout if no webhook set


# ── Helper: lazy init so app.py doesn't crash if token missing ─────────────────
def _make_dflow():
    from discordflow import DiscordFlow
    return DiscordFlow(
        webhook_url     = DISCORD_WEBHOOK,
        experiment_name = "ZoonoMoE",
        state_file      = ".discordflow_state.json",  # persist thread IDs across restarts
        async_logging   = True,    # never block the Flask request
        dry_run         = DRY_RUN, # safe during local dev with no webhook
        username        = "ZoonoMoE Logger",
        avatar_url      = "https://i.redd.it/6bjrd13klgmf1.jpeg",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMARY LOGGER: log_pipeline_run()
#  Call once per /analyze + /stream cycle.
# ══════════════════════════════════════════════════════════════════════════════

def log_pipeline_run(
    *,
    # ── Input ────────────────────────────────────────────────────────
    transcript:      str,
    input_method:    str,           # "voice" | "text"

    # ── [1] ASR ──────────────────────────────────────────────────────
    asr_time_s:      float,
    asr_backend:     str,           # "mlx" | "openai"
    whisper_size:    str,           # "tiny" | "base" | "small" | "medium"
    hallucination:   bool = False,  # True if guard fired

    # ── [2] NER ──────────────────────────────────────────────────────
    ner_time_s:      float,
    epi_fields:      dict,          # full NER output dict

    # ── [3] Router ───────────────────────────────────────────────────
    route_time_s:    float,
    domain:          str,           # winning domain
    confidence:      float,         # 0-1
    all_scores:      dict,          # {domain: score, ...}
    off_topic:       bool = False,  # True if regex guard overrode MLP

    # ── [4] RAG ──────────────────────────────────────────────────────
    rag_time_s:      float,
    rag_chunks:      list,          # list of retrieved text strings

    # ── [5] LLM ──────────────────────────────────────────────────────
    llm_time_s:      float,
    risk_level:      str,           # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | "NONE"
    report_flag:     bool,
    full_response:   str,           # complete LLM output

    # ── [6] TTS ──────────────────────────────────────────────────────
    tts_chunks:      int,           # number of audio chunks sent
    tts_time_s:      float,
):
    """
    Log one full ZoonoMoE inference to a Discord Forum thread.
    Each call creates a new thread named:
        ZoonoMoE · {domain} · {timestamp}
    """
    dflow = _make_dflow()

    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time  = asr_time_s + ner_time_s + route_time_s + rag_time_s + llm_time_s + tts_time_s
    run_name    = f"{domain} · {timestamp}"
    description = f"{'🛑 HALLUCINATION' if hallucination else input_method.upper()} | Risk: {risk_level} | {total_time:.1f}s total"

    with dflow.start_forum_run(run_name, description=description) as run:

        # ── Metadata tags ─────────────────────────────────────────────
        run.set_tag("domain",       domain)
        run.set_tag("risk_level",   risk_level)
        run.set_tag("input_method", input_method)
        run.set_tag("off_topic",    str(off_topic))
        run.set_tag("timestamp",    timestamp)

        # ── [1] ASR stage ─────────────────────────────────────────────
        run.log_params({
            "asr_backend":   asr_backend,
            "whisper_size":  whisper_size,
            "hallucination": hallucination,
            "transcript_len": len(transcript.split()),
        })
        run.log_metric("asr_latency_s", asr_time_s, step=1)
        if hallucination:
            run.log_text(
                f"[HALLUCINATION DETECTED]\nRaw output:\n{transcript}",
                filename="asr_hallucination.txt",
            )
        else:
            run.log_text(transcript, filename="asr_transcript.txt")

        # ── [2] NER stage ─────────────────────────────────────────────
        species  = epi_fields.get("species")  or []
        symptoms = epi_fields.get("symptoms") or []
        run.log_params({
            "ner_species":    ", ".join(species)  if species  else "—",
            "ner_symptoms":   ", ".join(symptoms) if symptoms else "—",
            "ner_mortality":  epi_fields.get("mortality_count") or "—",
            "ner_affected":   epi_fields.get("affected_count")  or "—",
            "ner_location":   epi_fields.get("location")        or "—",
            "ner_timeframe":  epi_fields.get("timeframe")       or "—",
            "ner_reporter":   epi_fields.get("reporter_role")   or "—",
        })
        run.log_metric("ner_latency_s", ner_time_s, step=2)

        # ── [3] Router stage ──────────────────────────────────────────
        run.log_params({
            "router_domain":     domain,
            "router_off_topic":  off_topic,
        })
        run.log_metric("router_confidence",  confidence,   step=3)
        run.log_metric("router_latency_s",   route_time_s, step=3)

        # log all 6 domain scores as individual metrics for comparison
        if all_scores:
            for d, score in all_scores.items():
                run.log_metric(f"score_{d}", float(score), step=3)

        # ── [4] RAG stage ─────────────────────────────────────────────
        run.log_params({"rag_chunks_retrieved": len(rag_chunks)})
        run.log_metric("rag_latency_s", rag_time_s, step=4)
        if rag_chunks:
            rag_text = "\n\n---\n\n".join(
                f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(rag_chunks)
            )
            run.log_text(rag_text, filename="rag_sources.txt")

        # ── [5] LLM Expert stage ──────────────────────────────────────
        run.log_params({
            "llm_risk_level": risk_level,
            "llm_report_flag": report_flag,
        })
        run.log_metric("llm_latency_s",      llm_time_s,              step=5)
        run.log_metric("llm_response_words",  len(full_response.split()), step=5)
        run.log_text(full_response, filename="llm_assessment.txt")

        # ── [6] TTS stage ─────────────────────────────────────────────
        run.log_params({"tts_audio_chunks": tts_chunks})
        run.log_metric("tts_latency_s", tts_time_s, step=6)

        # ── Summary metrics ───────────────────────────────────────────
        run.log_metrics(
            {
                "total_latency_s": total_time,
                "pipeline_stages": 6 - int(hallucination),  # ASR halluci = pipeline cut short
            },
            step=7,
            system_metrics=["cpu", "ram"],  # attach hardware stats to final embed
        )

    dflow.save()   # persist thread ID → resume after restart
    dflow.finish() # flush async queue


# ══════════════════════════════════════════════════════════════════════════════
#  RETRAIN LOGGER: log_router_retrain()
#  Call from models/router.py after each training run.
# ══════════════════════════════════════════════════════════════════════════════

def log_router_retrain(
    *,
    n_samples:       int,
    domain_counts:   dict,   # {domain: count}
    cv_f1:           float,  # cross-val macro F1
    train_time_s:    float,
    model_path:      str     = "models/router.pkl",
    embedder:        str     = "all-MiniLM-L6-v2",
    hidden_layers:   tuple   = (128, 64),
    extra_data_path: str     = "data/router_training.jsonl",
):
    """
    Log a MoE router retraining event to Discord Forum.
    Thread title: Router Retrain · {timestamp}
    """
    dflow = _make_dflow()

    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name    = f"Router Retrain · {timestamp}"
    description = f"n={n_samples} samples | F1={cv_f1:.3f} | {train_time_s:.1f}s"

    with dflow.start_forum_run(run_name, description=description) as run:

        run.set_tag("event_type", "router_retrain")
        run.set_tag("timestamp",  timestamp)

        # Architecture params
        run.log_params({
            "embedder":          embedder,
            "hidden_layers":     str(hidden_layers),
            "total_samples":     n_samples,
            "extra_data_source": extra_data_path,
        })

        # Per-domain sample counts
        run.log_params({f"count_{d}": c for d, c in domain_counts.items()})

        # Performance metric
        run.log_metric("cv_f1_macro", cv_f1, step=1)
        run.log_metric("train_time_s", train_time_s, step=1)

        # Upload the trained model artifact
        if Path(model_path).exists():
            run.log_artifact(model_path)

        # Upload training data snapshot
        if Path(extra_data_path).exists():
            run.log_artifact(extra_data_path)

        run.log_metrics(
            {"cv_f1_macro": cv_f1},
            step=1,
            system_metrics=["cpu", "ram"],
        )

    dflow.save()
    dflow.finish()


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: log_hallucination_event()
#  Call from /upload when the hallucination guard fires.
# ══════════════════════════════════════════════════════════════════════════════

def log_hallucination_event(*, raw_transcript: str, asr_time_s: float):
    """Lightweight log — just posts transcript + time to a Forum thread."""
    dflow = _make_dflow()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name  = f"🛑 Hallucination · {timestamp}"

    with dflow.start_forum_run(run_name, description="ASR guard fired — garbled audio") as run:
        run.set_tag("event_type", "hallucination")
        run.log_metric("asr_latency_s", asr_time_s, step=1)
        run.log_text(raw_transcript, filename="rejected_transcript.txt")

    dflow.save()
    dflow.finish()


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST (dry-run)
#  python3 scripts/discord_logger.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running dry-run test (no real webhook needed)...\n")

    log_pipeline_run(
        transcript      = "Three chickens died overnight with cyanotic combs and convulsions near the pond.",
        input_method    = "voice",
        asr_time_s      = 1.42,
        asr_backend     = "mlx",
        whisper_size    = "base",
        hallucination   = False,
        ner_time_s      = 2.87,
        epi_fields      = {
            "species":         ["chicken"],
            "symptoms":        ["cyanotic combs", "convulsions"],
            "mortality_count": 3,
            "affected_count":  None,
            "location":        "near the pond",
            "timeframe":       "overnight",
            "reporter_role":   "farmer",
            "raw_summary":     "Three chickens died overnight with cyanotic combs.",
        },
        route_time_s    = 0.22,
        domain          = "avian_flu",
        confidence      = 0.81,
        all_scores      = {
            "avian_flu":     0.81,
            "nipah_hendra":  0.07,
            "leptospirosis": 0.05,
            "fmd":           0.04,
            "rabies":        0.02,
            "general":       0.01,
        },
        off_topic       = False,
        rag_time_s      = 0.01,
        rag_chunks      = [
            "HPAI H5N1 causes sudden death with up to 100% mortality in poultry...",
            "Key differentials: Newcastle disease, fowl cholera...",
            "Outbreak response: quarantine, culling, 10 km movement ban...",
        ],
        llm_time_s      = 9.1,
        risk_level      = "HIGH",
        report_flag     = True,
        full_response   = (
            "Based on the neurological signs and cyanosis in your flock, this is "
            "consistent with Highly Pathogenic Avian Influenza (HPAI H5N1). "
            "RISK LEVEL: HIGH. Isolate surviving birds immediately and contact your "
            "District Livestock Officer. Report to authorities."
        ),
        tts_chunks      = 3,
        tts_time_s      = 4.2,
    )

    print("\n--- Router retrain test ---\n")

    log_router_retrain(
        n_samples     = 100,
        domain_counts = {
            "avian_flu":     27,
            "fmd":           25,
            "general":       25,
            "leptospirosis": 26,
            "nipah_hendra":  30,
            "rabies":        27,
        },
        cv_f1         = 0.740,
        train_time_s  = 12.3,
    )

    print("\nDry-run complete. Set DISCORD_WEBHOOK env var to send to real Discord.")

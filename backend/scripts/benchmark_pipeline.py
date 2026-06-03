"""
scripts/benchmark_pipeline.py
=============================
End-to-end pipeline latency benchmark (ICONIP 2026).

Drives a *running* ZoonoMoE server and measures the real per-stage latency of the
full pipeline: NER -> Router (cascade) -> RAG -> Expert LLM -> TTS. Optionally
also times ASR if an audio file is provided. Because the heavy models (Whisper,
vLLM, Kokoro) need a GPU, this script must be run against a live server on the
GPU host (it does not import the models itself):

    # on the GPU machine, with the server already up on :7860
    python3 scripts/benchmark_pipeline.py --base-url http://localhost:7860 --runs 5

It reports, per stage, mean +/- std and p50/p95 over `runs` repetitions of each
of the six canonical test reports, plus the streaming-specific metrics that
matter for spoken UX: time-to-first-token (TTFT) and time-to-first-audio (TTFA).

Outputs (scripts/eval_out/):
  pipeline_results.json   raw per-run measurements + aggregates
  pipeline_table.md       LaTeX/MD-ready end-to-end latency table

Stages measured (seconds):
  ner    : NER / epi-field extraction (LLM)          [/analyze]
  route  : domain router (cascade)                   [/analyze]  (+ stage 1|2)
  rag    : FAISS retrieval, top-3                     [/analyze]
  ttft   : expert LLM time-to-first-token            [/stream]
  llm    : expert LLM total generation               [/stream]
  tts    : speech synthesis (sum over sentences)     [/stream]
  ttfa   : time-to-first-audio                       [/stream]
  total  : analyze + stream wall clock (client-side)
"""
import sys, json, time, argparse, statistics as st
from pathlib import Path
from urllib import request as urlreq

OUT = Path(__file__).resolve().parent / "eval_out"
OUT.mkdir(exist_ok=True)

TEST_INPUTS = [
    ("avian_flu",     "thirty chickens died this morning with purple combs and "
                      "twisted necks, labored breathing"),
    ("fmd",           "my cattle have blisters on tongue and feet, limping badly "
                      "and salivating heavily"),
    ("nipah_hendra",  "pig farmer here, pigs died suddenly overnight, two workers "
                      "now have fever and confusion"),
    ("leptospirosis", "five rice farmers have fever, muscle pain, red eyes after "
                      "wading in flooded paddy fields"),
    ("rabies",        "a stray dog bit two children, foaming at the mouth, running "
                      "in circles before collapsing"),
    ("general",       "hi, how do I protect myself when working near livestock"),
]


def _post(url, payload, timeout=300):
    data = json.dumps(payload).encode()
    req = urlreq.Request(url, data=data,
                         headers={"Content-Type": "application/json"})
    return urlreq.urlopen(req, timeout=timeout)


def call_analyze(base, transcript):
    t0 = time.time()
    resp = _post(f"{base}/analyze", {"transcript": transcript})
    body = json.loads(resp.read())
    body["_client_analyze_sec"] = time.time() - t0
    return body


def call_stream(base, domain, epi_fields, rag_chunks, transcript=""):
    """Consume the SSE stream and return the final 'done' event's latency dict."""
    t0 = time.time()
    resp = _post(f"{base}/stream", {
        "domain": domain, "epi_fields": epi_fields,
        "rag_chunks": rag_chunks, "transcript": transcript})
    latency = {}
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        try:
            evt = json.loads(line[5:].strip())
        except Exception:
            continue
        if evt.get("type") == "done":
            latency = evt.get("latency", {})
    latency["_client_stream_sec"] = time.time() - t0
    return latency


def agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    s = sorted(values)
    return dict(
        mean=round(st.mean(values), 3),
        std=round(st.pstdev(values), 3) if len(values) > 1 else 0.0,
        p50=round(s[int(0.50 * (len(s) - 1))], 3),
        p95=round(s[int(0.95 * (len(s) - 1))], 3),
        n=len(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:7860")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1,
                    help="Discarded warmup runs per input (model/cache warm-up)")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Benchmarking {base}  ({args.runs} runs x {len(TEST_INPUTS)} inputs, "
          f"{args.warmup} warmup)")
    # connectivity check
    try:
        urlreq.urlopen(f"{base}/health", timeout=10)
    except Exception as e:
        print(f"ERROR: cannot reach {base}/health ({e}). Is the server up?")
        sys.exit(1)

    buckets = {k: [] for k in
               ["ner", "route", "rag", "ttft", "ttfa", "llm", "tts", "total"]}
    stage_counts = {1: 0, 2: 0, None: 0}
    runs_log = []

    for domain_hint, transcript in TEST_INPUTS:
        for r in range(args.runs + args.warmup):
            a = call_analyze(base, transcript)
            tmg = a.get("timing", {})
            s = call_stream(base, a["domain"], a.get("epi_fields", {}),
                            a.get("rag_chunks", []), transcript=transcript)
            if r < args.warmup:
                continue
            total = a["_client_analyze_sec"] + s.get("_client_stream_sec", 0)
            rec = dict(
                input=domain_hint, routed_to=a["domain"],
                route_stage=a.get("route_stage"),
                router_backend=a.get("router_backend"),
                ner=tmg.get("ner_s"), route=tmg.get("route_s"),
                rag=tmg.get("rag_s"),
                ttft=s.get("ttft_s"), ttfa=s.get("ttfa_s"),
                llm=s.get("llm_s"), tts=s.get("tts_s"),
                audio_chunks=s.get("audio_chunks"),
                total=round(total, 3))
            runs_log.append(rec)
            for k in buckets:
                buckets[k].append(rec.get(k))
            stage_counts[a.get("route_stage")] = \
                stage_counts.get(a.get("route_stage"), 0) + 1
            print(f"  {domain_hint:13s} -> {a['domain']:13s} "
                  f"stage{a.get('route_stage')} "
                  f"ner={rec['ner']} llm={rec['llm']} tts={rec['tts']} "
                  f"ttft={rec['ttft']} total={rec['total']}")

    aggregates = {k: agg(v) for k, v in buckets.items()}
    n_stage1 = stage_counts.get(1, 0)
    n_total = sum(v for k, v in stage_counts.items() if k is not None) or 1
    results = dict(
        base_url=base, n_runs=args.runs, warmup=args.warmup,
        n_measurements=len(runs_log),
        stage1_fraction=round(n_stage1 / n_total, 3),
        stage_counts={str(k): v for k, v in stage_counts.items()},
        aggregates=aggregates, measurements=runs_log)
    (OUT / "pipeline_results.json").write_text(json.dumps(results, indent=2))

    # Markdown table
    def row(label, key, unit="s"):
        a = aggregates.get(key)
        if not a:
            return f"| {label} | n/a | n/a | n/a |"
        return (f"| {label} | {a['mean']:.3f} ± {a['std']:.3f} | "
                f"{a['p50']:.3f} | {a['p95']:.3f} |")
    lines = [
        f"End-to-end pipeline latency on `{base}` "
        f"({len(runs_log)} measurements, {args.runs} runs x "
        f"{len(TEST_INPUTS)} inputs; {args.warmup} warmup discarded).",
        f"Cascade Stage-1 resolved {100*results['stage1_fraction']:.0f}% of "
        f"queries (encoder skipped).", "",
        "| Stage | Mean ± SD (s) | p50 (s) | p95 (s) |",
        "|---|---|---|---|",
        row("NER (epi extraction)", "ner"),
        row("Router (cascade)", "route"),
        row("RAG (FAISS top-3)", "rag"),
        row("Expert LLM — TTFT", "ttft"),
        row("Expert LLM — total", "llm"),
        row("TTS (synthesis)", "tts"),
        row("Time-to-first-audio", "ttfa"),
        row("End-to-end total", "total"),
    ]
    table = "\n".join(lines)
    (OUT / "pipeline_table.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nWritten to {OUT}/pipeline_results.json and pipeline_table.md")


if __name__ == "__main__":
    main()

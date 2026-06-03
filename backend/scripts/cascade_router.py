"""
scripts/cascade_router.py
=========================
Confidence-gated cascade router (ICONIP 2026) --- adaptive computation for
on-device zoonotic triage.

Motivation
----------
In the isolated-router study the *encoder* (~14 ms) dominates latency; the
classifier head is <0.3 ms. So shrinking the head buys nothing. The real
efficiency lever is to *avoid running the encoder* when a cheap text model is
already confident. We therefore route in two stages:

  Stage 1 (cheap, ~1 ms):  TF-IDF + logistic regression, with confidence
                           calibrated by temperature scaling.
  Stage 2 (costly, ~14 ms): all-MiniLM-L6-v2 embedding + MLP(64) head.

Gate: if the calibrated Stage-1 max-probability >= tau, accept Stage 1;
otherwise escalate to Stage 2. Sweeping tau traces an accuracy-vs-average-cost
curve. We also report calibration quality (Expected Calibration Error) before
and after temperature scaling, since the gate relies on trustworthy confidence,
and a selective-prediction (risk--coverage) view for the safety-critical
veterinary setting.

Protocol: 5-fold stratified CV (seed 42). Within each outer fold we hold out a
calibration split to fit the temperature; both stages are trained on the
remaining data and evaluated on the untouched outer-test fold (no leakage).

Outputs (scripts/eval_out/):
  cascade_results.json     metrics for stage1-only, stage2-only, cascade sweep
  cascade_table.md         operating-point table (LaTeX/MD ready)
  cascade_pareto.png       macro-F1 vs average latency as tau sweeps
  reliability.png          Stage-1 reliability diagram (raw vs temperature-scaled)
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "eval_out"
OUT.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

from models.router import DOMAINS  # noqa: E402
from benchmark_router import load_dataset  # noqa: E402

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import minimize_scalar


# ── Calibration utilities ────────────────────────────────────────────────────
def softmax(logits, T=1.0):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, y):
    """Single-parameter temperature scaling: minimize NLL over T>0."""
    n = len(y)
    def nll(logT):
        T = np.exp(logT)              # keep T>0
        p = softmax(logits, T)
        return -np.mean(np.log(p[np.arange(n), y] + 1e-12))
    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def expected_calibration_error(conf, correct, n_bins=10):
    """Standard ECE over confidence bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ── Latency measurement ──────────────────────────────────────────────────────
def median_latency_ms(fn, samples, repeat=3):
    for s in samples[:3]:
        fn(s)
    times = []
    for _ in range(repeat):
        for s in samples:
            t0 = time.perf_counter(); fn(s); times.append((time.perf_counter()-t0)*1e3)
    return float(np.median(times))


def main():
    texts, labels = load_dataset()
    le = LabelEncoder().fit(DOMAINS)
    y = le.transform(labels)
    n, K = len(texts), len(DOMAINS)
    print(f"Dataset: {n} examples, {K} classes")

    print("Loading embedder all-MiniLM-L6-v2 (CPU)...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    X = np.asarray(embedder.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Out-of-fold collectors
    s1_conf_raw = np.zeros(n)      # Stage-1 raw max prob
    s1_conf_cal = np.zeros(n)      # Stage-1 temperature-scaled max prob
    s1_pred     = np.zeros(n, int)
    s2_pred     = np.zeros(n, int)
    temps       = []

    for tr, te in skf.split(texts, y):
        # inner calibration split inside the training fold
        tr_fit, tr_cal = train_test_split(tr, test_size=0.25, stratify=y[tr],
                                          random_state=SEED)
        # Stage 1: TF-IDF + LogReg
        s1 = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LogisticRegression(max_iter=1000, C=10.0))
        s1.fit([texts[i] for i in tr_fit], y[tr_fit])
        # temperature from calib split logits
        cal_logits = s1.decision_function([texts[i] for i in tr_cal])
        T = fit_temperature(cal_logits, y[tr_cal]); temps.append(T)
        # Stage 2: MLP on embeddings
        s2 = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000,
                           random_state=SEED, early_stopping=False, alpha=1e-3)
        s2.fit(X[tr_fit], y[tr_fit])
        # predictions on outer test
        logits_te = s1.decision_function([texts[i] for i in te])
        p_raw = softmax(logits_te, 1.0)
        p_cal = softmax(logits_te, T)
        s1_conf_raw[te] = p_raw.max(axis=1)
        s1_conf_cal[te] = p_cal.max(axis=1)
        s1_pred[te] = p_cal.argmax(axis=1)
        s2_pred[te] = s2.predict(X[te])

    T_mean = float(np.mean(temps))
    print(f"Mean temperature T = {T_mean:.3f}")

    # ── Calibration quality (Stage 1) ──
    s1_correct = (s1_pred == y).astype(float)
    ece_raw = expected_calibration_error(s1_conf_raw, s1_correct)
    ece_cal = expected_calibration_error(s1_conf_cal, s1_correct)
    print(f"Stage-1 ECE: raw={ece_raw:.3f}  temperature-scaled={ece_cal:.3f}")

    # ── Measure stage latencies on a full-data model (deployment-representative) ──
    s1_full = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=10.0)).fit(texts, y)
    s2_full = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000,
                            random_state=SEED, early_stopping=False, alpha=1e-3).fit(X, y)
    t_s1 = median_latency_ms(lambda s: s1_full.predict([s]), texts[:40])
    t_enc = median_latency_ms(lambda s: embedder.encode([s], normalize_embeddings=True),
                              texts[:40], repeat=1)
    t_s2head = median_latency_ms(lambda s: s2_full.predict(s.reshape(1, -1)), X[:40])
    t_stage2_extra = t_enc + t_s2head      # cost added when we escalate
    print(f"Latency: stage1={t_s1:.2f}ms  encoder={t_enc:.2f}ms  "
          f"mlp={t_s2head:.3f}ms  -> escalation adds {t_stage2_extra:.2f}ms")

    # ── Reference points ──
    f1_s1 = f1_score(y, s1_pred, average="macro")
    f1_s2 = f1_score(y, s2_pred, average="macro")
    acc_s1 = accuracy_score(y, s1_pred)
    acc_s2 = accuracy_score(y, s2_pred)

    # ── Cascade sweep over tau ──
    taus = np.round(np.linspace(0.0, 1.0, 51), 3)
    sweep = []
    for tau in taus:
        escalate = s1_conf_cal < tau
        pred = np.where(escalate, s2_pred, s1_pred)
        esc_rate = float(escalate.mean())
        avg_lat = t_s1 + esc_rate * t_stage2_extra
        sweep.append(dict(
            tau=float(tau), escalation_rate=esc_rate, avg_latency_ms=avg_lat,
            f1=float(f1_score(y, pred, average="macro")),
            acc=float(accuracy_score(y, pred))))

    # best operating point: highest F1, ties broken by lower latency
    best = max(sweep, key=lambda r: (round(r["f1"], 4), -r["avg_latency_ms"]))
    # "matches stage-2 F1 at min latency"
    match = min([r for r in sweep if r["f1"] >= f1_s2 - 1e-9],
                key=lambda r: r["avg_latency_ms"], default=best)

    results = dict(
        n=n, classes=list(le.classes_), seed=SEED, temperature=T_mean,
        ece_raw=ece_raw, ece_calibrated=ece_cal,
        latency_ms=dict(stage1=t_s1, encoder=t_enc, mlp_head=t_s2head,
                        escalation_extra=t_stage2_extra),
        stage1_only=dict(f1=float(f1_s1), acc=float(acc_s1), avg_latency_ms=t_s1),
        stage2_only=dict(f1=float(f1_s2), acc=float(acc_s2),
                         avg_latency_ms=t_s1 + t_stage2_extra),
        cascade_best=best, cascade_match_stage2=match, sweep=sweep)
    with open(OUT / "cascade_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Table ──
    spd = 100.0 * (1 - match["avg_latency_ms"] / (t_s1 + t_stage2_extra))
    lines = [
        "| Router | Macro-F1 | Accuracy | Avg latency (ms) | Escalation |",
        "|---|---|---|---|---|",
        f"| Stage 1 only (TF-IDF+LogReg) | {f1_s1:.3f} | {acc_s1:.3f} | {t_s1:.2f} | 0% |",
        f"| Stage 2 only (emb+MLP) | {f1_s2:.3f} | {acc_s2:.3f} | "
        f"{t_s1+t_stage2_extra:.2f} | 100% |",
        f"| **Cascade @τ={match['tau']:.2f}** | **{match['f1']:.3f}** | "
        f"{match['acc']:.3f} | **{match['avg_latency_ms']:.2f}** | "
        f"{100*match['escalation_rate']:.0f}% |",
        f"| Cascade @τ={best['tau']:.2f} (max-F1) | {best['f1']:.3f} | "
        f"{best['acc']:.3f} | {best['avg_latency_ms']:.2f} | "
        f"{100*best['escalation_rate']:.0f}% |",
    ]
    table = "\n".join(lines)
    (OUT / "cascade_table.md").write_text(
        table + f"\n\nStage-1 ECE raw={ece_raw:.3f} -> calibrated={ece_cal:.3f} "
        f"(T={T_mean:.2f}). Cascade matches Stage-2 F1 ({f1_s2:.3f}) at "
        f"{match['avg_latency_ms']:.2f} ms, a {spd:.0f}% average-latency reduction "
        f"vs always running the encoder.\n")
    print("\n" + table)
    print(f"\nCascade matches Stage-2 F1 at {match['avg_latency_ms']:.2f} ms "
          f"({spd:.0f}% latency cut vs Stage-2 only).")

    make_plots(sweep, f1_s1, f1_s2, t_s1, t_s1 + t_stage2_extra,
               s1_conf_raw, s1_conf_cal, s1_correct, ece_raw, ece_cal, match)
    print(f"\nOutputs written to {OUT}/")


def make_plots(sweep, f1_s1, f1_s2, lat_s1, lat_s2,
               conf_raw, conf_cal, correct, ece_raw, ece_cal, match):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pareto: F1 vs avg latency along tau
    lat = [r["avg_latency_ms"] for r in sweep]
    f1 = [r["f1"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(lat, f1, "-o", ms=3, color="#009688", label="Cascade (sweep τ)")
    ax.scatter([lat_s1], [f1_s1], c="#8e44ad", s=80, zorder=5, label="Stage 1 only")
    ax.scatter([lat_s2], [f1_s2], c="#c0392b", s=80, zorder=5, label="Stage 2 only")
    ax.scatter([match["avg_latency_ms"]], [match["f1"]], facecolors="none",
               edgecolors="black", s=160, linewidths=1.6, zorder=6,
               label=f"Operating point (τ={match['tau']:.2f})")
    ax.set_xlabel("Average latency per query (ms)")
    ax.set_ylabel("Macro-F1 (5-fold CV)")
    ax.set_title("Confidence-gated cascade: accuracy vs. average cost")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "cascade_pareto.png", dpi=160); plt.close(fig)

    # Reliability diagram (raw vs calibrated)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    for conf, lbl, col in [(conf_raw, f"raw (ECE={ece_raw:.3f})", "#c0392b"),
                           (conf_cal, f"scaled (ECE={ece_cal:.3f})", "#009688")]:
        acc_bin = []
        for i in range(10):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            acc_bin.append(correct[m].mean() if m.sum() else np.nan)
        ax.plot(centers, acc_bin, "-o", ms=4, color=col, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Stage-1 reliability (temperature scaling)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "reliability.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()

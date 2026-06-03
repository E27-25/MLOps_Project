"""
scripts/benchmark_router.py
===========================
ZoonoMoE — Router Trade-off Benchmark (ICONIP 2026 experiments)

Compares the lightweight neural MoE router (MiniLM embedding + MLP) against a
spectrum of baselines on the 6-way zoonotic-domain routing task, reporting
*both* accuracy (macro-F1, accuracy) and deployment cost (inference latency,
model size). The output is the accuracy-vs-latency Pareto picture that
motivates using a small on-device router instead of an LLM call per query.

Methods compared
----------------
  1. Majority           — predicts the most frequent class (chance floor)
  2. Keyword/regex      — hand-written disease keyword rules (rule baseline)
  3. TF-IDF + LogReg    — classic shallow text classifier (no embeddings)
  4. Centroid (cosine)  — zero-shot: nearest domain centroid in MiniLM space
  5. kNN (k=5)          — non-parametric on MiniLM embeddings
  6. LogReg on emb      — linear head on MiniLM embeddings
  7. MLP (128,64)       — *our router* (MiniLM emb -> MLP)  [the deployed model]
  8. LLM zero-shot      — optional, prompt an LLM to pick a domain (--with-llm)

Evaluation
----------
  - Stratified 5-fold cross-validation (fixed seed) over the full labelled set
    (SEED_EXAMPLES + data/router_training.jsonl).
  - Reports mean +/- std of macro-F1 and accuracy across folds.
  - Latency: median per-query wall-clock at inference (embedding excluded and
    included variants reported), averaged over the eval set, single CPU thread.
  - Model size: serialized classifier size on disk (+ shared embedder size noted
    separately, since it is amortized across all embedding-based methods).
  - Confusion matrix for the deployed MLP router (out-of-fold predictions).

Outputs (written to scripts/eval_out/)
  - results.json          machine-readable metrics for every method
  - results_table.md      LaTeX/Markdown-ready comparison table
  - confusion_matrix.png  6x6 normalized confusion matrix (MLP, OOF)
  - pareto.png            macro-F1 vs latency scatter (the money figure)
  - per_class_f1.png      per-domain F1 for the MLP router

Usage
-----
  python3.12 scripts/benchmark_router.py
  python3.12 scripts/benchmark_router.py --with-llm   # also benchmark LLM router

Reproducibility: SEED=42 everywhere; sentence-transformers/all-MiniLM-L6-v2;
scikit-learn StratifiedKFold. Run on CPU for deployment-representative latency.
"""

import os
import re
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # backend/
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "eval_out"
OUT.mkdir(exist_ok=True)

SEED = 42
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
np.random.seed(SEED)

from models.router import SEED_EXAMPLES, DOMAINS   # noqa: E402


# ── Data loading ─────────────────────────────────────────────────────────────
def load_dataset():
    """Combine SEED_EXAMPLES + data/router_training.jsonl into (texts, labels)."""
    texts = [t for t, _ in SEED_EXAMPLES]
    labels = [d for _, d in SEED_EXAMPLES]

    jsonl = ROOT / "data" / "router_training.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["domain"])
    return texts, labels


# ── Keyword / regex baseline ─────────────────────────────────────────────────
KEYWORD_RULES = {
    "avian_flu":     ["chicken", "poultry", "duck", "hen", "comb", "bird",
                      "geese", "goose", "turkey", "fowl", "wattle", "flock"],
    "rabies":        ["dog", "bit", "bite", "foam", "rabid", "aggressi",
                      "hydrophobia", "saliva", "fox", "bat", "raccoon",
                      "stray", "scratch"],
    "fmd":           ["blister", "hoof", "hooves", "vesic", "lame", "mouth",
                      "tongue", "cattle", "cow", "pig", "sheep", "goat",
                      "buffalo", "drool", "ulcer", "feet"],
    "nipah_hendra":  ["horse", "encephal", "fruit bat", "respiratory",
                      "snort", "neurolog", "pig pen", "handler", "stable",
                      "mango", "orchard", "confusion"],
    "leptospirosis": ["flood", "rice", "jaundice", "rat", "rodent", "urine",
                      "kidney", "renal", "yellow", "wading", "paddy",
                      "floodwater", "rain"],
    "general":       ["not sure", "advice", "prevent", "worried", "unclear",
                      "general", "what should", "understand", "concern"],
}


def keyword_predict(text):
    t = text.lower()
    scores = {d: 0 for d in DOMAINS}
    for d, kws in KEYWORD_RULES.items():
        for kw in kws:
            if kw in t:
                scores[d] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best


# ── Timing helper ─────────────────────────────────────────────────────────────
def median_latency_ms(fn, samples, repeat=3):
    """Median per-call latency in ms. fn takes a single sample, returns label."""
    times = []
    # warmup
    for s in samples[:3]:
        fn(s)
    for _ in range(repeat):
        for s in samples:
            t0 = time.perf_counter()
            fn(s)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ── Main benchmark ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="Also benchmark an LLM zero-shot router (needs OPENAI/vLLM).")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                                  classification_report)

    texts, labels = load_dataset()
    print(f"Dataset: {len(texts)} examples")
    dist = Counter(labels)
    for d in DOMAINS:
        print(f"  {d:16s}: {dist.get(d,0)}")

    le = LabelEncoder().fit(DOMAINS)
    y = le.transform(labels)
    classes = list(le.classes_)

    print(f"\nLoading embedder {EMB_MODEL} (CPU)...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMB_MODEL, device="cpu")
    t0 = time.perf_counter()
    X = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    X = np.asarray(X, dtype=np.float32)
    emb_total_s = time.perf_counter() - t0
    emb_latency_ms = median_latency_ms(
        lambda s: embedder.encode([s], normalize_embeddings=True), texts[:40], repeat=1)
    print(f"Embedded {len(texts)} texts in {emb_total_s:.1f}s "
          f"(median single-encode {emb_latency_ms:.1f} ms)")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    results = {}

    def cv_emb(name, make_clf):
        """5-fold CV macro-F1/acc for an embedding-based sklearn classifier."""
        f1s, accs = [], []
        for tr, te in skf.split(X, y):
            clf = make_clf()
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            f1s.append(f1_score(y[te], pred, average="macro"))
            accs.append(accuracy_score(y[te], pred))
        # fit full for size + latency
        clf = make_clf(); clf.fit(X, y)
        blob = pickle.dumps(clf)
        size_kb = len(blob) / 1024.0
        clf_latency = median_latency_ms(lambda s: clf.predict(s.reshape(1, -1)), X[:40])
        results[name] = dict(
            f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
            acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
            clf_size_kb=size_kb,
            latency_clf_ms=clf_latency,
            latency_total_ms=clf_latency + emb_latency_ms,
            family="embedding",
        )
        print(f"  {name:22s} F1={np.mean(f1s):.3f}±{np.std(f1s):.3f} "
              f"acc={np.mean(accs):.3f} clf={size_kb:.0f}KB "
              f"lat={clf_latency+emb_latency_ms:.1f}ms")
        return clf

    print("\n=== Embedding-based methods (5-fold CV) ===")
    cv_emb("Centroid (cosine)", lambda: NearestCentroidCosine())
    cv_emb("kNN (k=5)", lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine"))
    cv_emb("LogReg (emb)", lambda: LogisticRegression(max_iter=1000, C=10.0))
    cv_emb("MLP (128,64) early-stop", lambda: MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED,
        early_stopping=True, validation_fraction=0.15))
    mlp_full = cv_emb("MLP (64,) [ours]", lambda: MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=1000, random_state=SEED,
        early_stopping=False, alpha=1e-3))

    # ── Non-embedding baselines ──
    print("\n=== Text / rule baselines (5-fold CV) ===")

    # Majority
    f1s, accs = [], []
    for tr, te in skf.split(texts, y):
        dc = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(tr), 1)), y[tr])
        pred = dc.predict(np.zeros((len(te), 1)))
        f1s.append(f1_score(y[te], pred, average="macro"))
        accs.append(accuracy_score(y[te], pred))
    results["Majority"] = dict(
        f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
        acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
        clf_size_kb=0.0, latency_clf_ms=0.0, latency_total_ms=0.0, family="trivial")
    print(f"  {'Majority':22s} F1={np.mean(f1s):.3f} acc={np.mean(accs):.3f}")

    # Keyword/regex (deterministic, no training -> evaluate on full set)
    kw_pred = np.array([le.transform([keyword_predict(t)])[0] for t in texts])
    kw_lat = median_latency_ms(lambda s: keyword_predict(s), texts[:40])
    results["Keyword/regex"] = dict(
        f1_mean=float(f1_score(y, kw_pred, average="macro")), f1_std=0.0,
        acc_mean=float(accuracy_score(y, kw_pred)), acc_std=0.0,
        clf_size_kb=0.0, latency_clf_ms=kw_lat, latency_total_ms=kw_lat,
        family="rule")
    print(f"  {'Keyword/regex':22s} F1={results['Keyword/regex']['f1_mean']:.3f} "
          f"acc={results['Keyword/regex']['acc_mean']:.3f} lat={kw_lat:.2f}ms")

    # TF-IDF + LogReg
    f1s, accs = [], []
    for tr, te in skf.split(texts, y):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LogisticRegression(max_iter=1000, C=10.0))
        pipe.fit([texts[i] for i in tr], y[tr])
        pred = pipe.predict([texts[i] for i in te])
        f1s.append(f1_score(y[te], pred, average="macro"))
        accs.append(accuracy_score(y[te], pred))
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=10.0)).fit(texts, y)
    tfidf_lat = median_latency_ms(lambda s: pipe.predict([s]), texts[:40])
    results["TF-IDF + LogReg"] = dict(
        f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
        acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
        clf_size_kb=len(pickle.dumps(pipe)) / 1024.0,
        latency_clf_ms=tfidf_lat, latency_total_ms=tfidf_lat, family="text")
    print(f"  {'TF-IDF + LogReg':22s} F1={np.mean(f1s):.3f}±{np.std(f1s):.3f} "
          f"acc={np.mean(accs):.3f} lat={tfidf_lat:.2f}ms")

    # ── Optional LLM zero-shot router ──
    if args.with_llm:
        try:
            run_llm_router(texts, y, le, results)
        except Exception as e:
            print(f"  [LLM router skipped: {e}]")

    # ── Out-of-fold confusion matrix + per-class F1 for the MLP router ──
    print("\n=== MLP router: out-of-fold analysis ===")
    oof = cross_val_predict(
        MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000,
                      random_state=SEED, early_stopping=False, alpha=1e-3),
        X, y, cv=skf)
    cm = confusion_matrix(y, oof)
    print(classification_report(y, oof, target_names=classes, digits=3))
    per_class_f1 = f1_score(y, oof, average=None)
    results["_meta"] = dict(
        n_examples=len(texts), n_classes=len(classes), classes=classes,
        embedder=EMB_MODEL, embedder_latency_ms=emb_latency_ms,
        seed=SEED, folds=args.folds,
        per_class_f1={c: float(v) for c, v in zip(classes, per_class_f1)},
        class_distribution={d: dist.get(d, 0) for d in DOMAINS},
    )

    # ── Persist ──
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    write_table(results)
    make_plots(results, cm, classes, per_class_f1)
    print(f"\nAll outputs written to {OUT}/")


class NearestCentroidCosine:
    """Zero-shot per-domain centroid classifier on normalized embeddings
    (mirrors the router's cosine fallback)."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.cent = {}
        for c in self.classes_:
            v = X[y == c].mean(axis=0)
            self.cent[c] = v / (np.linalg.norm(v) + 1e-9)
        return self

    def predict(self, X):
        out = []
        for x in X:
            best, bs = None, -1e9
            for c, v in self.cent.items():
                s = float(np.dot(x, v))
                if s > bs:
                    bs, best = s, c
            out.append(best)
        return np.array(out)


def run_llm_router(texts, y, le, results):
    """Zero-shot LLM domain router via an OpenAI-compatible endpoint
    (set OPENAI_API_KEY/OPENAI_BASE_URL or LLM_BASE_URL+LLM_MODEL)."""
    from openai import OpenAI
    base = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    client = OpenAI(base_url=base) if base else OpenAI()
    sys_prompt = (
        "You are a routing classifier for zoonotic disease field reports. "
        "Reply with EXACTLY ONE of these labels and nothing else: "
        + ", ".join(DOMAINS) + ".")
    preds, lats = [], []
    for t in texts:
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": t}])
        lats.append((time.perf_counter() - t0) * 1000.0)
        ans = r.choices[0].message.content.strip().lower()
        match = next((d for d in DOMAINS if d in ans), "general")
        preds.append(match)
    from sklearn.metrics import f1_score, accuracy_score
    yp = le.transform(preds)
    results["LLM zero-shot"] = dict(
        f1_mean=float(f1_score(y, yp, average="macro")), f1_std=0.0,
        acc_mean=float(accuracy_score(y, yp)), acc_std=0.0,
        clf_size_kb=float("nan"),
        latency_clf_ms=float(np.median(lats)),
        latency_total_ms=float(np.median(lats)),
        family="llm", model=model)
    print(f"  {'LLM zero-shot':22s} F1={results['LLM zero-shot']['f1_mean']:.3f} "
          f"acc={results['LLM zero-shot']['acc_mean']:.3f} "
          f"lat={np.median(lats):.0f}ms")


def write_table(results):
    order = ["Majority", "Keyword/regex", "TF-IDF + LogReg", "Centroid (cosine)",
             "kNN (k=5)", "LogReg (emb)", "MLP (128,64) early-stop",
             "MLP (64,) [ours]", "LLM zero-shot"]
    lines = ["| Method | Macro-F1 | Accuracy | Latency (ms) | Model size |",
             "|---|---|---|---|---|"]
    for k in order:
        if k not in results:
            continue
        r = results[k]
        f1 = f"{r['f1_mean']:.3f}" + (f"±{r['f1_std']:.3f}" if r['f1_std'] else "")
        acc = f"{r['acc_mean']:.3f}"
        lat = f"{r['latency_total_ms']:.1f}" if r['latency_total_ms'] else "—"
        sz = ("—" if not r['clf_size_kb'] or r['clf_size_kb'] != r['clf_size_kb']
              else f"{r['clf_size_kb']:.0f} KB")
        lines.append(f"| {k} | {f1} | {acc} | {lat} | {sz} |")
    table = "\n".join(lines)
    (OUT / "results_table.md").write_text(table + "\n")
    print("\n" + table)


def make_plots(results, cm, classes, per_class_f1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Confusion matrix (row-normalized)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("MLP router — out-of-fold confusion (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Per-class F1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(classes, per_class_f1, color="#009688")
    ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.set_title("MLP router — per-domain F1 (OOF)")
    ax.axhline(np.mean(per_class_f1), ls="--", c="gray",
               label=f"macro={np.mean(per_class_f1):.3f}")
    ax.legend(); plt.xticks(rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(OUT / "per_class_f1.png", dpi=160)
    plt.close(fig)

    # Pareto: macro-F1 vs latency
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, r in results.items():
        if k.startswith("_") or r.get("latency_total_ms", 0) <= 0:
            continue
        x = max(r["latency_total_ms"], 0.05)
        ax.scatter(x, r["f1_mean"], s=70)
        ax.annotate(k, (x, r["f1_mean"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Median latency per query (ms, log scale)")
    ax.set_ylabel("Macro-F1 (5-fold CV)")
    ax.set_title("Accuracy vs. latency trade-off")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT / "pareto.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

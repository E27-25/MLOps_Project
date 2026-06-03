"""
scripts/ablation_router.py
==========================
Ablations for the ZoonoMoE router (ICONIP 2026).

Two studies:
  (A) Classifier head on frozen MiniLM embeddings:
        - MLP as deployed (early_stopping=True, val=0.15)   [current product]
        - MLP no early stopping
        - MLP (256,) / (128,64) / (64,) capacity sweep
        - LogReg, Linear SVM, kNN, NearestCentroid
      -> shows the deployed early-stopping config underfits on small data.

  (B) Embedding backbone sweep (head fixed = LogReg):
        - all-MiniLM-L6-v2 (384d, deployed)
        - all-MiniLM-L12-v2 (384d)
        - paraphrase-multilingual-MiniLM-L12-v2 (multilingual; matters for Thai)
        - BAAI/bge-small-en-v1.5 (384d)
      -> justifies the backbone choice / shows headroom.

5-fold stratified CV, SEED=42. Writes scripts/eval_out/ablation_*.{json,md}.
"""
import os, sys, json, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "eval_out"
OUT.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

from models.router import SEED_EXAMPLES, DOMAINS  # noqa: E402
from benchmark_router import load_dataset, NearestCentroidCosine  # noqa: E402

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import f1_score, accuracy_score


def cv(make_clf, X, y, skf):
    f1s, accs = [], []
    for tr, te in skf.split(X, y):
        clf = make_clf(); clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        f1s.append(f1_score(y[te], pred, average="macro"))
        accs.append(accuracy_score(y[te], pred))
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(accs))


def study_A(X, y, skf):
    heads = {
        "MLP (128,64) early-stop [deployed]": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED,
            early_stopping=True, validation_fraction=0.15),
        "MLP (128,64) no early-stop": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=1000, random_state=SEED,
            early_stopping=False, alpha=1e-3),
        "MLP (256,) no early-stop": lambda: MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=1000, random_state=SEED,
            early_stopping=False, alpha=1e-3),
        "MLP (64,) no early-stop": lambda: MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=1000, random_state=SEED,
            early_stopping=False, alpha=1e-3),
        "LogReg": lambda: LogisticRegression(max_iter=1000, C=10.0),
        "Linear SVM": lambda: LinearSVC(C=1.0),
        "kNN (k=5, cosine)": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "NearestCentroid (cosine)": lambda: NearestCentroidCosine(),
    }
    rows = []
    print("\n=== Study A: classifier head on frozen MiniLM-L6 ===")
    for name, mk in heads.items():
        f1, sd, acc = cv(mk, X, y, skf)
        rows.append((name, f1, sd, acc))
        print(f"  {name:38s} F1={f1:.3f}±{sd:.3f} acc={acc:.3f}")
    return rows


def study_B(texts, y, skf):
    backbones = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-small-en-v1.5",
    ]
    from sentence_transformers import SentenceTransformer
    rows = []
    print("\n=== Study B: embedding backbone (head=LogReg) ===")
    for name in backbones:
        try:
            emb = SentenceTransformer(name, device="cpu")
            X = np.asarray(emb.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False), dtype=np.float32)
            f1, sd, acc = cv(lambda: LogisticRegression(max_iter=1000, C=10.0),
                             X, y, skf)
            dim = X.shape[1]
            rows.append((name, dim, f1, sd, acc))
            print(f"  {name:55s} d={dim} F1={f1:.3f}±{sd:.3f} acc={acc:.3f}")
        except Exception as e:
            print(f"  {name:55s} SKIPPED ({str(e)[:50]})")
            rows.append((name, -1, float('nan'), float('nan'), float('nan')))
    return rows


def main():
    texts, labels = load_dataset()
    le = LabelEncoder().fit(DOMAINS)
    y = le.transform(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    X = np.asarray(emb.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False), dtype=np.float32)

    a = study_A(X, y, skf)
    b = study_B(texts, y, skf)

    # write markdown
    lines = ["### Study A — classifier head on frozen all-MiniLM-L6-v2 (5-fold CV)",
             "", "| Head | Macro-F1 | Accuracy |", "|---|---|---|"]
    for name, f1, sd, acc in a:
        lines.append(f"| {name} | {f1:.3f}±{sd:.3f} | {acc:.3f} |")
    lines += ["", "### Study B — embedding backbone (head = LogReg, 5-fold CV)",
              "", "| Backbone | Dim | Macro-F1 | Accuracy |", "|---|---|---|---|"]
    for name, dim, f1, sd, acc in b:
        sn = name.split("/")[-1]
        if dim < 0:
            lines.append(f"| {sn} | — | skipped | — |")
        else:
            lines.append(f"| {sn} | {dim} | {f1:.3f}±{sd:.3f} | {acc:.3f} |")
    (OUT / "ablation_tables.md").write_text("\n".join(lines) + "\n")
    with open(OUT / "ablation.json", "w") as f:
        json.dump({"study_A": a, "study_B": b}, f, indent=2)
    print(f"\nWritten to {OUT}/ablation_tables.md")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

"""
models/router.py
================
Mini MoE Router — classifies incoming zoonotic reports into disease domains.

Architecture:
  Text (+ optional epi fields) → Sentence embedding → MLP classifier
  → domain label + confidence + all_scores

Training:
  Run:  python models/router.py --train
  Uses seed data in SEED_EXAMPLES + any data/router_training.jsonl if present.

Domains:
  avian_flu | rabies | fmd | nipah_hendra | leptospirosis | general
"""

import os, json, pickle, logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# ─── Domain definitions ────────────────────────────────────────────────────

DOMAINS = [
    "avian_flu",
    "rabies",
    "fmd",
    "nipah_hendra",
    "leptospirosis",
    "general",
]

# Seed training examples — (text, domain)
# These are zero-shot-style anchors for cosine baseline +
# initial training set for the MLP.
SEED_EXAMPLES = [
    # ── avian_flu ────────────────────────────────────────────────────────
    ("Several chickens died overnight with neurological twitching and cyanotic combs", "avian_flu"),
    ("Mass death in poultry, birds found with swollen heads and blue discoloration", "avian_flu"),
    ("Wild ducks behaving strangely, falling over, trembling near the pond", "avian_flu"),
    ("High mortality in the chicken farm, birds not eating, respiratory distress", "avian_flu"),
    ("Ten chickens dead this morning, runny nose, twisted necks", "avian_flu"),
    ("Geese in the area have been dying, greenish diarrhea, sudden deaths", "avian_flu"),
    ("Poultry workers are also getting fever after handling sick birds", "avian_flu"),
    ("All my chickens look sick, combs turning dark, lying on the ground", "avian_flu"),
    ("Ducks and turkeys dying in groups, some have bleeding from beaks", "avian_flu"),
    ("Birds in neighboring farm all died within two days, very sudden", "avian_flu"),

    # ── rabies ──────────────────────────────────────────────────────────
    ("A dog bit multiple people in the village, it was acting aggressively and foaming", "rabies"),
    ("Wild fox came into the village, attacking without provocation", "rabies"),
    ("Stray dog showing aggression, wandering aimlessly, biting other animals", "rabies"),
    ("Cow was bitten by a wild animal three weeks ago, now showing aggression and paralysis", "rabies"),
    ("Bat was found on the ground unable to fly, scratched one of the children", "rabies"),
    ("Dog died after showing excessive salivation, fear of water, aggression", "rabies"),
    ("Raccoon approaching humans in daylight, abnormal behavior, bit a farmer", "rabies"),
    ("Monkey attacked several people in forest clearing, unusual aggression", "rabies"),
    ("Wolf acting strangely, no fear of humans, entered village repeatedly", "rabies"),
    ("Several dogs showing hydrophobia signs, biting each other in the compound", "rabies"),

    # ── fmd ─────────────────────────────────────────────────────────────
    ("Cows have blisters on their mouths and feet, unable to walk or eat", "fmd"),
    ("Pigs with severe lameness, sores between hooves and around snout", "fmd"),
    ("Cattle showing drooling, blisters on tongue, foot sores, spreading fast", "fmd"),
    ("Outbreak of vesicular disease in the livestock, animals not eating", "fmd"),
    ("Sheep with mouth ulcers and limping, multiple animals affected in two days", "fmd"),
    ("Goats developing blisters on gums and tongue, refuse to stand up", "fmd"),
    ("Rapid spreading lameness through my cattle herd, sores visible on feet", "fmd"),
    ("Water buffalo with bleeding sores in mouth and feet after market visit", "fmd"),
    ("All pigs in the pen showing blistering lesions, salivating heavily", "fmd"),
    ("Cattle blisters appeared after we purchased new animals from another farm", "fmd"),

    # ── nipah_hendra ────────────────────────────────────────────────────
    ("Pigs are making unusual respiratory sounds, snorting, many dying in a week", "nipah_hendra"),
    ("Fruit bats roosting near our pig pen, now pigs are very sick", "nipah_hendra"),
    ("Horse showing fever, severe breathing difficulty, muscle tremors, died fast", "nipah_hendra"),
    ("A horse handler developed encephalitis after treating a sick horse", "nipah_hendra"),
    ("Multiple pigs suddenly ill with neurological symptoms near bat habitat", "nipah_hendra"),
    ("Bat colony near the mango orchard, several workers fell ill with fever and confusion", "nipah_hendra"),
    ("Pigs coughing violently, encephalitis-like symptoms, handlers also feeling ill", "nipah_hendra"),
    ("Horses dying in cluster with acute respiratory distress, stable near large bat tree", "nipah_hendra"),
    ("Village near fruit bat habitat reporting both pig deaths and human encephalitis", "nipah_hendra"),
    ("Hendra-like illness suspected in horse, vet has fever three days after examination", "nipah_hendra"),

    # ── leptospirosis ───────────────────────────────────────────────────
    ("Cattle and dogs sick after flooding, jaundice, not eating, some died", "leptospirosis"),
    ("Workers in the rice field developed fever and jaundice after contact with floodwater", "leptospirosis"),
    ("Rats in the farm area, cows showing bloody urine, eye inflammation", "leptospirosis"),
    ("After the flood several cows miscarried, some show yellow eyes and skin", "leptospirosis"),
    ("Dogs with hemorrhagic diarrhea and jaundice after rodent exposure", "leptospirosis"),
    ("Slaughterhouse workers with fever, muscle pain, and jaundice after wet season", "leptospirosis"),
    ("Heavy rain last week, now multiple farm animals showing renal failure symptoms", "leptospirosis"),
    ("Three cows died after wading through flooded fields, yellow discoloration", "leptospirosis"),
    ("Farm near river bank, rat infestation, animals showing kidney-related symptoms", "leptospirosis"),
    ("Cattle aborting, dark urine, lethargy — all after recent flood in the area", "leptospirosis"),

    # ── general ─────────────────────────────────────────────────────────
    ("Some animals on my farm seem unwell but I am not sure what is happening", "general"),
    ("I want to know what zoonotic diseases I should be concerned about in my area", "general"),
    ("Can you help me understand what symptoms I should watch for in my livestock", "general"),
    ("My animals have been acting differently but no specific symptoms I can describe", "general"),
    ("General advice on preventing disease spread between animals and humans", "general"),
    ("What should I do when I find a dead wild animal near my property", "general"),
    ("I am worried about my workers getting sick from the animals but nothing specific yet", "general"),
    ("Multiple animals look sick but symptoms don't fit any particular disease", "general"),
    ("Strange illness affecting both animals and people but unclear pattern", "general"),
    ("I heard there's an outbreak nearby, what should I watch for", "general"),
]


# ─── Router class ──────────────────────────────────────────────────────────

class ZoonoticRouter:
    """
    Dual-mode router:
      1. MLP mode  — if trained model found in model_dir/router.pkl
      2. Cosine mode — zero-shot fallback using SEED_EXAMPLES as anchors

    Training:
      python models/router.py --train [--extra data/router_training.jsonl]
    """

    def __init__(self, embedder, model_dir: Path = None):
        self.embedder  = embedder
        self.domains   = DOMAINS
        self.n_domains = len(DOMAINS)
        self.backend   = "cosine"   # default

        # Try to load trained MLP
        if model_dir:
            model_path = Path(model_dir) / "router.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self._mlp = pickle.load(f)
                    self.backend = "mlp"
                    log.info(f"Router: loaded MLP from {model_path}")
                except Exception as e:
                    log.warning(f"Router: failed to load MLP ({e}), using cosine fallback")
                    self._mlp = None
            else:
                self._mlp = None
        else:
            self._mlp = None

        # Pre-compute seed embeddings for cosine fallback
        log.info("Router: computing seed embeddings for cosine fallback...")
        seed_texts  = [ex[0] for ex in SEED_EXAMPLES]
        seed_labels = [ex[1] for ex in SEED_EXAMPLES]
        seed_embs   = embedder.encode(seed_texts, show_progress_bar=False, normalize_embeddings=True)

        # Build per-domain centroid embeddings
        self._centroids = {}
        for domain in DOMAINS:
            idxs = [i for i, l in enumerate(seed_labels) if l == domain]
            if idxs:
                self._centroids[domain] = seed_embs[idxs].mean(axis=0)
                self._centroids[domain] /= np.linalg.norm(self._centroids[domain])

        log.info(f"Router: cosine centroids ready for {len(self._centroids)} domains")

    # ── Public API ─────────────────────────────────────────────────────

    def classify(self, text: str, epi_fields: dict = None) -> tuple[str, float, dict]:
        """
        Classify text into a disease domain.

        Returns:
          (domain, confidence, all_scores)
        """
        # Enrich query with epi fields if available
        enriched = self._enrich(text, epi_fields or {})
        emb = self.embedder.encode([enriched], normalize_embeddings=True)[0]

        if self.backend == "mlp" and self._mlp is not None:
            return self._classify_mlp(emb)
        else:
            return self._classify_cosine(emb)

    def _enrich(self, text: str, epi_fields: dict) -> str:
        """Append extracted NER fields to improve embedding quality."""
        parts = [text]
        species  = epi_fields.get("species") or []
        symptoms = epi_fields.get("symptoms") or []
        if species:
            parts.append("Species: " + ", ".join(species))
        if symptoms:
            parts.append("Symptoms: " + ", ".join(symptoms[:5]))
        return ". ".join(parts)

    def _classify_cosine(self, emb: np.ndarray) -> tuple[str, float, dict]:
        scores = {}
        for domain, centroid in self._centroids.items():
            scores[domain] = float(np.dot(emb, centroid))

        # Softmax for probability-like scores
        vals   = np.array(list(scores.values()))
        vals   = np.exp((vals - vals.max()) * 10)   # temperature scaling
        vals  /= vals.sum()
        probs  = {d: float(v) for d, v in zip(scores.keys(), vals)}

        best   = max(probs, key=probs.get)
        return best, probs[best], probs

    def _classify_mlp(self, emb: np.ndarray) -> tuple[str, float, dict]:
        proba  = self._mlp.predict_proba([emb])[0]
        probs  = {d: float(p) for d, p in zip(self.domains, proba)}
        best   = max(probs, key=probs.get)
        return best, probs[best], probs


# ─── Training script ───────────────────────────────────────────────────────

def train(model_dir: Path, extra_data: Path = None):
    """Train an MLP classifier and save to model_dir/router.pkl."""
    from sentence_transformers import SentenceTransformer
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')

    print("Loading embedder...")
    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Collect examples
    examples = list(SEED_EXAMPLES)

    if extra_data and Path(extra_data).exists():
        print(f"Loading extra training data from {extra_data}...")
        with open(extra_data) as f:
            for line in f:
                item = json.loads(line.strip())
                examples.append((item["text"], item["domain"]))

    texts  = [ex[0] for ex in examples]
    labels = [ex[1] for ex in examples]

    print(f"Encoding {len(texts)} examples...")
    X = emb_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    print("Training MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    clf.fit(X, y)

    # Wrap with label encoder info
    clf.classes_named_ = list(le.classes_)

    # Cross-val score
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    print(f"Cross-val F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")

    # Per-domain counts
    from collections import Counter
    counts = Counter(labels)
    print("Training set distribution:")
    for domain in DOMAINS:
        print(f"  {domain:20s}: {counts.get(domain, 0):3d} examples")

    # Save
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "router.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to: {out_path}")

    # Also save label mapping
    meta = {"classes": list(le.classes_), "domains": DOMAINS, "n_examples": len(texts)}
    with open(model_dir / "router_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print("Metadata saved.")


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ZoonoticSense Router Training")
    parser.add_argument("--train",   action="store_true", help="Train the MLP router")
    parser.add_argument("--extra",   type=str, default=None, help="Path to extra JSONL training data")
    parser.add_argument("--out-dir", type=str, default="models", help="Output directory for model")
    parser.add_argument("--test",    type=str, default=None, help="Quick test: classify this text")
    args = parser.parse_args()

    if args.train:
        train(model_dir=Path(args.out_dir), extra_data=args.extra)

    elif args.test:
        from sentence_transformers import SentenceTransformer
        print("Loading embedder for test...")
        emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        r   = ZoonoticRouter(embedder=emb, model_dir=Path(args.out_dir))
        domain, conf, scores = r.classify(args.test)
        print(f"\nText: {args.test}")
        print(f"Domain: {domain} (confidence: {conf:.3f})")
        print("All scores:")
        for d, s in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(s * 40)
            print(f"  {d:20s}: {s:.3f} {bar}")
    else:
        parser.print_help()

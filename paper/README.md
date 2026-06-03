# ICONIP 2026 paper — ZoonoMoE router study

**Title:** How Small a Router Do You Need? A Trade-off Study of Lightweight
Neural Domain Routing for On-Device Zoonotic Triage

## Files
- `paper.tex` — LNCS manuscript (compile with **pdfLaTeX**)
- `llncs.cls`, `splncs04.bst` — Springer LNCS class + bib style (bundled)
- `figs/pareto.png` — accuracy-vs-latency Pareto (Fig. 1)
- `figs/confusion_matrix.png` — out-of-fold confusion, corrected router (Fig. 2)
- `figs/cascade_pareto.png` — cascade accuracy vs avg latency (Fig. 3)
- `figs/reliability.png` — Stage-1 calibration before/after temp scaling (Fig. 4)
- `figs/per_class_f1.png` — per-domain F1 (optional)

## Build (Overleaf — recommended, no local LaTeX needed)
1. New Project → Upload Project → zip this `paper/` folder (incl. `figs/`).
2. Menu → Compiler: **pdfLaTeX**.
3. Recompile. (No bibtex run needed — references are inline `thebibliography`.)

## Build (local, if you install TeX Live)
```bash
cd paper
pdflatex paper.tex && pdflatex paper.tex
```

## Reproducing the numbers
```bash
cd ../backend
python3.12 scripts/benchmark_router.py     # main table + figures (Tab.1, Fig.1-2)
python3.12 scripts/ablation_router.py      # ablation tables (A: head, B: backbone)
python3.12 scripts/cascade_router.py       # cascade study (Tab.3, Fig.3-4, calibration)
# outputs land in backend/scripts/eval_out/
```
All results: 5-fold stratified CV, seed 42, all-MiniLM-L6-v2 on CPU, n=160.

To rebuild the deployed cascade-enabled router:
```bash
python3.12 models/router.py --train --extra data/router_training.jsonl   # Stage-2 MLP
python3.12 models/router.py --train-cascade --extra data/router_training.jsonl --tau 0.54
python3.12 models/router.py --test "thirty chickens died with purple combs"  # shows stage used
```

## Before submitting — checklist
- [ ] Fill author names / affiliations / emails (currently placeholders).
- [ ] **Length: ICONIP requires 12–15 pages.** This draft is ~6–8 pp; expand
      Related Work, add a system figure, and enlarge Discussion/Limitations,
      or it will be desk-rejected for being under length.
- [ ] Anonymize if the track is double-blind (remove author block, repo links).
- [ ] Choose a track on EasyChair (suggest **Track 4 – Applications** or
      **Track 3 – Healthcare / Human-centred computing**).
- [ ] Replace inline `thebibliography` with proper citations if you prefer bibtex.

# Release notes — v1.0.0

Release accompanying the IEEE Transactions on Smart Grid submission.

## What this tag covers

Public companion artefacts for **Convex Stacking Ensemble (CSE)** / FAHE forecast combination with KKT sparsity analysis on five hourly energy series (UCI, ETTh1, ETTh2, OPSD_DE_household, OPSD_DE_total).

### Benchmark results (`results/`)

- One-step and recursive leaderboards (`leaderboards.csv`), including per-model training times
- CSE simplex weights and KKT diagnostics (`cse_weights_by_dataset.csv`, `cse_diagnostics/`)
- Rolling-origin multi-horizon tables under `multistep_rolling/` only (`multistep/` is diagnostic, not for paper tables)
- Significance suite: Diebold–Mariano (HLN), Holm, Cohen’s *d*, Cliff’s δ, Friedman, CD diagrams (`significance/`)
- Cached one-step / recursive predictions (`predictions/`)

### Manuscript figures (`figures/`)

Numbered PNG/PDF assets `fig1_*` … `fig5_*` (architecture TikZ, KKT boundary, CD diagrams, error drift, DM heatmaps), regenerated from `results/` via `make figures`.

### Code and reproducibility

- Pinned CPU stack: Python 3.12.10, `requirements.txt` (optional DL/UI in `requirements-optional.txt`)
- Data acquisition notes: `DATA.md` / `make data` (raw series not shipped)
- Smoke path and claim checks: `make smoke`, `make verify` (`scripts/verify_claims.py`)
- Method, feature, rank/DM, and figure/table maps under `docs/`
- Paper handoff notes: `docs/PAPER_README.md`

### Explicit non-claims

- No Transformer / Autoformer / Informer / PatchTST result rows in the shipped tables
- OPSD national load uses 19 standalone models in the shared-rank pool (CSE still stacks its five internal members)
- Raw datasets are not included in the repository

## Install check

```bash
pip install -r requirements.txt
make data    # or seed files per DATA.md
make smoke
make verify
```

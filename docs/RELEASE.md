# Release proposal (do not push yet)

## Suggested GitHub repository

- **Name:** `convex-forecast-combination-kkt`
- **Description:** Convex forecast stacking with KKT sparsity boundaries — multi-dataset hourly energy forecasting benchmarks (CSE / FAHE). Companion code for an IEEE Transactions on Smart Grid submission.

## Tag

`v1.0.0`

## Release notes (draft)

### Artefacts covered

- One-step and recursive leaderboards for five energy series (`results/leaderboards.csv`)
- CSE / FAHE simplex weights and KKT diagnostics (`results/cse_weights_by_dataset.csv`, `cse_diagnostics/`)
- Rolling-origin horizon tables only under `results/multistep_rolling/`
- DM / HLN / Holm, Cohen’s d, Cliff’s δ, Friedman, CD diagrams (`results/significance/`, `figures/`)
- Regenerated manuscript figures `figures/fig1_*` … `fig5_*`
- Claim verifier: `scripts/verify_claims.py` / `make verify`
- CPU-pinned `requirements.txt` (Python 3.12.10); optional DL/UI in `requirements-optional.txt`

### Explicit non-claims

- No Transformer / Autoformer / Informer / PatchTST result rows
- OPSD national load uses 19 standalone models (CSE still has internal SARIMA/HW)
- Raw datasets are not shipped; see `DATA.md`

### Install check

```bash
pip install -r requirements.txt
make data   # or seed from DATA.md
make smoke  # ~20 s on CPU in this environment
make verify
```

## Git hygiene report

- Repository initialised in Phase 1; **no commits yet** (staged only). History secret scan: empty history — nothing to rewrite.
- Working tree / stage: no API keys found; author email only in `README.md` / `CITATION.cff` (intentional).
- Raw `data/*` gitignored except `data/README.md`. Logs and `paper_pack/` ignored.
- Largest tracked artefacts are prediction parquets and diagnostics (&lt;300 KB each). Figures ~6 MB total. **Git LFS not required** at current size (~20–30 MB results+figures without raw data).
- Host absolute paths appear only in historical `AUDIT.md` narrative (acceptable as audit log). No `*.log` shipped.

## Next step for the maintainer

1. Review `git status` / staged file list.
2. Create the first commit when ready.
3. Create GitHub repo `convex-forecast-combination-kkt`, push, tag `v1.0.0`.
4. Replace DOI placeholders in `CITATION.cff` / README when available.

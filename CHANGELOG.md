# Changelog

## Unreleased — Phase 1 restructure (behaviour-preserving)

### Layout

- Renamed `results_paper/` → `results/` (single source of truth; no copy).
- Moved Streamlit UI from `app/` → `extras/streamlit_app.py`.
- Added `src/cse/` as the paper-facing package namespace (re-exports existing
  modules; **no forecast logic rewrite**).
- Added `docs/features.md` (57 columns as emitted by code).
- Added root `figures/` copies of PDFs from `results/figures/` for convenience
  (Phase 3 will regenerate via scripts).
- Created empty `paper/` placeholder for TikZ / manuscript assets.

### Dependencies

- `requirements.txt` pinned to exact versions from the Python **3.12.10** `.venv`
  that produced the artefacts (CPU stack).
- Optional GPU / foundation / Streamlit packages moved to
  `requirements-optional.txt`.

### Documentation

- README: removed claim that Transformer models “participate on GPU”; replaced
  with: optional code paths, no artefacts, no paper results.
- Path strings updated from `results_paper/` to `results/` across scripts and docs.

### Deletions

- `results/paper_run.log`, `results/multistep_rolling/run.log` (contained host
  absolute paths; must not ship).
- `results/checkpoints/` (ephemeral resume markers).
- `results/leaderboards_partial.csv` (duplicate of `leaderboards.csv`).
- `app/` package directory (replaced by `extras/`).
- `scripts/run_full_experiment.py` (superseded by `run_paper_experiments.py`).

### Behaviour affecting

- **None intended.** FeatureConfig, model code, and result CSV/parquet bytes were
  not regenerated. Verify with MD5 of key artefacts after the directory rename.

### Naming

- `FAHE` (code / CSV label `FAHE (Ours)`) ≡ `CSE` (manuscript). Prefer
  `from src.cse import CSEForecaster`; `FAHEForecaster` remains a deprecated alias.


## Unreleased — Phases 2–7 (public release packaging)

### Added
- `Makefile`, `scripts/download_data.py`, `scripts/verify_claims.py`,
  `scripts/build_figures.py`, `scripts/build_tables.py`, `src/seed.py`
- `LICENSE` (Apache-2.0), `NOTICE`, `CITATION.cff`, `DATA.md`, `CONTRIBUTING.md`,
  `environment.yml`, `docs/method.md`, `docs/figure_map.md`, `docs/table_map.md`,
  `results/README.md`, `paper/fig1_architecture.tex`
- Regenerated `figures/fig1_*` … `fig5_*` (PNG+PDF where applicable)
- Apache short headers on `src/` and `scripts/` Python files

### Behaviour affecting
- None intended for published numbers. Smoke test window sized for lag-168 features.

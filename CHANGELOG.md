# Changelog

## Unreleased: exact simplex solver and uncentred KKT diagnostics

### Changed (affects reported numbers)

- `src/models/novel.py`: `_solve_convex_weights` now solves
  `min ||y - Pw||^2  s.t. w >= 0, sum(w) = 1` exactly (new public helper
  `solve_simplex_qp`). The previous code ran NNLS with a ridge row and then
  divided by the L1 norm, which is a different optimisation problem from the
  one analysed in the paper whenever the raw NNLS weights do not sum to one.
- `src/evaluation/kkt_analysis.py`: `Sigma`, `sigma` and `rho` are now
  **uncentred** (`E'E/n`), matching the objective `w' Sigma w`. The old centred
  statistics remain available through `centred=True`. `kkt_lambda` is now
  `w' Sigma w`, and `kkt_conditions` uses a tolerance relative to lambda.

### Added

- `scripts/recheck_simplex_kkt.py`: re-solves the weights from the stored
  validation errors, with no model refit, and prints the Proposition 1 table
  and the full KKT check. Output: `results/tables/kkt_simplex_recheck.csv`.
- Four tests in `tests/test_kkt.py` covering the solver and the diagnostics.

### Added (November 2026)

- `FAHEForecaster.refit_only_active` (default True): members with zero simplex
  weight are not refit on the full training set. The blend is identical;
  the second fit of the excluded members is saved (`refit_times_`,
  `skipped_refits_`).
- `scripts/rerun_cse.py`: refits only the CSE and replaces its rows in every
  artefact (leaderboards, predictions, diagnostics, rolling origin, refit
  times). `python scripts/run_significance_analysis.py --in results/` then
  rebuilds the DM and Friedman files.
- `src/models/stacking.py` and `scripts/membership_sensitivity.py`: simplex
  stacking over any pool of registry models; six pools per dataset.
- `scripts/multi_fold_cse.py`: expanding-window folds for the CSE.
- `scripts/build_paper.py`: writes every manuscript table and inline number
  to `paper/generated/`; `paper/main.tex` contains no hand-typed numbers.
- `scripts/recheck_simplex_kkt.py --update-artefacts`: rewrites the
  diagnostics pickles and `cse_weights_by_dataset.csv` with the exact
  weights and uncentred moments (no refit needed).
- KKT boundary figure: log ratio axis, uncentred moments, colour-safe
  markers, a separate marker for admissible members zeroed by the full
  system.
- `scripts/verify_claims.py` updated to the exact solver (tree mass 0.965 on
  OPSD-H, SARIMA ratio about 508 on OPSD-N, qualitative Friedman check).

### Fixed: OPSD household target (v1.2, 7 November 2026)

- `src/data/extra_loaders.py`: the OPSD-H target averaged the households'
  cumulative meters and then differenced the average. When a household
  started or stopped reporting, the average jumped and produced a fake spike
  (137 kWh in one hour on 1 Feb 2017); a 201 kWh jump in one meter on
  8 Jul 2017 was also kept. Each meter is now differenced first, negative
  steps and steps above 43.5 kWh/h are set to missing, and the reporting
  households are averaged (`opsd_household_series`). Tests in
  `tests/test_opsd_household.py`.
- Every OPSD-H artefact must be regenerated: `scripts/rerun_dataset.py`
  (new) reruns all standalone models on one dataset and replaces only its
  rows; then the lite-DL retuning, rolling origin, CSE, significance, pools
  and folds for OPSD-H.
- `scripts/verify_claims.py` now checks invariants (no boundary violation,
  tree-dominated weights, Friedman pattern) instead of exact counts.
- `scripts/build_paper.py`: sentences that name winners, ties and divergent
  models are generated from the results; identical CSE and member forecasts
  are reported as "identical", not as a CSE win; refit savings table.

### Fixed (7 November 2026)

- SARIMA and ARIMA fits (in the CSE and as standalone models) now use
  statsmodels `low_memory=True`. Parameters and forecasts are bit-for-bit
  identical (checked on ETTh1 and ETTh2), but peak memory for one seasonal
  fit on 12k hourly points drops from about 3.9 GB to 0.2 GB. Without it the
  CSE refit on the two OPSD series was killed by the out-of-memory killer on
  a laptop.
- `scripts/rerun_cse.py` writes `results/tables/cse_refit_times.csv` after
  every dataset, so a crash does not lose the datasets already finished.

### Corrected

- The pairwise boundary of Proposition 1 is exact for two members only. For
  K > 2 it is a screen that can fail in both directions (a complementary
  member can admit a member above the boundary); the manuscript now says so
  and the fold experiment records one such case.

### Still to regenerate

CSE rows for ETTh1 and ETTh2 (both protocols, rolling origin, DM, Friedman)
and the diagnostics of all five datasets have been regenerated with the
exact solver. The CSE test-set rows for UCI, OPSD_DE_household and
OPSD_DE_total still come from the old solver until
`python scripts/rerun_cse.py --datasets UCI OPSD_DE_household OPSD_DE_total`
and `python scripts/run_significance_analysis.py --in results/` have been
run (the raw data for those series must be downloaded first). The
manuscript prints those entries in red until then.

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

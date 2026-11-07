# Changelog

## v1.2.0 (7 November 2026)

Exact simplex solver and uncentred KKT diagnostics (v1.1), corrected OPSD
household target (v1.2), and a clean-up of the repository.

### Removed

- `scripts/build_tables.py` and `results/tables/table_*`: the paper reads
  `paper/generated/`, written by `scripts/build_paper.py`.
- `scripts/run_multistep_benchmark.py` and `results/multistep/`: the paper uses
  `results/multistep_rolling/` only.
- `scripts/rerun_cse_recursive.py` (replaced by `scripts/rerun_cse.py`) and
  `scripts/create_paper_pack.py`.
- `figures/fig4_error_drift_*` and `results/figures/error_drift__*`, which
  predated the solver and OPSD household fixes. The paper's drift figure is
  `figures/fig5_error_drift_uci.pdf`.
- `RELEASE_NOTES_v1.0.0.md` and `docs/RELEASE.md`.

### Dependencies

- `requirements.txt` now holds only the CPU packages needed to reproduce
  `results/` and run the tests. The other pins moved to `requirements-gpu.txt`.

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
- `scripts/verify_claims.py` updated to the exact solver (SARIMA ratio about
  508 on OPSD-N, qualitative Friedman check).

### Results regenerated (v1.2, 7 November 2026)

All five datasets now carry the exact simplex solver, and OPSD-H the
corrected target. Headline changes against the v1.0 artefacts: Ridge is the
best one-step model on OPSD-H (0.147 kW) as well as ETTh1; the CSE wins
OPSD-N, ties UCI and ETTh2 and loses OPSD-H and ETTh1 one step ahead; tree
mass of the CSE weights rounds to 1.000 on every series; Friedman one-step
chi2 = 69.47 (p = 5.6e-8), recursive 19.60 (p = 0.356); Kendall tau between
the protocols is negative on OPSD-H (-0.42); refit-only-active saves 36 to
53% of CSE training time on four series. `paper/main.tex` reads all of this
from `paper/generated/`.

### Fixed: OPSD household target (v1.2, 7 November 2026)

- `src/data/extra_loaders.py`: the OPSD-H target averaged the households'
  cumulative meters and then differenced the average. When a household
  started or stopped reporting, the average jumped and produced a fake spike
  (137 kWh in one hour on 1 Feb 2017); an 804 kWh jump in one meter on
  8 Jul 2017 (201 kWh in the average) was also kept. Each meter is now differenced first, negative
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

## v1.0.0: restructure (behaviour-preserving)

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


## v1.0.0: public release packaging

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

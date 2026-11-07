# Results artefacts

All paper numbers trace to files under this directory, through
`scripts/build_paper.py`. Horizon results come from `multistep_rolling/`.

Model label **`FAHE (Ours)`** in CSVs is the same method as **CSE** in the manuscript.

## Top-level

| File | Description |
|------|-------------|
| `leaderboards.csv` | Long-format metrics for every dataset, protocol and model |
| `cse_weights_by_dataset.csv` | CSE member weights, scale, correlation with the best member, KKT residual |

### `leaderboards.csv` columns

| Column | Meaning |
|--------|---------|
| `Dataset` | One of UCI, ETTh1, ETTh2, OPSD_DE_household, OPSD_DE_total |
| `Protocol` | `one_step` (teacher forced) or `recursive` |
| `Model` | Model display name (`FAHE (Ours)` = CSE) |
| `Family` | Model family tag |
| `TrainTime(s)`, `PredictTime(s)` | Wall times |
| `MAE`, `RMSE`, `MAPE`, `sMAPE`, `R2`, `MASE` | Point metrics (MASE seasonal period 24) |

OPSD_DE_total has **19** models (standalone heavy classical models skipped for memory).
The CSE still includes its internal SARIMA and Holt-Winters members there.

### `cse_weights_by_dataset.csv` columns

| Column | Meaning |
|--------|---------|
| `dataset` | Dataset id |
| `member` | `sarima`, `hw`, `lgbm`, `catboost`, `extratrees` |
| `weight` | Exact least-squares weight on the probability simplex, on the validation slice |
| `sigma` | Uncentred validation error scale (validation RMSE) |
| `rho_vs_best` | Uncentred error correlation with the best (lowest `sigma`) member |
| `is_best` | Whether this member has the lowest `sigma` |
| `kkt_residual` | `(Σw)_i − λ` |

## Subdirectories

| Path | Contents |
|------|----------|
| `predictions/` | Parquet `y_true` / `y_pred` series per dataset, protocol and model |
| `cse_diagnostics/` | Pickled `CSEDiagnostics` for the KKT analysis |
| `multistep_rolling/` | Rolling-origin horizons `h ∈ {1,3,6,12,24}`, 24 origins |
| `significance/` | DM matrices (with Holm `p_adj`, Cohen's d, Cliff's δ), Friedman summary |
| `figures/` | CD diagrams, DM heat maps and the KKT boundary, as written by the analysis scripts |
| `tuning/lite_dl/` | Optuna histories and best parameters for NLinear and DLinear |

### `tables/`

| File | Written by |
|------|------------|
| `cse_refit_times.csv` | `scripts/rerun_cse.py` |
| `membership_sensitivity.csv`, `membership_sensitivity_weights.csv` | `scripts/membership_sensitivity.py` |
| `cse_folds.csv`, `cse_folds_weights.csv` | `scripts/multi_fold_cse.py` |
| `kkt_simplex_recheck.csv` | `scripts/recheck_simplex_kkt.py --out` |
| `OPSD_DE_household_removed_readings.csv` | `scripts/rerun_dataset.py` |

Manuscript tables are not stored here. They are written to `paper/generated/` by
`scripts/build_paper.py`.

## Regenerating

The full order is in the Reproduce section of the top-level `README.md`. From the
shipped `results/`, without retraining:

```bash
make stats      # DM, Friedman, KKT PDF under results/figures
make figures    # figures/fig2_*, fig3_* (PNG and PDF)
python scripts/build_paper_figures.py   # figures/fig4_*, fig5_*, fig6_*
python scripts/build_paper.py           # paper/generated/
make verify     # check headline claims
```

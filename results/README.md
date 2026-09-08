# Results artefacts

All paper numbers must trace to files under this directory. Horizon tables for
the manuscript come from **`multistep_rolling/` only**. The `multistep/` folder
is a prefix diagnostic and must not be cited in the paper or README.

Model label **`FAHE (Ours)`** in CSVs is the same method as **CSE** in the manuscript.

## Top-level

| File | Description |
|------|-------------|
| `leaderboards.csv` | Long-format metrics for every dataset × protocol × model |
| `cse_weights_by_dataset.csv` | CSE member weights, σ, ρ vs best, KKT residual |

### `leaderboards.csv` columns

| Column | Meaning |
|--------|---------|
| `Dataset` | One of UCI, ETTh1, ETTh2, OPSD_DE_household, OPSD_DE_total |
| `Protocol` | `one_step` (teacher forced) or `recursive` |
| `Model` | Model display name (`FAHE (Ours)` = CSE) |
| `Family` | Model family tag |
| `TrainTime(s)`, `PredictTime(s)` | Wall times |
| `MAE`, `RMSE`, `MAPE`, `sMAPE`, `R2`, `MASE` | Point metrics (MASE seasonal period 24) |

OPSD_DE_total has **19** models (standalone heavy classical skipped for RAM).
CSE still includes internal SARIMA / Holt–Winters members.

### `cse_weights_by_dataset.csv` columns

| Column | Meaning |
|--------|---------|
| `dataset` | Dataset id |
| `member` | `sarima`, `hw`, `lgbm`, `catboost`, `extratrees` |
| `weight` | Simplex weight from NNLS on validation |
| `sigma` | Validation error standard deviation |
| `rho_vs_best` | Error correlation with the best (lowest-σ) member |
| `is_best` | Whether this member has lowest σ |
| `kkt_residual` | `(Σw)_i − λ` |

## Subdirectories

| Path | Contents |
|------|----------|
| `predictions/` | Parquet `y_true` / `y_pred` series per dataset × protocol × model |
| `cse_diagnostics/` | Pickled `CSEDiagnostics` for the KKT figure |
| `multistep_rolling/` | **Paper** rolling-origin horizons `h ∈ {1,3,6,12,24}`, 24 origins |
| `multistep/` | Diagnostic prefix horizons — **do not use for paper tables** |
| `significance/` | DM matrices (incl. Holm `p_adj`, Cohen’s d, Cliff’s δ), Friedman summary |
| `figures/` | Intermediate PDFs from analysis scripts |
| `tuning/lite_dl/` | Optuna best params for NLinear / DLinear |
| `tables/` | CSV + LaTeX exports from `scripts/build_tables.py` |

## Regenerating derived files

```bash
make stats      # DM, Friedman, KKT PDF under results/figures
make figures    # numbered figures/figN_* PNG+PDF
make tables     # results/tables/*
make verify     # assert headline claims
```

Full retrain: `make benchmark` and `make rolling` (many hours, CPU).

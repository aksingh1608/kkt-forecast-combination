# Guide to the paper's artefacts

This page lists what the manuscript uses and what to keep in mind when reading
it. Build instructions are in [`paper/README.md`](../paper/README.md).

## Question

Does a convex stacking ensemble over classical and tree forecasters collapse
onto tree learners, and do the KKT conditions of the simplex problem explain
which members receive zero weight?

## Datasets

`UCI`, `ETTh1`, `ETTh2`, `OPSD_DE_household` and `OPSD_DE_total`, all through
one causal feature pipeline and one chronological 80/20 split. Sources and
cleaning: [`DATA.md`](../DATA.md).

## Models

- Persistence and classical: Naive, NaiveDrift, SeasonalNaive, ARIMA, SARIMA,
  Holt-Winters, Theta.
- Linear, kernel and instance: Ridge, Lasso, ElasticNet, SVR, KNN.
- Tree and boosting: RandomForest, ExtraTrees, GradientBoosting,
  HistGradientBoosting, XGBoost, LightGBM, CatBoost.
- Quantile point baseline: QuantileLightGBM.
- Lightweight linear baselines: NLinear and DLinear, tuned with Optuna.
- CSE: an exact least-squares fit on the probability simplex over SARIMA,
  Holt-Winters, LightGBM, CatBoost and ExtraTrees. `FAHE (Ours)` in code and CSV
  files is the CSE.

## Protocols

- **One-step (teacher forced):** supervised learners receive true lag and
  rolling features from the test set.
- **Recursive:** lag and rolling features are rebuilt from the model's own
  predictions. The CSE rolls out as one unit.
- **Rolling origin:** horizons 1, 3, 6, 12 and 24 from 24 forecast origins per
  dataset, on the 19 models shared by all series. Output:
  `results/multistep_rolling/`.

## Files

| Topic | File |
|-------|------|
| Metrics by dataset, protocol and model | `results/leaderboards.csv` |
| CSE weights and KKT quantities | `results/cse_weights_by_dataset.csv`, `results/cse_diagnostics/` |
| Horizon results | `results/multistep_rolling/long_all__rolling.csv` |
| Diebold-Mariano, Holm, Cohen's d, Cliff's delta | `results/significance/dm_matrix__*.csv` |
| Friedman test | `results/significance/friedman_summary.csv` |
| Member pools and folds | `results/tables/membership_sensitivity*.csv`, `results/tables/cse_folds*.csv` |
| Full KKT check | `results/tables/kkt_simplex_recheck.csv` |
| Lite-DL tuning | `results/tuning/lite_dl/summary.csv` |

Which file feeds which table or figure: [`table_map.md`](table_map.md) and
[`figure_map.md`](figure_map.md).

## Limitations

1. `OPSD_DE_total` has 19 models instead of 23 because the standalone heavy
   classical models exceeded the memory budget. The CSE still includes its
   internal SARIMA and Holt-Winters members there.
2. Transformer models exist as optional code paths but produced no artefacts,
   and no result in the paper comes from them.
3. NLinear and DLinear are simple linear baselines and remain weak on the ETT
   series.
4. Smoke-test output is a path check and is not a result.

# Paper Handoff README

This bundle contains the final artefacts for drafting an IEEE Transactions-style paper on controlled electric-load forecasting and convex forecast combination.

## Working Title

**When Forecast Combination Collapses onto Tree Learners: A Controlled Multi-Dataset Benchmark for Hourly Energy Forecasting**

## Central Thesis

The project tests the hypothesis that a convex stacking ensemble over classical and tree-based forecasters often collapses onto tree learners because the simplex-constrained optimum assigns zero weight to members whose validation error variance/correlation profile violates a KKT exclusion condition.

In short:

- Tree learners dominate the one-step teacher-forced setting because they exploit true lag and rolling features.
- Under fair recursive multi-step rollout, the tree advantage shrinks or reverses on some datasets because trees must consume their own generated lags.
- The learned convex ensemble weights are not arbitrary: the zero-weight pattern is explainable through the KKT boundary figure.

## What Was Done

### 1. Multi-Dataset Benchmark

The benchmark no longer relies only on the 12-month UCI household window.

Datasets used:

- `UCI`: individual household electric power consumption, France.
- `ETTh1`: electricity transformer benchmark, China.
- `ETTh2`: electricity transformer benchmark, China.
- `OPSD_DE_household`: German residential household load.
- `OPSD_DE_total`: German national load.

All datasets are evaluated through one shared feature pipeline and chronological train/test split.

### 2. Model Zoo

The benchmark includes:

- Persistence/classical baselines: Naive, NaiveDrift, SeasonalNaive, ARIMA, SARIMA, Holt-Winters, Theta.
- Linear/kernel/instance learners: Ridge, Lasso, ElasticNet, SVR, KNN.
- Tree/boosting learners: RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
- Probabilistic point baseline: QuantileLightGBM.
- Lightweight linear DL-style baselines: NLinear and DLinear.
- Proposed ensemble: CSE / FAHE, a convex NNLS stacking ensemble over SARIMA, Holt-Winters, LightGBM, CatBoost, and ExtraTrees.

Important note: `FAHE (Ours)` in the code corresponds to **CSE (Convex Stacking Ensemble)** in the manuscript.

### 3. Two Evaluation Protocols

Both protocols were run:

- **One-step / teacher-forced:** supervised learners receive true lag/rolling features from the test set.
- **Recursive multi-step:** lag/rolling features are recomputed from the model's own previous predictions.

The CSE recursive issue was fixed: CSE no longer reuses the one-step tree features under recursive evaluation. Its tree branches now consume lag/rolling features rebuilt from the ensemble's own past predictions.

### 4. Rolling-Origin Multi-Step Evaluation

The original prefix horizon table used the first `h` steps of one recursive rollout, which made `h=1` a single noisy point. This was corrected.

Final paper horizon tables use rolling-origin evaluation:

- horizons: `h = 1, 3, 6, 12, 24`
- origins: 24 forecast origins per dataset
- output folder: `results/multistep_rolling/`

Use these rolling-origin tables for the paper, not the prefix diagnostic tables in `results/multistep/`.

### 5. Statistical Significance

The project generated:

- Diebold-Mariano matrices with HLN small-sample correction.
- Holm-adjusted p-values.
- Cohen's d effect sizes.
- Cliff's delta distribution-free effect sizes.
- Friedman summaries.
- Nemenyi critical-difference diagrams.
- DM p-value heatmaps.

Use `results/significance/` and the PDF figures in `results/figures/`.

### 6. KKT Boundary Analysis

The core theory contribution is visualized in:

`results/figures/kkt_boundary.pdf`

This figure plots each CSE base learner in the error variance ratio / error correlation plane and overlays the theoretical exclusion boundary. It supports the claim that zero-weight members are excluded by the convex optimum's KKT geometry.

### 7. Error-Over-Time Figures

The reviewer-requested drift plots were generated:

`results/figures/error_drift__*.pdf`

Each figure shows:

- daily average MAE over time
- cumulative absolute error over time

These show when classical, tree, and CSE models drift during the test period.

### 8. DLinear / NLinear Retuning

DLinear and NLinear were retuned after DLinear originally diverged badly on OPSD household. The final version uses robust clipping and Optuna search over lookback, ridge, kernel size, RevIN, and clipping quantile.

Final tuning summary:

`results/tuning/lite_dl/summary.csv`

Outcome:

- DLinear catastrophic divergence on OPSD household was fixed.
- NLinear and DLinear remain weak on ETT datasets, which should be reported as a result rather than hidden.
- DLinear/NLinear are now credible enough to include as lightweight linear DL-style baselines, but they should not be oversold.

## Key Files to Use

### Main Tables

- `results/leaderboards.csv`
  - Main dataset/protocol/model metrics.
  - Use for one-step and recursive full-window leaderboards.

- `results/cse_weights_by_dataset.csv`
  - CSE weight collapse table.
  - Use for showing tree learner dominance in the convex ensemble.

- `results/multistep_rolling/long_all__rolling.csv`
  - Paper-grade rolling-origin horizon results.
  - Use for h = 1, 3, 6, 12, 24 tables.

### Significance

- `results/significance/dm_matrix__*.csv`
  - DM tests, adjusted p-values, Cohen's d, Cliff's delta.

- `results/significance/friedman_summary.csv`
  - Friedman test summaries.

### Figures

- `results/figures/kkt_boundary.pdf`
  - Main theory figure.

- `results/figures/cd_diagram__one_step__RMSE.pdf`
  - Critical-difference diagram for one-step.

- `results/figures/cd_diagram__recursive__RMSE.pdf`
  - Critical-difference diagram for recursive.

- `results/figures/dm_heatmap__*.pdf`
  - DM adjusted p-value heatmaps.

- `results/figures/error_drift__*.pdf`
  - Daily/cumulative error drift plots.

## Important Caveats to Mention in the Paper

1. `OPSD_DE_total` has 19 models instead of 23 because standalone heavy classical models were skipped for RAM safety. CSE still includes its internal SARIMA/Holt-Winters branches.

2. Transformer baselines are implemented as optional GPU models, but the current `results/` artefacts do not contain Autoformer/FEDformer/Informer/iTransformer rows. Do not claim Transformer results unless the `--dl` experiments are run separately.

3. DLinear/NLinear are simple linear DL-style baselines, not full neural SOTA. They are included to address DLinear/NLinear-style lightweight baselines, but their weak ETT results should be presented honestly.

4. Use `multistep_rolling/` for the paper's horizon tables. The older `multistep/` folder is a prefix diagnostic and should not be used for headline horizon claims.

5. CSE recursive evaluation was fixed. The current `leaderboards.csv` and predictions reflect true recursive CSE rollout, not the older teacher-forced CSE path.

## Suggested Paper Structure

1. Introduction
   - Forecast combination is widely used, but convex ensembles may collapse onto a small subset of members.
   - Main claim: collapse is not accidental; it follows from error geometry under simplex constraints.

2. Related Work
   - Load forecasting.
   - Classical vs ML forecasting.
   - Forecast combination and convex stacking.
   - Multi-step recursive evaluation.

3. Methods
   - Datasets.
   - Feature engineering.
   - Model zoo.
   - CSE formulation.
   - KKT exclusion condition.
   - Evaluation protocols.

4. Experimental Design
   - One-step vs recursive protocols.
   - Rolling-origin horizon evaluation.
   - Metrics.
   - Significance tests.

5. Results
   - Main leaderboards.
   - CSE weights and collapse.
   - Recursive protocol effect.
   - Rolling-origin horizon tables.
   - Statistical significance.

6. Theory/Diagnostic Analysis
   - KKT boundary figure.
   - Error variance/correlation interpretation.

7. Error Dynamics
   - Daily MAE and cumulative error drift figures.

8. Limitations
   - Transformer results not included unless GPU run is added.
   - OPSD national heavy-classical memory guard.
   - DLinear/NLinear weakness on ETT.

9. Conclusion
   - Multi-dataset evidence supports the claim that convex forecast combination can collapse onto tree learners, and that this collapse is mathematically interpretable.

## What Not to Do

- Do not cite smoke-test results.
- Do not use `multistep/` prefix results as the headline horizon table.
- Do not claim Transformer baseline results unless separate GPU experiments are run.
- Do not invent numbers; all paper values should come from the CSV files in this pack.


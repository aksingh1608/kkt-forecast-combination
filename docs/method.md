# Convex Stacking Ensemble (CSE) and the KKT boundary

This note is for a reader who has not opened the paper. Code name **FAHE** is
the same method as **CSE**.

## Setup

We forecast an hourly energy series \(y_t\) with a fixed causal feature matrix
\(X_t\) (57 columns; see `docs/features.md`). Train and test are a single
chronological 80/20 split. A validation tail is carved from the training
partition for stacking weights. No future information enters training.

## Members

CSE uses \(K=5\) base forecasters:

1. SARIMA  
2. Holt–Winters  
3. LightGBM  
4. CatBoost  
5. Extra-Trees  

Each member produces a validation forecast \(\hat y_i^{\mathrm{val}}\). Let
\(P\in\mathbb{R}^{n_{\mathrm{val}}\times K}\) stack those forecasts and let
\(y^{\mathrm{val}}\) be the held-out targets.

## Convex stacking

Weights solve a non-negative least-squares problem on the validation slice,
then are normalised onto the probability simplex:

\[
\min_{w \ge 0}\ \|Pw - y^{\mathrm{val}}\|_2^2
\quad\text{then}\quad
w \leftarrow w / \mathbf{1}^\top w,
\qquad
w\in\Delta^{K-1}.
\]

Members are then **refitted on the full training partition** before test
forecasts. Under the recursive protocol the ensemble rolls out as a unit:
tree members consume lag/rolling features rebuilt from the ensemble’s own
past predictions (not teacher-forced true lags).

Implementation: `src/models/novel.py` (`FAHEForecaster` / `CSEForecaster`).

## Why weights collapse onto trees

On these series the validation errors of classical members are typically
both larger in scale and highly correlated with the best tree member. The
simplex optimum then puts nearly all mass on tree learners. Empirically,
tree mass is \(1.000\) on four datasets and \(0.889\) on OPSD household
(`results/cse_weights_by_dataset.csv`). On ETTh2 the blend is exactly
LightGBM (\(w=1\)).

## KKT exclusion boundary

Write validation errors \(e_i = y^{\mathrm{val}} - \hat y_i^{\mathrm{val}}\),
\(\sigma_i = \mathrm{std}(e_i)\), and \(\rho_{ij}=\mathrm{corr}(e_i,e_j)\).
For the simplex problem \(\min_{w\in\Delta} w^\top \Sigma w\) with
\(\Sigma_{ij}=\langle e_i,e_j\rangle / n\), the KKT conditions imply a
two-member exclusion rule (Proposition 1 in the paper). Relative to the
best member (lowest \(\sigma\)), a weaker member \(i\) receives **exactly
zero weight** when its error correlation with the leader is high enough:

\[
\rho_{i,\mathrm{best}} \;\ge\; \frac{\sigma_{\mathrm{best}}}{\sigma_i}.
\]

Points **above** the curve \(\rho = \sigma_{\mathrm{best}}/\sigma_i\) in the
\((\sigma_i/\sigma_{\mathrm{best}},\,\rho)\) plane are excluded. The figure
`figures/fig2_kkt_boundary.pdf` plots every non-leading member on every
dataset against that curve. Across 20 non-leading pairs there is no
counterexample to the boundary; agreement with the NNLS zeros is exact in
17/20 cases. Three admissible members are zeroed by collinearity under the
full KKT system; of 11 admissible members, 8 receive positive weight.

An instructive exception is OPSD national load: SARIMA has enormous
\(\sigma\) (\(\approx 243\times\) the leader) but near-orthogonal errors, so
the boundary admits a tiny positive weight.

## Protocols and statistics

- **Teacher forced (one-step):** supervised models see true lag features.  
- **Recursive:** models feed on their own predictions.  
- Metrics: MAE, RMSE, MAPE, sMAPE, \(R^2\), MASE (\(m=24\)).  
- Ranks for omnibus tests use `rankdata(..., method="average")` on the 19
  models shared by all five series.  
- Pairwise tests: Diebold–Mariano with HLN correction, Holm adjustment,
  Cohen’s \(d\), Cliff’s \(\delta\); Friedman + Nemenyi CD diagrams.

See `docs/rank_and_dm.md` for the CSE win/tie/lose rule against the RMSE leader.

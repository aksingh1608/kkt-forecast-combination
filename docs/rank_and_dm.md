# Rank function and win/tie/lose (Phase 1 pre-check)

This note records the canonical rank function and the win/tie/lose rule used in
the public README (confirmed for release).

## Rank function

**Canonical definition (use everywhere):**

```python
from scipy.stats import rankdata
ranks = rankdata(rmse_vector, method="average")  # lower RMSE → better → rank 1
```

Applied per dataset to the **19 models present on all five series**, one-step
RMSE from `results/leaderboards.csv`.

### Per-dataset ranks (full float precision)

| dataset | CSE rank | QuantileLightGBM | ExtraTrees | CSE RMSE | notes |
|---------|---------:|-----------------:|-----------:|---------:|-------|
| ETTh1 | 10 | 5 | 7 | 0.6814 | vs GradientBoosting RMSE 0.6811 (not a tie; GB rank 9) |
| ETTh2 | 4.5 | 8 | 1 | 0.7035 | **exact tie** with LightGBM RMSE 0.7035 → average ranks 4.5 / 4.5 |
| OPSD_DE_household | 2 | 1 | 15 | 0.1590 | |
| OPSD_DE_total | 1 | 7 | 3 | 592.7 | |
| UCI | 4 | 2 | 1 | 0.6390 | |

**CSE mean rank** = (10 + 4.5 + 2 + 1 + 4) / 5 = **4.3** exactly (21.5 / 5).

The earlier “4.4” figure came from integer dense ranks that broke the ETTh2
CSE–LightGBM tie (assigning 4 and 5 instead of 4.5 / 4.5).

### Leaderboard vs CD diagram

| Location | Rank function |
|----------|----------------|
| `results/leaderboards.csv` | **No rank column** |
| `scripts/run_significance_analysis.py` (Friedman + CD) | `DataFrame.rank(axis=1, ascending=True, method="average")` |

`pandas.DataFrame.rank(..., method="average")` matches `scipy.stats.rankdata(..., method="average")` on these matrices (0 mismatches). **Not a bug** — the CD path already uses average ties. Use the same helper in any new verify/README tooling.

## Win / tie / lose (README rule)

Rule: for each dataset, take the best one-step RMSE model as DM **reference**.
Classify CSE using Holm-adjusted `p_adj` at α = 0.05 from
`results/significance/dm_matrix__{dataset}__one_step.csv`:

- **win** if CSE is the reference
- **tie** if `p_adj >= 0.05` (CSE behind on RMSE but not significant)
- **loss** if `p_adj < 0.05` and CSE is behind

DM files label the ensemble as `FAHE_(Ours)` (underscore), while the leaderboard
uses `FAHE (Ours)`.

| dataset | reference | best RMSE | CSE RMSE | outcome | p_adj | Cohen's d | Cliff's δ |
|---------|-----------|----------:|---------:|---------|------:|----------:|----------:|
| ETTh1 | Ridge | 0.6313 | 0.6814 | **loss** | 1.297e-13 | -0.1553 | -0.0648 |
| ETTh2 | ExtraTrees | 0.6843 | 0.7035 | **tie** | 0.0662 | -0.0502 | -0.0199 |
| OPSD_DE_household | QuantileLightGBM | 0.1560 | 0.1590 | **loss** | 0.0413 | -0.3006 | -0.2030 |
| OPSD_DE_total | FAHE (Ours) | 592.7 | 592.7 | **win** | — | — | — |
| UCI | ExtraTrees | 0.6325 | 0.6390 | **tie** | 0.3832 | -0.0213 | -0.0009 |

Counts under this rule: **1 win, 2 ties, 2 losses**.

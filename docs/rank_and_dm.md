# Rank function and win/tie/lose rule

This note records the rank function and the win/tie/lose rule used in the
README and the manuscript. The manuscript reads the same values through
`scripts/build_paper.py`.

## Rank function

```python
from scipy.stats import rankdata
ranks = rankdata(rmse_vector, method="average")  # lower RMSE, better, rank 1
```

Applied per dataset to the **19 models present on all five series**, one-step
RMSE from `results/leaderboards.csv`.

### Per-dataset ranks

| dataset | CSE rank | ExtraTrees | Ridge | CSE RMSE | notes |
|---------|---------:|-----------:|------:|---------:|-------|
| ETTh1 | 8 | 7 | 1 | 0.6717 | |
| ETTh2 | 4.5 | 1 | 2 | 0.7035 | exact tie with LightGBM (the CSE weight on LightGBM is 1), so both take rank 4.5 |
| OPSD_DE_household | 4 | 10 | 1 | 0.1546 | |
| OPSD_DE_total | 1 | 3 | 10 | 591.8 | |
| UCI | 4 | 1 | 9 | 0.6385 | |
| **mean rank** | **4.3** | 4.4 | 4.6 | | lowest three of the 19 |

CSE mean rank = (8 + 4.5 + 4 + 1 + 4) / 5 = 4.3.

### Leaderboard and CD diagram

| Location | Rank function |
|----------|---------------|
| `results/leaderboards.csv` | No rank column |
| `scripts/run_significance_analysis.py` (Friedman and CD) | `DataFrame.rank(axis=1, ascending=True, method="average")` |

`pandas.DataFrame.rank(..., method="average")` gives the same ranks as
`scipy.stats.rankdata(..., method="average")`.

## Win, tie, lose

For each dataset, take the best one-step RMSE model as the Diebold-Mariano
**reference**. Classify the CSE using the Holm-adjusted `p_adj` at 0.05 from
`results/significance/dm_matrix__{dataset}__one_step.csv`:

- **win** if the CSE is the reference
- **tie** if `p_adj >= 0.05` (the CSE is behind on RMSE, not significantly)
- **loss** if `p_adj < 0.05` and the CSE is behind

DM files label the ensemble `FAHE_(Ours)`; the leaderboard uses `FAHE (Ours)`.

| dataset | reference | best RMSE | CSE RMSE | outcome | p_adj | Cohen's d | Cliff's delta |
|---------|-----------|----------:|---------:|---------|------:|----------:|--------------:|
| ETTh1 | Ridge | 0.6313 | 0.6717 | **loss** | 6.3e-10 | -0.130 | -0.052 |
| ETTh2 | ExtraTrees | 0.6843 | 0.7035 | **tie** | 0.066 | -0.050 | -0.020 |
| OPSD_DE_household | Ridge | 0.1473 | 0.1546 | **loss** | < 1e-4 | -0.278 | -0.116 |
| OPSD_DE_total | CSE | 591.8 | 591.8 | **win** | n/a | n/a | n/a |
| UCI | ExtraTrees | 0.6325 | 0.6385 | **tie** | 0.467 | -0.016 | 0.001 |

Counts under this rule: **1 win, 2 ties, 2 losses**.

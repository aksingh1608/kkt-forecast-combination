# Table map

| Manuscript table (role) | CSV / TeX | Script | Source |
|-------------------------|-----------|--------|--------|
| One-step / recursive leaderboard | `results/tables/table_onestep_metrics.*`, `table_recursive_metrics.*`, `table_leaderboards_long.*` | `scripts/build_tables.py` | `results/leaderboards.csv` |
| CSE weights / collapse | `results/tables/table_cse_weights.*` | `scripts/build_tables.py` | `results/cse_weights_by_dataset.csv` |
| Horizon metrics | `results/tables/table_horizons_rolling.*` | `scripts/build_tables.py` | **`results/multistep_rolling/long_all__rolling.csv` only** |
| Friedman omnibus | `results/tables/table_friedman.*` | `scripts/build_tables.py` | `results/significance/friedman_summary.csv` |
| DM / effect sizes | `results/significance/dm_matrix__{dataset}__{protocol}.csv` | `scripts/run_significance_analysis.py` | `results/predictions/` |

**Never** populate horizon tables from `results/multistep/` (prefix diagnostic).

```bash
make tables
```

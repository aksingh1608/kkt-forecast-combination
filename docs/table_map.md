# Table map

Every results table in the manuscript is written by `scripts/build_paper.py`
into `paper/generated/` and included by `paper/main.tex`. Table I (the datasets)
is written directly in `paper/main.tex`. Table numbers are those of the full
version; `paper/README.md` explains the shorter version.

| Table | Content | Generated file | Source under `results/` |
|-------|---------|----------------|-------------------------|
| II | Test RMSE, both protocols, mean rank | `tab_main.tex` | `leaderboards.csv` |
| III | Degradation ratios and rolling-origin ranks | `tab_family.tex` | `leaderboards.csv`, `multistep_rolling/long_all__rolling.csv` |
| IV | CSE weights and tree mass | `tab_weights.tex` | `cse_diagnostics/` |
| V | Exclusion boundary against learned weight | `tab_agree.tex` | `cse_diagnostics/` |
| VI | Member-pool sensitivity | `tab_sensitivity.tex` | `tables/membership_sensitivity.csv`, `tables/membership_sensitivity_weights.csv` |
| VII | Expanding-window folds | `tab_folds.tex` | `tables/cse_folds.csv`, `tables/cse_folds_weights.csv` |
| VIII | Diebold-Mariano, CSE against the best model | `tab_dm.tex` | `significance/dm_matrix__*.csv` |
| IX | Tuned NLinear and DLinear configurations | `tab_tuned.tex` | `tuning/lite_dl/` |
| X | CSE time when only active members are refit | `tab_refit.tex` | `tables/cse_refit_times.csv` |
| XI | Training times | `tab_cost.tex` | `leaderboards.csv` |
| XII | Rolling-origin RMSE by horizon (appendix) | `tab_app.tex` | `multistep_rolling/long_all__rolling.csv` |

Inline numbers in the text are macros in `paper/generated/numbers.tex`, written
by the same script.

Horizon results come only from `results/multistep_rolling/`.

```bash
python scripts/build_paper.py
```

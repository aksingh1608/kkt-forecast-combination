# Convex Forecast Combination and Sparsity via KKT Boundaries: Multi-Dataset Benchmarking in Energy Time Series

Companion code for a manuscript under submission to an IEEE Transactions journal. Convex stacking over classical and tree forecasters collapses onto tree learners; a KKT boundary predicts which members receive zero weight.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](requirements.txt)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](CITATION.cff)

![CSE architecture](figures/fig1_architecture.png)

## Summary

A **Convex Stacking Ensemble (CSE)** blends five members (SARIMA, Holt–Winters, LightGBM, CatBoost, Extra-Trees) using the exact least-squares solution on the probability simplex, computed on a validation tail ([`docs/method.md`](docs/method.md), `CHANGELOG.md`). Members with positive weight are then refit on the full training set. Under recursion the ensemble rolls out as one unit.

Because validation errors of weaker members are often large and highly correlated with the best tree, the simplex optimum puts nearly all mass on trees. The KKT geometry of that problem yields an exclusion boundary $\rho_i \ge \sigma_{\mathrm{best}}/\sigma_i$: members above it take weight exactly zero. Details: [`docs/method.md`](docs/method.md).

## Key results

Every number links to a file under [`results/`](results/). Rank definition: `scipy.stats.rankdata(method="average")` on the 19 models shared by all five series ([`docs/rank_and_dm.md`](docs/rank_and_dm.md)).

| Claim | Value | Source |
|-------|-------|--------|
| Tree mass of CSE weights (exact simplex solver) | rounds to 1.000 on all five series | [`results/cse_weights_by_dataset.csv`](results/cse_weights_by_dataset.csv) |
| Exact single-member collapse | ETTh2 → LightGBM weight 1.0 | same |
| Pairwise screen vs simplex zeros | 18/20 exact; 11 excluded, none with weight; 9 admissible, 7 with positive weight; full KKT system holds for 25/25 members | same (+ `results/tables/kkt_simplex_recheck.csv`) |
| OPSD-N SARIMA scale | validation RMSE ≈ 508× the leader (uncentred); weight ≈ 5.5e-5 | same |
| Friedman (one-step RMSE) | χ² = 69.47, p = 5.6×10⁻⁸ | [`results/significance/friedman_summary.csv`](results/significance/friedman_summary.csv) |
| Friedman (recursive RMSE) | χ² = 19.60, p = 0.356 | same |
| CSE mean rank (one-step) | **4.3** (best of 19, ExtraTrees 4.4) | [`results/leaderboards.csv`](results/leaderboards.csv) |
| CSE vs RMSE leader (Holm α=0.05) | win OPSD-N; tie UCI & ETTh2; **loss OPSD-H & ETTh1 (both to Ridge)** | [`results/significance/dm_matrix__*__one_step.csv`](results/significance/) |
| CSE training cost | about 34 to 390 times HistGradientBoosting across the five datasets, median near 170 | [`results/leaderboards.csv`](results/leaderboards.csv) |

All numbers come from the exact simplex solver (v1.1) and the OPSD household target as corrected in v1.2; see `CHANGELOG.md` and [`DATA.md`](DATA.md). The manuscript in `paper/` reads every number from `results/` through `scripts/build_paper.py`.

Negative results are intentional: under one-step evaluation CSE is not uniformly best. Losses (significant vs the RMSE leader): **OPSD_DE_household** and **ETTh1**, both against Ridge. See [`docs/rank_and_dm.md`](docs/rank_and_dm.md).

## Figures

![KKT boundary](figures/fig2_kkt_boundary.png)

![CD diagram one-step](figures/fig3_cd_diagrams_one_step.png)

Map: [`docs/figure_map.md`](docs/figure_map.md).

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data          # or: place files per DATA.md
make smoke         # < 2 minutes, CPU
```

`requirements.txt` is the CPU set that reproduces `results/` and runs the tests. The CUDA and GPU packages are in `requirements-gpu.txt`, and the optional deep learning models and Streamlit UI are in `requirements-optional.txt`.

## Reproduce

All published artefacts were produced on CPU, with seeds fixed at 42 (`src.seed.set_global_seed`). The training steps take many hours. Run in this order:

1. `make data`: downloads the five series into `data/raw/` and checks their SHA-256 digests.
2. `python scripts/rerun_dataset.py --datasets UCI ETTh1 ETTh2 OPSD_DE_household OPSD_DE_total`: writes the standalone model rows of `results/leaderboards.csv` and their forecasts in `results/predictions/`.
3. `python scripts/rerun_lite_dl_tuned.py --out results/ --trials 40`: writes the tuned NLinear and DLinear rows, their forecasts and `results/tuning/lite_dl/`.
4. `python scripts/run_rolling_origin_benchmark.py --out results/multistep_rolling/ --n-origins 24 --n-jobs 1`: writes the rolling-origin horizon tables in `results/multistep_rolling/`.
5. `python scripts/rerun_cse.py`: writes the CSE rows, forecasts and rolling-origin rows, `results/cse_diagnostics/`, `results/cse_weights_by_dataset.csv` and `results/tables/cse_refit_times.csv`.
6. `python scripts/run_significance_analysis.py --in results/`: writes `results/significance/` and the heat maps and CD diagrams in `results/figures/`.
7. `python scripts/membership_sensitivity.py`: writes `results/tables/membership_sensitivity*.csv`.
8. `python scripts/multi_fold_cse.py`: writes `results/tables/cse_folds*.csv`.
9. `python scripts/build_figures.py`: writes `figures/fig2_*`, `figures/fig3_*` and `figures/fig5_dm_heatmap__*`.
10. `python scripts/build_paper_figures.py`: writes `figures/fig4_dm_pvalues.*`, `figures/fig5_error_drift_uci.*` and `figures/fig6_horizon_ranks.*`.
11. `python scripts/build_paper.py`: writes the manuscript tables and inline numbers to `paper/generated/`.
12. `python scripts/verify_claims.py`: writes nothing; it checks the headline claims against `results/` and exits with an error if one fails.

`scripts/rerun_dataset.py` replaces rows in the shipped `results/leaderboards.csv`, so it needs that file to be present. A clone that includes `results/` can run steps 9 to 12 without retraining. `make smoke` is a short check of the code path.

## Repository map

| Path | Role |
|------|------|
| `src/` | Library (models, features, evaluation, pipeline) |
| `src/cse/` | Paper-facing CSE package (re-exports; FAHE ≡ CSE) |
| `scripts/` | Download, benchmark, rolling origin, statistics, figures, manuscript tables, verify |
| `results/` | All published artefacts ([`results/README.md`](results/README.md)) |
| `figures/` | Regenerated manuscript figures |
| `paper/` | Manuscript, Fig. 1 TikZ source, generated tables and numbers ([`paper/README.md`](paper/README.md)) |
| `docs/` | Method, feature list, figure/table maps |
| `extras/` | Optional Streamlit UI |
| `tests/` | Unit tests |

## Scope and limitations

- **No Transformer results.** Autoformer, FEDformer, Informer, iTransformer, PatchTST and related models exist as optional code paths (`src/models/optional_dl.py`) and produced **no** artefacts; they appear in no result in the paper.
- **OPSD-N carries 19 models, not 23**, because standalone heavy classical fits exceeded the memory budget. CSE still keeps internal SARIMA and Holt–Winters branches there.
- Horizon tables use `results/multistep_rolling/`.
- The main results use one chronological split per dataset, with seeds fixed at 42. Expanding-window folds for the CSE are in `results/tables/cse_folds.csv`.
- Under one-step evaluation, CSE wins on one dataset, ties on two, and loses on two (Holm-adjusted DM vs the RMSE leader; see Key results).

## Naming map

| Code / CSV | Manuscript |
|------------|------------|
| `FAHE`, `FAHEForecaster`, `FAHE (Ours)` | **CSE** (Convex Stacking Ensemble) |
| `CSEForecaster`, `CSE (Ours)` | Preferred alias for the same class |

They are **one method**, not two.

## Citation

Please cite the **paper**, not only this repository. Metadata: [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{singh2026convex,
  title={Convex Forecast Combination and Sparsity via {KKT} Boundaries:
         Multi-Dataset Benchmarking in Energy Time Series},
  author={Singh, Aryan Kumar and Al Farid, Fahmid and Al Qwaid, Marran
          and Miah, Abu Saleh Musa and Islam, Syed Mohammed Shamsul
          and Husen, Mohd Nizam},
  year={2026},
  note={Manuscript under submission}
}
```

## Licence, contact, acknowledgements

- Code and documentation: [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
- Datasets: upstream terms only — [DATA.md](DATA.md).
- Corresponding author: Fahmid Al Farid.
- Code: Aryan Kumar Singh, ORCID [0009-0003-8791-1869](https://orcid.org/0009-0003-8791-1869), [github.com/aksingh1608](https://github.com/aksingh1608).

Feature column listing for methods text: [`docs/features.md`](docs/features.md).

# Figure map

| Manuscript figure | Output files | Script | Source artefacts |
|-------------------|--------------|--------|------------------|
| Fig. 1 Architecture | `figures/fig1_architecture.png`, `.pdf` | `scripts/build_figures.py` (+ `paper/fig1_architecture.tex`) | TikZ / matplotlib fallback |
| Fig. 2 KKT boundary | `figures/fig2_kkt_boundary.png`, `.pdf` | `scripts/build_figures.py` / `plot_kkt_boundary.py` | `results/cse_diagnostics/*.pkl` |
| Fig. 3 CD diagrams | `figures/fig3_cd_diagrams_one_step.*`, `fig3_cd_diagrams_recursive.*` | `scripts/build_figures.py` | `results/leaderboards.csv` (19 shared models, average ranks) |
| Fig. 4 Error drift | `figures/fig4_error_drift_*.pdf` | `scripts/build_figures.py` ← `plot_error_drift.py` | `results/predictions/*.parquet` |
| Fig. 5 DM heatmaps | `figures/fig5_dm_heatmap__*.pdf` | `scripts/build_figures.py` ← `run_significance_analysis.py` | `results/significance/dm_matrix__*.csv` |

Regenerate:

```bash
make figures
```

All figures are produced from `results/` (or TikZ for Fig. 1). No manual exports.

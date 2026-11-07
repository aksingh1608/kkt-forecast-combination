# Figure map

File names do not always carry the manuscript number, because the manuscript
order changed after the files were named. Figure numbers are those of the full
version; `paper/README.md` explains the shorter version.

| Manuscript figure | Output files | Script | Source |
|-------------------|--------------|--------|--------|
| Fig. 1 Architecture | TikZ inside the paper; `figures/fig1_architecture.png`, `.pdf` are rendered copies | `paper/fig1_architecture.tex` | `paper/generated/numbers.tex` |
| Fig. 2 KKT boundary | `figures/fig2_kkt_boundary.png`, `.pdf` | `scripts/build_figures.py` | `results/cse_diagnostics/*.pkl` |
| Fig. 3 Rolling-origin rank by horizon | `figures/fig6_horizon_ranks.png`, `.pdf` | `scripts/build_paper_figures.py` | `results/multistep_rolling/` |
| Fig. 4 CD diagrams | `figures/fig3_cd_diagrams_one_step.*`, `fig3_cd_diagrams_recursive.*` | `scripts/build_figures.py` | `results/leaderboards.csv` (19 shared models, average ranks) |
| Fig. 5 DM p-values | `figures/fig4_dm_pvalues.png`, `.pdf` | `scripts/build_paper_figures.py` | `results/significance/dm_matrix__*.csv` |
| Fig. 6 Error drift on UCI | `figures/fig5_error_drift_uci.png`, `.pdf` | `scripts/build_paper_figures.py` | `results/predictions/*.parquet` |

`figures/fig5_dm_heatmap__*` (one heat map per dataset and protocol, copied by
`scripts/build_figures.py` from `results/figures/`) are not used by the paper.

`scripts/build_figures.py` keeps the existing `fig1_architecture` files and does
not re-render them. If `paper/fig1_architecture.tex` or the numbers change,
render the figure again by hand.

Regenerate:

```bash
python scripts/build_figures.py
python scripts/build_paper_figures.py
```

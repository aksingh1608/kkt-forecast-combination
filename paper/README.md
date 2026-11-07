# Manuscript

`main.tex` is the IEEE Transactions manuscript (IEEEtran journal class).
Nothing in it is typed by hand: every table and every inline number is
read from `generated/`, which `scripts/build_paper.py` writes from
`results/`.

```bash
python scripts/build_paper.py          # -> paper/generated/*.tex
python scripts/build_figures.py        # -> figures/fig2_kkt_boundary.pdf, fig3_cd_diagrams_*.pdf
python scripts/build_paper_figures.py  # -> figures/fig4_dm_pvalues.pdf, fig5_error_drift_uci.pdf, fig6_horizon_ranks.pdf
cd paper && pdflatex main && pdflatex main && pdflatex main
```

## Two versions from one file

`\fullversiontrue` near the top of `main.tex` builds the full paper (about
14 pages: six figures, twelve tables, appendix). `\fullversionfalse` builds
the 10-page submission version: Fig. 1 and Figs. 4 to 6, the pool, fold,
tuned, refit-time and cost tables and the appendix table are left out and
referred to as supplementary material (Tables S1 to S5). `make paper-short`
does this without editing the file. The supplementary file is the full
version.

Files

| File | Purpose |
|------|---------|
| `main.tex` | the manuscript |
| `fig1_architecture.tex` | Fig. 1 (TikZ), numbers taken from `generated/numbers.tex` |
| `generated/numbers.tex` | `\newcommand` macros for every inline number |
| `generated/tab_*.tex` | table bodies (Tables II to IX and the appendix) |

## Red entries

Until every dataset has been refit with the exact simplex solver
(`python scripts/rerun_cse.py`), the CSE results of the datasets that are
still missing from `results/tables/cse_refit_times.csv` print in red. The
colour is switched off automatically by `build_paper.py` once no dataset
is pending.

## Overleaf

Upload the `paper/` folder together with `figures/`, or the zip produced by
`make paper-zip`. Compiler: pdfLaTeX. Main document: `paper/main.tex`.

# Manuscript

`main.tex` is the manuscript (IEEEtran journal class). Every results table and
every inline result number is read from `generated/`, which
`scripts/build_paper.py` writes from `results/`. Table I, which describes the
datasets, is written in `main.tex`.

```bash
python scripts/build_paper.py          # -> paper/generated/*.tex
python scripts/build_figures.py        # -> figures/fig2_kkt_boundary.pdf, fig3_cd_diagrams_*.pdf
python scripts/build_paper_figures.py  # -> figures/fig4_dm_pvalues.pdf, fig5_error_drift_uci.pdf, fig6_horizon_ranks.pdf
cd paper && pdflatex main && pdflatex main && pdflatex main
```

## Two versions from one file

`\fullversiontrue` near the top of `main.tex` builds the full paper (about
14 pages: six figures, twelve tables, appendix). `\fullversionfalse` builds
the 10-page submission version: the architecture figure, the horizon-rank,
DM p-value and error-drift figures, the pool, fold, tuned, refit-time and cost
tables and the appendix table are left out and
referred to as supplementary material (Tables S1 to S5). `make paper-short`
does this without editing the file. The supplementary file is the full
version.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | the manuscript |
| `fig1_architecture.tex` | Fig. 1 (TikZ), numbers taken from `generated/numbers.tex` |
| `generated/numbers.tex` | `\newcommand` macros for every inline number |
| `generated/tab_*.tex` | table bodies (Tables II to XII in the full version; see `docs/table_map.md`) |

## Overleaf

Upload the `paper/` folder together with `figures/`, or the zip produced by
`make paper-zip`. Compiler: pdfLaTeX. Main document: `paper/main.tex`.

File names in `figures/` do not always carry the manuscript number; see
`docs/figure_map.md`.

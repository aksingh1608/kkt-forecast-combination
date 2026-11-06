# Manuscript

`main.tex` is the IEEE Transactions manuscript (IEEEtran journal class).
Nothing in it is typed by hand: every table and every inline number is
read from `generated/`, which `scripts/build_paper.py` writes from
`results/`.

```bash
python scripts/build_paper.py          # -> paper/generated/*.tex
python scripts/build_figures.py        # -> figures/fig2_kkt_boundary.pdf, fig3_cd_diagrams_*.pdf
cd paper && pdflatex main && pdflatex main && pdflatex main
```

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

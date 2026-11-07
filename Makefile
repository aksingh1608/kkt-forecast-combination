# Reproduce paper artefacts on CPU (Python 3.12).
# Typical wall-clock on a 8-core laptop (approximate):
#   data        minutes (network)
#   features    <1 min
#   smoke       <2 min
#   benchmark   many hours
#   rolling     many hours
#   stats       minutes
#   figures     minutes
#   verify      <1 min

PYTHON ?= .venv/bin/python
ifeq ($(wildcard $(PYTHON)),)
  PYTHON := python3
endif

.PHONY: help data features smoke benchmark rolling stats figures verify all rerun-cse recheck sensitivity folds paper paper-short paper-zip

help:
	@echo "Targets: data features smoke benchmark rolling stats figures verify all recheck rerun-cse sensitivity folds paper paper-zip"

data:
	$(PYTHON) scripts/download_data.py

features:
	$(PYTHON) -c "from src.pipeline import load_project_dataset; from src.config import FeatureConfig; from src.features.engineering import build_supervised_frame; df,t,_=load_project_dataset('ETTh1'); X,y,n=build_supervised_frame(df,t,FeatureConfig()); assert len(n)==57; print('features_ok', len(n), X.shape)"

smoke:
	$(PYTHON) scripts/smoke_test.py

benchmark:
	$(PYTHON) scripts/run_paper_experiments.py --out results/ --n-jobs 1

rolling:
	$(PYTHON) scripts/run_rolling_origin_benchmark.py --out results/multistep_rolling/ --n-origins 24 --n-jobs 1

stats:
	$(PYTHON) scripts/run_significance_analysis.py --in results/
	$(PYTHON) scripts/plot_kkt_boundary.py --in results/ --out results/figures/kkt_boundary.pdf

figures:
	$(PYTHON) scripts/build_figures.py --results results/ --out figures/

verify:
	$(PYTHON) scripts/verify_claims.py

# --- exact simplex solver (v1.1) ------------------------------------------
recheck:
	$(PYTHON) scripts/recheck_simplex_kkt.py --update-artefacts --out results/tables/kkt_simplex_recheck.csv

rerun-cse:
	$(PYTHON) scripts/rerun_cse.py
	$(PYTHON) scripts/run_significance_analysis.py --in results/

sensitivity:
	$(PYTHON) scripts/membership_sensitivity.py

folds:
	$(PYTHON) scripts/multi_fold_cse.py

paper: figures
	$(PYTHON) scripts/build_paper_figures.py
	$(PYTHON) scripts/build_paper.py
	cd paper && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main | tail -2

# the 10-page submission version: supplementary tables and four figures left out (see paper/README.md)
paper-short: paper
	cd paper && sed 's/^\\fullversiontrue/\\fullversionfalse/' main.tex > main_short.tex && pdflatex -interaction=nonstopmode main_short >/dev/null && pdflatex -interaction=nonstopmode main_short >/dev/null && pdflatex -interaction=nonstopmode main_short | tail -2

paper-zip: paper
	rm -f paper_overleaf.zip
	zip -qr paper_overleaf.zip paper/main.tex paper/fig1_architecture.tex paper/generated paper/README.md figures/fig2_kkt_boundary.pdf figures/fig3_cd_diagrams_one_step.pdf figures/fig3_cd_diagrams_recursive.pdf figures/fig4_dm_pvalues.pdf figures/fig5_error_drift_uci.pdf figures/fig6_horizon_ranks.pdf
	@echo wrote paper_overleaf.zip

# End-to-end: download → feature check → full benchmark chain → derived artefacts.
# Prefer `make verify` after a clone that already ships `results/`.
all: data features benchmark rolling stats figures verify

# Copyright 2026 Aryan Kumar Singh and co-authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Statistical-significance analysis for the paper.

Reads the persisted predictions from ``run_paper_experiments.py`` and
produces:

* ``results/significance/dm_matrix__{dataset}__{protocol}.csv`` —
  pairwise Diebold–Mariano (HLN-corrected) versus the best model,
  with Holm-adjusted p-values, Cohen's d and Cliff's δ.
* ``results/figures/dm_heatmap__{dataset}__{protocol}.pdf`` —
  the Holm-corrected DM p-value heatmap.
* ``results/significance/friedman_summary.csv`` — omnibus
  Friedman test on the (dataset × model) rank matrix.
* ``results/figures/cd_diagram__{protocol}__{metric}.pdf`` —
  Nemenyi critical-difference diagram (paper's cross-dataset figure).

Run::

    python scripts/run_significance_analysis.py --in results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_tests import (
    pairwise_dm_matrix, cohens_d, cliffs_delta, friedman_test,
    plot_dm_pvalue_heatmap,
)
from src.evaluation.cd_diagram import plot_cd_diagram

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("significance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=Path,
                   default=Path("results"))
    p.add_argument("--protocols", nargs="+",
                   default=["one_step", "recursive"])
    p.add_argument("--metric", default="RMSE",
                   help="Metric used for ranking (CD diagram + best-model selection).")
    p.add_argument("--dm-horizon", type=int, default=1,
                   help="DM Newey-West bandwidth (h in DM/HLN). 1 for one_step; use 24 for recursive.")
    return p.parse_args()


def _load_all(in_dir: Path, protocol: str):
    pred_dir = in_dir / "predictions"
    y_true_files = list(pred_dir.glob(f"*__{protocol}__y_true.parquet"))
    out = {}
    for f in y_true_files:
        dataset = f.name.split("__")[0]
        y_true = pd.read_parquet(f)["y_true"]
        preds = {}
        for pf in pred_dir.glob(f"{dataset}__{protocol}__*.parquet"):
            if pf.name.endswith("__y_true.parquet"):
                continue
            name = pf.stem.split(f"__{protocol}__")[1]
            preds[name] = pd.read_parquet(pf)["y_pred"].values
        out[dataset] = (y_true.values, preds)
    return out


def _run_dm(y_true, preds, dm_h: int) -> pd.DataFrame:
    if not preds:
        return pd.DataFrame()
    rmses = {n: float(np.sqrt(np.mean((y_true - np.asarray(p)) ** 2)))
             for n, p in preds.items()}
    best = min(rmses, key=rmses.get)
    dm = pairwise_dm_matrix(y_true, preds, reference=best, h=dm_h,
                            power=2.0, correction="holm")
    dm["cohens_d"] = [cohens_d(y_true, preds[best], preds[m]) for m in dm["model"]]
    dm["cliffs_delta"] = [cliffs_delta(y_true, preds[best], preds[m]) for m in dm["model"]]
    dm["rmse_ref"] = rmses[best]
    dm["rmse_model"] = [rmses[m] for m in dm["model"]]
    return dm


def main() -> None:
    args = parse_args()
    in_dir = args.in_dir
    sig_dir = in_dir / "significance"
    fig_dir = in_dir / "figures"
    sig_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    lb_path = in_dir / "leaderboards.csv"
    if not lb_path.exists():
        raise FileNotFoundError(f"{lb_path} not found — run run_paper_experiments.py first.")
    leaderboards = pd.read_csv(lb_path)

    friedman_rows = []
    for protocol in args.protocols:
        loaded = _load_all(in_dir, protocol)
        for dataset, (y_true, preds) in loaded.items():
            if not preds:
                continue
            dm = _run_dm(y_true, preds, dm_h=args.dm_horizon)
            path = sig_dir / f"dm_matrix__{dataset}__{protocol}.csv"
            dm.to_csv(path, index=False)
            logger.info("Wrote %s", path)
            try:
                fig = plot_dm_pvalue_heatmap(
                    dm, title=f"DM vs. {dm['reference'].iloc[0]} — {dataset} ({protocol})"
                )
                fpath = fig_dir / f"dm_heatmap__{dataset}__{protocol}.pdf"
                fig.savefig(fpath, bbox_inches="tight")
                logger.info("Wrote %s", fpath)
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception as exc:
                logger.warning("DM heatmap failed for %s/%s: %s", dataset, protocol, exc)

        # Friedman + CD across datasets under this protocol.
        proto_lb = leaderboards[leaderboards["Protocol"] == protocol]
        if proto_lb.empty:
            continue
        pivot = proto_lb.pivot_table(
            index="Dataset", columns="Model", values=args.metric, aggfunc="first"
        )
        pivot = pivot.dropna(axis=1, how="any")
        if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
            ascending = args.metric != "R2"
            ranks = pivot.rank(axis=1, ascending=ascending, method="average")
            try:
                stat, p = friedman_test(ranks)
                friedman_rows.append({
                    "protocol": protocol, "metric": args.metric,
                    "n_datasets": pivot.shape[0], "n_models": pivot.shape[1],
                    "friedman_stat": stat, "p_value": p,
                })
            except Exception as exc:
                logger.warning("Friedman failed for %s: %s", protocol, exc)
            try:
                fig, cd = plot_cd_diagram(
                    ranks, title=f"CD diagram — {protocol} ({args.metric})"
                )
                fpath = fig_dir / f"cd_diagram__{protocol}__{args.metric}.pdf"
                fig.savefig(fpath, bbox_inches="tight")
                logger.info("Wrote %s (CD=%.2f, n_datasets=%d)",
                            fpath, cd.cd, cd.n_datasets)
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception as exc:
                logger.warning("CD diagram failed for %s: %s", protocol, exc)

    if friedman_rows:
        pd.DataFrame(friedman_rows).to_csv(
            sig_dir / "friedman_summary.csv", index=False
        )
        logger.info("Wrote %s", sig_dir / "friedman_summary.csv")


if __name__ == "__main__":
    main()

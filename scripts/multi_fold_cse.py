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

"""Expanding-window folds for the Convex Stacking Ensemble.

The main benchmark uses one chronological 80/20 split. This script refits
the CSE on expanding training windows ending at 50%, 60%, 70% and 80% of
each series, scores the following 20% block (or what remains) one step
ahead, and records per fold the weights, the tree mass, the Proposition 1
check and the test RMSE of the blend against every member. All five members
are refit on every fold so each member's own test RMSE is available.

Output: ``results/tables/cse_folds.csv`` and
``results/tables/cse_folds_weights.csv``.

Run::

    python scripts/multi_fold_cse.py --datasets ETTh1 ETTh2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.evaluation.metrics import rmse  # noqa: E402
from src.features.engineering import build_supervised_frame  # noqa: E402
from src.models.novel import FAHEForecaster  # noqa: E402
from src.pipeline import load_project_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("folds")

TREES = {"lgbm", "catboost", "extratrees"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--train-ends", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    p.add_argument("--test-frac", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or list(config.PROJECT_DATASETS)
    rows, wrows = [], []
    for dataset in datasets:
        df, target, meta = load_project_dataset(dataset)
        season = int(meta.get("season", 24))
        X, y, _ = build_supervised_frame(df, target, cfg=config.FeatureConfig())
        y = df[[target]].loc[X.index][target]
        n = len(X)
        for f in args.train_ends:
            tr_end = int(round(n * f))
            te_end = min(n, tr_end + int(round(n * args.test_frac)))
            X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
            X_te, y_te = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]
            cse = FAHEForecaster(seasonal_periods=season, refit_only_active=False)
            t0 = time.perf_counter()
            cse.fit(y_tr, X_train=X_tr)
            pred = cse.predict(len(y_te), X_test=X_te)
            fit_time = time.perf_counter() - t0
            diag = cse.diagnostics(dataset=dataset)
            frame = diag.to_frame()
            viol = 0
            for _, r in frame.iterrows():
                if r.is_best:
                    continue
                if r.rho_vs_best >= 1.0 / r.sigma_ratio_vs_best and r.weight > 1e-9:
                    viol += 1
            member_rmse = {k: rmse(y_te.values, v) for k, v in cse.base_forecasts_test_.items()}
            blend = rmse(y_te.values, pred)
            best_member = min(member_rmse, key=member_rmse.get)
            w = cse.weights_
            rows.append({
                "dataset": dataset, "train_end_frac": f, "n_train": tr_end,
                "n_test": te_end - tr_end,
                "tree_mass": round(sum(v for k, v in w.items() if k in TREES), 4),
                "n_zero": int(sum(v <= 1e-9 for v in w.values())),
                "argmax": max(w, key=w.get), "max_weight": round(max(w.values()), 4),
                "necessity_violations": viol,
                "test_rmse_cse": round(blend, 4),
                "test_rmse_best_member": round(member_rmse[best_member], 4),
                "best_member": best_member,
                "cse_minus_best_pct": round(100.0 * (blend / member_rmse[best_member] - 1.0), 2),
                **{f"rmse_{k}": round(v, 4) for k, v in member_rmse.items()},
                "fit_time_s": round(fit_time, 1),
            })
            for _, r in frame.iterrows():
                wrows.append({"dataset": dataset, "train_end_frac": f, "member": r.member,
                              "weight": round(float(r.weight), 6),
                              "rmse_ratio_vs_best": round(float(r.sigma_ratio_vs_best), 4),
                              "rho_vs_best": round(float(r.rho_vs_best), 4),
                              "is_best": bool(r.is_best)})
            logger.info("[%s] train_end=%.1f tree_mass=%.3f CSE=%.4f best=%.4f (%s) viol=%d %.0fs",
                        dataset, f, rows[-1]["tree_mass"], blend,
                        member_rmse[best_member], best_member, viol, fit_time)

    out_dir = args.results / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data, keys in (("cse_folds.csv", rows, ["dataset", "train_end_frac"]),
                             ("cse_folds_weights.csv", wrows, ["dataset", "train_end_frac", "member"])):
        new = pd.DataFrame(data)
        path = out_dir / name
        if path.exists():
            old = pd.read_csv(path)
            m = old.merge(new[keys].drop_duplicates(), on=keys, how="left", indicator=True)
            old = old[m["_merge"].values == "left_only"]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)
        print("wrote", path)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

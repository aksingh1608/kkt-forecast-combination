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

"""Sensitivity of the weight collapse to the member pool.

For each dataset and each pool below, stack the pool's registry models on
the simplex (validation tail, exact solve, one-step protocol) and record the
weights, the share of weight on tree and boosting members, the number of
members zeroed, whether any member above the Proposition 1 boundary received
weight (a necessity violation), and the test RMSE of the blend against its
strongest member.

Pools (registry names; heavy classical members are skipped on OPSD_DE_total
as in the main benchmark):

* paper5         SARIMA, Holt-Winters, LightGBM, CatBoost, ExtraTrees
* paper5+xgb_rf  paper5 plus XGBoost and RandomForest
* paper5+linear  paper5 plus Ridge, Lasso, ElasticNet
* paper5+lite_dl paper5 plus tuned NLinear and DLinear
* trees8         the eight tree and boosting models only
* all            every registry model except the CSE itself

Output: ``results/tables/membership_sensitivity.csv`` (one row per dataset
and pool) and ``results/tables/membership_sensitivity_weights.csv`` (one
row per member).

Run::

    python scripts/membership_sensitivity.py --datasets ETTh1 ETTh2
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data.preprocessing import train_test_split_temporal  # noqa: E402
from src.evaluation.metrics import rmse  # noqa: E402
from src.features.engineering import build_supervised_frame  # noqa: E402
from src.models.registry import build_registry  # noqa: E402
from src.models.stacking import ConvexStacker  # noqa: E402
from src.pipeline import load_project_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("membership")

TREES = {"RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting",
         "XGBoost", "LightGBM", "CatBoost", "QuantileLightGBM"}
PAPER5 = ["SARIMA", "Holt-Winters", "LightGBM", "CatBoost", "ExtraTrees"]
POOLS = {
    "paper5": PAPER5,
    "paper5+xgb_rf": PAPER5 + ["XGBoost", "RandomForest"],
    "paper5+linear": PAPER5 + ["Ridge", "Lasso", "ElasticNet"],
    "paper5+lite_dl": PAPER5 + ["NLinear", "DLinear"],
    "trees8": sorted(TREES),
    "all": None,
}


def _tuned_lite_dl():
    """Reuse the tuned NLinear/DLinear factories from the rolling-origin script."""
    path = ROOT / "scripts" / "run_rolling_origin_benchmark.py"
    spec = importlib.util.spec_from_file_location("rolling_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._apply_tuned_lite_dl_specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--pools", nargs="*", default=list(POOLS))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or list(config.PROJECT_DATASETS)
    apply_tuned = _tuned_lite_dl()
    rows, wrows = [], []

    for dataset in datasets:
        df, target, meta = load_project_dataset(dataset)
        season = int(meta.get("season", 24))
        X, y, _ = build_supervised_frame(df, target, cfg=config.FeatureConfig())
        df_target = df[[target]].loc[X.index]
        sx = train_test_split_temporal(X, test_size=args.test_size)
        sy = train_test_split_temporal(df_target, test_size=args.test_size)
        X_train, X_test = sx["train"], sx["test"]
        y_train, y_test = sy["train"][target], sy["test"][target]

        heavy_ok = dataset != "OPSD_DE_total"
        registry = build_registry(seasonal_period=season, n_jobs=args.n_jobs,
                                  include_heavy_classical=heavy_ok)
        registry = apply_tuned(registry, dataset, args.results)
        by_name = {s.name: s for s in registry if not s.name.lower().startswith("fahe")}

        for pool in args.pools:
            wanted = POOLS[pool] or list(by_name)
            specs = [by_name[n] for n in wanted if n in by_name]
            if len(specs) < 2:
                logger.warning("[%s] pool %s has fewer than two members; skipped", dataset, pool)
                continue
            t0 = time.perf_counter()
            st = ConvexStacker(specs)
            st.fit(y_train, X_train)
            pred = st.predict(len(y_test), X_test)
            fit_time = time.perf_counter() - t0
            diag = st.diagnostics(dataset=dataset)
            frame = diag.to_frame()
            best = frame.loc[frame.is_best, "member"].iloc[0]
            viol, excl, adm_pos, adm_zero = 0, 0, 0, 0
            for _, r in frame.iterrows():
                if r.is_best:
                    continue
                excluded = r.rho_vs_best >= 1.0 / r.sigma_ratio_vs_best
                if excluded:
                    excl += 1
                    viol += int(r.weight > 1e-9)
                elif r.weight > 1e-9:
                    adm_pos += 1
                else:
                    adm_zero += 1
            w = st.weights_
            tree_mass = sum(v for k, v in w.items() if k in TREES)
            member_rmse = {k: rmse(y_test.values, v) for k, v in st.base_forecasts_test_.items()}
            blend_rmse = rmse(y_test.values, pred)
            rows.append({
                "dataset": dataset, "pool": pool, "K": len(w),
                "failed": ";".join(st.failed_),
                "n_zero": int(sum(v <= 1e-9 for v in w.values())),
                "n_active": int(sum(v > 1e-9 for v in w.values())),
                "max_weight": round(max(w.values()), 4),
                "argmax": max(w, key=w.get),
                "tree_mass": round(tree_mass, 4),
                "val_leader": best,
                "excluded_by_boundary": excl,
                "necessity_violations": viol,
                "admissible_positive": adm_pos,
                "admissible_zeroed": adm_zero,
                "test_rmse_blend": round(blend_rmse, 4),
                "test_rmse_best_active_member": round(min(member_rmse.values()), 4)
                if member_rmse else np.nan,
                "best_active_member": min(member_rmse, key=member_rmse.get)
                if member_rmse else "",
                "fit_time_s": round(fit_time, 1),
            })
            for _, r in frame.iterrows():
                wrows.append({"dataset": dataset, "pool": pool, "member": r.member,
                              "weight": round(float(r.weight), 6),
                              "rmse_ratio_vs_best": round(float(r.sigma_ratio_vs_best), 4),
                              "rho_vs_best": round(float(r.rho_vs_best), 4),
                              "is_best": bool(r.is_best)})
            logger.info("[%s] %-15s K=%2d zero=%2d tree_mass=%.3f blend=%.4f best_member=%.4f (%s) %.0fs",
                        dataset, pool, len(w), rows[-1]["n_zero"], tree_mass, blend_rmse,
                        rows[-1]["test_rmse_best_active_member"], rows[-1]["best_active_member"],
                        fit_time)

    out_dir = args.results / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data, keys in (("membership_sensitivity.csv", rows, ["dataset", "pool"]),
                             ("membership_sensitivity_weights.csv", wrows, ["dataset", "pool", "member"])):
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

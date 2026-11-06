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

"""Refit only the Convex Stacking Ensemble and refresh its artefacts.

The 22 standalone models do not depend on the stacking solver, so after the
solver change only the CSE rows need regenerating. This script refits the
CSE per dataset with the exact simplex solver and rewrites, in place:

* its rows in ``results/leaderboards.csv`` (both protocols),
* ``results/predictions/{dataset}__{protocol}__FAHE_(Ours).parquet``,
* ``results/cse_diagnostics/{dataset}.pkl`` and
  ``results/cse_weights_by_dataset.csv``,
* its rows in ``results/multistep_rolling/{dataset}__rolling__*.csv`` and
  ``long_all__rolling.csv`` (24 rolling origins, horizons 1 to 24),
* ``results/tables/cse_refit_times.csv`` (time saved by skipping zero
  weight members in the second fit).

Afterwards run ``python scripts/run_significance_analysis.py --in results/``
to rebuild the Diebold-Mariano and Friedman files from the predictions.

Run::

    python scripts/rerun_cse.py                       # all five datasets
    python scripts/rerun_cse.py --datasets ETTh1 ETTh2
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data.preprocessing import train_test_split_temporal  # noqa: E402
from src.evaluation.horizon import (  # noqa: E402
    choose_rolling_origins, rolling_origin_horizon_metrics,
)
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.features.engineering import build_supervised_frame  # noqa: E402
from src.models.novel import FAHEForecaster  # noqa: E402
from src.pipeline import load_project_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rerun_cse")

MODEL_NAME = "FAHE (Ours)"          # name used in every artefact
SAFE_NAME = "FAHE_(Ours)"
FAMILY = "novel-hybrid"
HORIZONS = [1, 3, 6, 12, 24]
ROLL_METRICS = ["RMSE", "MAE", "R2", "MASE"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-origins", type=int, default=24)
    p.add_argument("--skip-rolling", action="store_true")
    return p.parse_args()


def _replace_rows(path: Path, new: pd.DataFrame, key_cols: list[str]) -> None:
    """Replace rows of ``path`` that match ``new`` on ``key_cols``; append the rest."""
    if path.exists():
        old = pd.read_csv(path)
        keys = new[key_cols].drop_duplicates()
        merged = old.merge(keys, on=key_cols, how="left", indicator=True)
        old = old[merged["_merge"].values == "left_only"]
        out = pd.concat([old, new], ignore_index=True)[old.columns.tolist()]
    else:
        out = new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _replace_rolling_wide(path: Path, dataset: str, per_h: pd.DataFrame, metric: str) -> None:
    # The rolling-origin script writes the model column in lower case.
    row = {"model": MODEL_NAME, "Dataset": dataset}
    for _, r in per_h.iterrows():
        row[f"h={int(r['horizon'])}"] = float(r[metric])
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        col = "model" if "model" in old.columns else "Model"
        old = old[old[col] != MODEL_NAME]
        new = new.rename(columns={"model": col})
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)


def run_dataset(dataset: str, args: argparse.Namespace) -> dict:
    res = args.results
    df, target, meta = load_project_dataset(dataset)
    season = int(meta.get("season", 24))
    feature_cfg = config.FeatureConfig()
    X, y, _ = build_supervised_frame(df, target, cfg=feature_cfg)
    df_target = df[[target]].loc[X.index]
    split_X = train_test_split_temporal(X, test_size=args.test_size)
    split_y = train_test_split_temporal(df_target, test_size=args.test_size)
    X_train, X_test = split_X["train"], split_X["test"]
    y_train, y_test = split_y["train"][target], split_y["test"][target]
    logger.info("[%s] rows=%d train=%d test=%d", dataset, len(X), len(X_train), len(X_test))

    # ---- fit once ---------------------------------------------------------
    cse = FAHEForecaster(seasonal_periods=season)
    t0 = time.perf_counter()
    cse.fit(y_train, X_train=X_train)
    fit_time = time.perf_counter() - t0
    logger.info("[%s] weights=%s  fit=%.1fs  skipped refits=%s",
                dataset, {k: round(v, 4) for k, v in cse.weights_.items()},
                fit_time, cse.skipped_refits_)

    # ---- diagnostics ------------------------------------------------------
    diag = cse.diagnostics(dataset=dataset)
    diag_path = res / "cse_diagnostics" / f"{dataset}.pkl"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "wb") as fh:
        pickle.dump(diag, fh)
    wdf = diag.to_frame()[["dataset", "member", "weight", "sigma", "rho_vs_best",
                           "is_best", "kkt_residual"]]
    _replace_rows(res / "cse_weights_by_dataset.csv", wdf, ["dataset", "member"])

    # ---- both protocols on the fixed split --------------------------------
    pred_dir = res / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    lb_rows = []
    for protocol in ("one_step", "recursive"):
        t1 = time.perf_counter()
        if protocol == "one_step":
            preds = cse.predict(len(y_test), X_test=X_test)
        else:
            preds = cse.predict_recursive(
                X_test, y_train.values, target=target, feature_cfg=feature_cfg,
            )
        pt = time.perf_counter() - t1
        preds = np.asarray(preds, dtype=float).ravel()
        if not np.all(np.isfinite(preds)):
            raise RuntimeError(f"{dataset}/{protocol}: non-finite CSE predictions")
        m = compute_all_metrics(y_test.values, preds, y_train=y_train.values,
                                seasonality=season)
        pd.Series(preds, index=y_test.index, name="y_pred").to_frame().to_parquet(
            pred_dir / f"{dataset}__{protocol}__{SAFE_NAME}.parquet")
        lb_rows.append({
            "Dataset": dataset, "Protocol": protocol, "Model": MODEL_NAME,
            "Family": FAMILY, "TrainTime(s)": round(fit_time, 3),
            "PredictTime(s)": round(pt, 3),
            **{k: round(float(v), 4) for k, v in m.items()},
        })
        logger.info("[%s] %s RMSE=%.4f", dataset, protocol, m["RMSE"])
    _replace_rows(res / "leaderboards.csv", pd.DataFrame(lb_rows),
                  ["Dataset", "Protocol", "Model"])

    # ---- rolling origins --------------------------------------------------
    if not args.skip_rolling:
        origins = choose_rolling_origins(len(y_test), max_horizon=max(HORIZONS),
                                         n_origins=args.n_origins)
        y_tr = y_train.to_numpy(dtype=float)
        y_te = y_test.to_numpy(dtype=float)
        per_origin = []
        for origin in origins:
            history = np.concatenate([y_tr, y_te[:origin]])
            h_eff = min(max(HORIZONS), len(y_te) - origin)
            X_win = X_test.iloc[origin: origin + h_eff]
            fc = cse.predict_recursive(X_win, history, target=target,
                                       feature_cfg=feature_cfg)
            per_origin.append((origin, np.asarray(fc, dtype=float).ravel()))
        per_h = rolling_origin_horizon_metrics(
            y_te, per_origin, horizons=HORIZONS, y_train=y_tr, seasonality=season,
        )
        roll_dir = res / "multistep_rolling"
        roll_dir.mkdir(parents=True, exist_ok=True)
        for metric in ROLL_METRICS:
            if metric in per_h.columns:
                _replace_rolling_wide(roll_dir / f"{dataset}__rolling__{metric}.csv",
                                      dataset, per_h, metric)
        long_rows = pd.DataFrame([{
            "Dataset": dataset, "Model": MODEL_NAME,
            "horizon": int(r["horizon"]), "n_pairs": int(r["n_pairs"]),
            **{mm: float(r[mm]) for mm in ROLL_METRICS if mm in per_h.columns},
        } for _, r in per_h.iterrows()])
        _replace_rows(roll_dir / f"{dataset}__rolling__long.csv", long_rows,
                      ["Dataset", "Model", "horizon"])
        parts = [pd.read_csv(p) for p in sorted(roll_dir.glob("*__rolling__long.csv"))
                 if p.name != "long_all__rolling.csv"]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(
                roll_dir / "long_all__rolling.csv", index=False)
        logger.info("[%s] rolling-origin h=1 RMSE=%.4f (origins=%d)",
                    dataset, float(per_h.loc[per_h.horizon == 1, "RMSE"].iloc[0]),
                    len(per_origin))

    return {
        "dataset": dataset, "fit_time_s": round(fit_time, 2),
        "skipped_refits": ";".join(cse.skipped_refits_),
        "refit_times_s": ";".join(f"{k}={v:.1f}" for k, v in cse.refit_times_.items()),
        **{f"w_{k}": round(v, 6) for k, v in cse.weights_.items()},
    }


def main() -> None:
    args = parse_args()
    datasets = args.datasets or list(config.PROJECT_DATASETS)
    summary = []
    for ds in datasets:
        summary.append(run_dataset(ds, args))
    out = args.results / "tables" / "cse_refit_times.csv"
    _replace_rows(out, pd.DataFrame(summary), ["dataset"])
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nNext: python scripts/run_significance_analysis.py --in", args.results)


if __name__ == "__main__":
    main()

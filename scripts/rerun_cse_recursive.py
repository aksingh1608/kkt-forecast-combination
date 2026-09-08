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

"""Re-score CSE / FAHE under the true recursive protocol only.

Use after the ``predict_recursive`` fix so you do **not** need to re-run the
full 23-model zoo. Updates:

* ``results/predictions/{dataset}__recursive__FAHE_(Ours).parquet``
* CSE recursive rows in ``leaderboards.csv`` / ``leaderboards_partial.csv``

Run::

    python scripts/rerun_cse_recursive.py --out results/
    python scripts/rerun_cse_recursive.py --out results/ --datasets UCI ETTh1
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

from src import config
from src.data.preprocessing import train_test_split_temporal
from src.evaluation.metrics import compute_all_metrics
from src.features.engineering import build_supervised_frame
from src.models.novel import FAHEForecaster
from src.pipeline import load_project_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rerun_cse_recursive")

MODEL_NAME = "FAHE (Ours)"
SAFE_NAME = "FAHE_(Ours)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Unused for CSE fit; kept for CLI symmetry.")
    return p.parse_args()


def _update_leaderboard(out_dir: Path, dataset: str, metrics: dict, times: dict) -> None:
    path = out_dir / "leaderboards.csv"
    partial = out_dir / "leaderboards_partial.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    mask = (
        (df["Dataset"] == dataset)
        & (df["Protocol"] == "recursive")
        & (df["Model"] == MODEL_NAME)
    )
    row = {
        "Dataset": dataset,
        "Protocol": "recursive",
        "Model": MODEL_NAME,
        "Family": "novel-hybrid",
        "TrainTime(s)": times["train"],
        "PredictTime(s)": times["predict"],
        **{k: float(metrics[k]) for k in ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE")},
    }
    if mask.any():
        for k, v in row.items():
            if k in df.columns:
                df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    df.to_csv(partial, index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.out
    datasets = args.datasets or list(config.PROJECT_DATASETS)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        logger.info("[%s] loading…", dataset)
        df, target, meta = load_project_dataset(dataset)
        season = int(meta.get("season", 24))
        feature_cfg = config.FeatureConfig()
        X, y, _ = build_supervised_frame(df, target, cfg=feature_cfg)
        df_target = df[[target]].loc[X.index]
        split_X = train_test_split_temporal(X, test_size=args.test_size)
        split_y = train_test_split_temporal(df_target, test_size=args.test_size)
        X_train, X_test = split_X["train"], split_X["test"]
        y_train = split_y["train"][target]
        y_test = split_y["test"][target]

        cse = FAHEForecaster(seasonal_periods=season)
        import time
        t0 = time.perf_counter()
        cse.fit(y_train, X_train=X_train)
        t1 = time.perf_counter()
        preds = cse.predict_recursive(
            X_test, y_train.values, target=target, feature_cfg=feature_cfg
        )
        t2 = time.perf_counter()

        one_step = cse.predict(len(y_test), X_test=X_test)
        mean_abs_delta = float(np.mean(np.abs(preds - one_step)))
        logger.info(
            "[%s] recursive vs one-step mean|Δ|=%.6f  weights=%s",
            dataset, mean_abs_delta, cse.weights_,
        )
        if mean_abs_delta < 1e-8:
            logger.warning(
                "[%s] recursive predictions still match one-step — check tree weights.",
                dataset,
            )

        metrics = compute_all_metrics(
            y_test.values, preds, y_train=y_train.values, seasonality=season
        )
        metrics = {k: round(float(v), 4) if np.isfinite(v) else v for k, v in metrics.items()}
        times = {
            "train": round(t1 - t0, 3),
            "predict": round(t2 - t1, 3),
        }
        logger.info("[%s] recursive CSE RMSE=%.4f MAE=%.4f", dataset,
                    metrics["RMSE"], metrics["MAE"])

        pred_s = pd.Series(preds, index=y_test.index, name="y_pred")
        pred_s.to_frame().to_parquet(pred_dir / f"{dataset}__recursive__{SAFE_NAME}.parquet")
        _update_leaderboard(out_dir, dataset, metrics, times)
        logger.info("[%s] leaderboard + predictions updated.", dataset)

    print("\nDone. Re-run multistep + significance if you need those tables refreshed:")
    print("  python scripts/run_multistep_benchmark.py --in", out_dir)
    print("  python scripts/run_significance_analysis.py --in", out_dir)


if __name__ == "__main__":
    main()

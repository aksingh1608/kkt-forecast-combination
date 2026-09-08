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

"""Tune and re-score NLinear / DLinear only.

This avoids re-running the full benchmark after fixing the lite-DL baselines.
It updates:

* ``results/predictions/{dataset}__{protocol}__NLinear.parquet``
* ``results/predictions/{dataset}__{protocol}__DLinear.parquet``
* matching rows in ``leaderboards.csv`` and ``leaderboards_partial.csv``
* tuning histories under ``results/tuning/lite_dl/``

Run::

    python scripts/rerun_lite_dl_tuned.py --out results/ --trials 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data.preprocessing import train_test_split_temporal
from src.evaluation.metrics import compute_all_metrics
from src.features.engineering import build_supervised_frame
from src.models.lite_dl import DLinearForecaster, NLinearForecaster
from src.pipeline import load_project_dataset
from src.tuning import tune_lite_linear_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rerun_lite_dl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=["NLinear", "DLinear"])
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _model_from_params(name: str, params: dict):
    if name == "NLinear":
        return NLinearForecaster(
            lookback=int(params["lookback"]),
            ridge=float(params.get("ridge", 1e-2)),
            clip_quantile=float(params.get("clip_quantile", 0.995)),
        )
    if name == "DLinear":
        return DLinearForecaster(
            lookback=int(params["lookback"]),
            kernel_size=int(params.get("kernel_size", 25)),
            ridge_trend=float(params.get("ridge_trend", 1e-2)),
            ridge_seasonal=float(params.get("ridge_seasonal", 1.0)),
            use_revin=bool(params.get("use_revin", True)),
            clip_quantile=float(params.get("clip_quantile", 0.995)),
        )
    raise ValueError(f"Unsupported lite-DL model: {name}")


def _update_leaderboards(out_dir: Path, dataset: str, model: str, metrics: dict, times: dict) -> None:
    row_values = {
        "Family": "lite-dl",
        "TrainTime(s)": round(float(times["train"]), 3),
        "PredictTime(s)": round(float(times["predict"]), 3),
        **{k: round(float(metrics[k]), 4) for k in ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE")},
    }
    for name in ("leaderboards.csv", "leaderboards_partial.csv"):
        path = out_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for protocol in ("one_step", "recursive"):
            mask = (
                (df["Dataset"] == dataset)
                & (df["Protocol"] == protocol)
                & (df["Model"] == model)
            )
            if not mask.any():
                new_row = {
                    "Dataset": dataset,
                    "Protocol": protocol,
                    "Model": model,
                    **row_values,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                continue
            for col, value in row_values.items():
                df.loc[mask, col] = value
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.out
    pred_dir = out_dir / "predictions"
    tuning_dir = out_dir / "tuning" / "lite_dl"
    pred_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(config.PROJECT_DATASETS)
    wanted_models = [m for m in args.models if m in {"NLinear", "DLinear"}]
    summary_rows = []

    for dataset in datasets:
        logger.info("========== %s ==========", dataset)
        df, target, meta = load_project_dataset(dataset)
        season = int(meta.get("season", 24))
        X, y, _ = build_supervised_frame(df, target, cfg=config.FeatureConfig())
        df_target = df[[target]].loc[X.index]
        split_y = train_test_split_temporal(df_target, test_size=args.test_size)
        y_train = split_y["train"][target]
        y_test = split_y["test"][target]

        # Ensure y_true parquet exists for both protocols.
        for protocol in ("one_step", "recursive"):
            y_test.to_frame(name="y_true").to_parquet(
                pred_dir / f"{dataset}__{protocol}__y_true.parquet"
            )

        for model_name in wanted_models:
            logger.info("[%s] tuning %s (%d trials)", dataset, model_name, args.trials)
            tune = tune_lite_linear_model(
                y_train,
                model_name=model_name.lower(),
                n_trials=args.trials,
                val_frac=args.val_frac,
                seed=args.seed,
            )
            tune.history.to_csv(tuning_dir / f"{dataset}__{model_name}__history.csv", index=False)
            (tuning_dir / f"{dataset}__{model_name}__best.json").write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "best_value": tune.best_value,
                        "best_params": tune.best_params,
                    },
                    indent=2,
                )
            )

            model = _model_from_params(model_name, tune.best_params)
            t0 = time.perf_counter()
            model.fit(y_train)
            t1 = time.perf_counter()
            preds = model.predict(len(y_test))
            t2 = time.perf_counter()
            metrics = compute_all_metrics(
                y_test.values, preds, y_train=y_train.values, seasonality=season
            )
            logger.info(
                "[%s] %s tuned RMSE=%.4f MAE=%.4f params=%s",
                dataset, model_name, metrics["RMSE"], metrics["MAE"], tune.best_params,
            )

            pred_s = pd.Series(preds, index=y_test.index, name="y_pred")
            for protocol in ("one_step", "recursive"):
                pred_s.to_frame().to_parquet(
                    pred_dir / f"{dataset}__{protocol}__{model_name}.parquet"
                )
            _update_leaderboards(
                out_dir, dataset, model_name, metrics,
                {"train": t1 - t0, "predict": t2 - t1},
            )
            summary_rows.append({
                "Dataset": dataset,
                "Model": model_name,
                "validation_RMSE": tune.best_value,
                **{f"test_{k}": float(metrics[k]) for k in ("MAE", "RMSE", "R2", "MASE")},
                **tune.best_params,
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(tuning_dir / "summary.csv", index=False)
        logger.info("Wrote %s", tuning_dir / "summary.csv")

    print("\nNext refresh dependent artefacts:")
    print("  python scripts/run_multistep_benchmark.py --in", out_dir)
    print("  python scripts/run_rolling_origin_benchmark.py --out", out_dir / "multistep_rolling", "--n-origins 24 --n-jobs 1")
    print("  python scripts/run_significance_analysis.py --in", out_dir)
    print("  python scripts/plot_error_drift.py --in", out_dir, "--protocol both")


if __name__ == "__main__":
    main()

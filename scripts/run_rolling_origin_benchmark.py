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

"""Rolling-origin per-horizon leaderboard (paper-grade h-step metrics).

Metrics are not taken from a prefix of one rollout, where h=1 would be a
single point. Instead, this script:

1. Fits each model once on the chronological train split.
2. Places ``n_origins`` evenly spaced forecast origins across the test window.
3. At origin *o*, seeds history with **observed** y up to *o*, then rolls
   out ``max(horizons)`` recursive steps.
4. Reports RMSE/MAE/... of the cross-section of horizon-*h* errors averaged
   over all origins.

Run::

    python scripts/run_rolling_origin_benchmark.py --out results/multistep_rolling/
    python scripts/run_rolling_origin_benchmark.py --datasets UCI --n-origins 24 --no-heavy
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
from src.evaluation.horizon import (
    choose_rolling_origins,
    rolling_origin_horizon_metrics,
    rolling_origin_leaderboard,
)
from src.features.engineering import build_supervised_frame
from src.models.lite_dl import DLinearForecaster, NLinearForecaster
from src.models.recursive import recursive_predict
from src.models.registry import ModelSpec, build_registry
from src.pipeline import load_project_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rolling_origin")

# Heavy classical models that re-fit at every origin are too slow / RAM-heavy
# for a full rolling-origin sweep on commodity hardware.
_HEAVY_CLASSICAL = {"ARIMA", "SARIMA", "Holt-Winters", "Theta"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/multistep_rolling"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12, 24])
    p.add_argument("--n-origins", type=int, default=48)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--no-heavy", action="store_true", default=True,
                   help="Skip ARIMA/SARIMA/HW/Theta (default: on).")
    p.add_argument("--with-heavy", action="store_true",
                   help="Include heavy classical (slow; may OOM).")
    p.add_argument("--metrics", nargs="+", default=["RMSE", "MAE", "R2", "MASE"])
    p.add_argument("--models", nargs="*", default=None,
                   help="Optional subset of model names.")
    return p.parse_args()


def _forecast_at_origins(
    model,
    *,
    requires_features: bool,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    origins: list[int],
    max_h: int,
    target: str,
    feature_cfg: config.FeatureConfig,
) -> list[tuple[int, np.ndarray]]:
    """Fit once, then issue recursive (or classical) forecasts at each origin."""
    if requires_features:
        model.fit(y_train, X_train=X_train)
    else:
        model.fit(y_train)

    y_train_arr = y_train.to_numpy(dtype=float)
    y_test_arr = y_test.to_numpy(dtype=float)
    out: list[tuple[int, np.ndarray]] = []

    for origin in origins:
        history = np.concatenate([y_train_arr, y_test_arr[:origin]])
        h_eff = min(max_h, len(y_test_arr) - origin)
        if h_eff < 1:
            continue
        try:
            if requires_features:
                X_win = X_test.iloc[origin: origin + h_eff]
                name = getattr(model, "name", "")
                if name.lower().startswith(("fahe", "cse")):
                    fc = model.predict_recursive(
                        X_win, history, target=target, feature_cfg=feature_cfg
                    )
                else:
                    fc = recursive_predict(
                        model, X_win, history,
                        target=target, feature_cfg=feature_cfg,
                    )
            else:
                # Classical: re-fit on observed history up to the origin so the
                # multi-step forecast is a true origin-conditioned rollout.
                y_hist = pd.Series(history, index=pd.RangeIndex(len(history)))
                model.fit(y_hist)
                fc = np.asarray(model.predict(h_eff), dtype=float).ravel()
        except Exception as exc:
            logger.warning("origin=%d model=%s failed: %s",
                           origin, getattr(model, "name", "?"), exc)
            continue
        out.append((origin, np.asarray(fc, dtype=float).ravel()))
    return out


def _apply_tuned_lite_dl_specs(specs: list[ModelSpec], dataset: str, results_root: Path) -> list[ModelSpec]:
    """Replace NLinear/DLinear factories with tuned params if present."""
    tuned_dir = results_root / "tuning" / "lite_dl"
    out: list[ModelSpec] = []
    for spec in specs:
        if spec.name not in {"NLinear", "DLinear"}:
            out.append(spec)
            continue
        path = tuned_dir / f"{dataset}__{spec.name}__best.json"
        if not path.exists():
            out.append(spec)
            continue
        try:
            payload = json.loads(path.read_text())
            params = payload["best_params"]
            if spec.name == "NLinear":
                factory = lambda p=params: NLinearForecaster(
                    lookback=int(p["lookback"]),
                    ridge=float(p.get("ridge", 1e-2)),
                    clip_quantile=float(p.get("clip_quantile", 0.995)),
                )
            else:
                factory = lambda p=params: DLinearForecaster(
                    lookback=int(p["lookback"]),
                    kernel_size=int(p.get("kernel_size", 25)),
                    ridge_trend=float(p.get("ridge_trend", 1e-2)),
                    ridge_seasonal=float(p.get("ridge_seasonal", 1.0)),
                    use_revin=bool(p.get("use_revin", True)),
                    clip_quantile=float(p.get("clip_quantile", 0.995)),
                )
            logger.info("[%s] using tuned %s params from %s", dataset, spec.name, path)
            out.append(ModelSpec(spec.name, spec.family, factory, spec.requires_features))
        except Exception as exc:
            logger.warning("[%s] failed to load tuned %s params: %s", dataset, spec.name, exc)
            out.append(spec)
    return out


def main() -> None:
    args = parse_args()
    skip_heavy = args.no_heavy and not args.with_heavy
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or list(config.PROJECT_DATASETS)
    horizons = list(args.horizons)
    max_h = max(horizons)
    feature_cfg = config.FeatureConfig()

    summary_rows: list[pd.DataFrame] = []

    for dataset in datasets:
        logger.info("========== %s ==========", dataset)
        df, target, meta = load_project_dataset(dataset)
        season = int(meta.get("season", 24))
        X, y, _ = build_supervised_frame(df, target, cfg=feature_cfg)
        df_target = df[[target]].loc[X.index]
        split_X = train_test_split_temporal(X, test_size=args.test_size)
        split_y = train_test_split_temporal(df_target, test_size=args.test_size)
        X_train, X_test = split_X["train"], split_X["test"]
        y_train = split_y["train"][target]
        y_test = split_y["test"][target]

        origins = choose_rolling_origins(
            len(y_test), max_horizon=max_h, n_origins=args.n_origins
        )
        logger.info("[%s] n_test=%d  n_origins=%d  max_h=%d",
                    dataset, len(y_test), len(origins), max_h)
        if not origins:
            logger.warning("[%s] test window too short — skipping.", dataset)
            continue

        # Memory guard: never fit standalone heavy classical on national load.
        include_heavy = (not skip_heavy) and dataset != "OPSD_DE_total"
        specs = build_registry(
            seasonal_period=season,
            n_jobs=args.n_jobs,
            include_heavy_classical=include_heavy,
        )
        results_root = out_dir.parent if out_dir.name == "multistep_rolling" else Path("results")
        specs = _apply_tuned_lite_dl_specs(specs, dataset, results_root)
        if skip_heavy:
            specs = [s for s in specs if s.name not in _HEAVY_CLASSICAL]
        if args.models:
            wanted = set(args.models)
            specs = [s for s in specs if s.name in wanted]

        forecasts_by_model: dict[str, list[tuple[int, np.ndarray]]] = {}
        long_metric_rows: list[dict] = []

        for spec in specs:
            t0 = time.perf_counter()
            logger.info("[%s] rolling-origin: %s …", dataset, spec.name)
            model = spec.factory()
            per_origin = _forecast_at_origins(
                model,
                requires_features=spec.requires_features,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                origins=origins, max_h=max_h,
                target=target, feature_cfg=feature_cfg,
            )
            if not per_origin:
                logger.warning("[%s] %s produced no origin forecasts.", dataset, spec.name)
                continue
            forecasts_by_model[spec.name] = per_origin
            per_h = rolling_origin_horizon_metrics(
                y_test.values, per_origin, horizons=horizons,
                y_train=y_train.values, seasonality=season,
            )
            for _, row in per_h.iterrows():
                long_metric_rows.append({
                    "Dataset": dataset,
                    "Model": spec.name,
                    "horizon": int(row["horizon"]),
                    "n_pairs": int(row["n_pairs"]),
                    **{m: float(row[m]) for m in args.metrics if m in row},
                })
            logger.info("[%s] %s done in %.1fs  (origins used=%d)",
                        dataset, spec.name, time.perf_counter() - t0, len(per_origin))

        if not forecasts_by_model:
            continue

        for metric in args.metrics:
            table = rolling_origin_leaderboard(
                y_test.values, forecasts_by_model,
                horizons=horizons, metric=metric,
                y_train=y_train.values, seasonality=season,
            )
            table.insert(0, "Dataset", dataset)
            path = out_dir / f"{dataset}__rolling__{metric}.csv"
            table.to_csv(path)
            logger.info("Wrote %s", path)

        if long_metric_rows:
            long_df = pd.DataFrame(long_metric_rows)
            long_df.to_csv(out_dir / f"{dataset}__rolling__long.csv", index=False)
            summary_rows.append(long_df)

            # Sanity: h=1 must use many pairs, not 1.
            h1 = long_df[long_df["horizon"] == 1]
            if not h1.empty:
                logger.info("[%s] h=1 n_pairs min/median/max = %d / %.0f / %d",
                            dataset,
                            int(h1["n_pairs"].min()),
                            float(h1["n_pairs"].median()),
                            int(h1["n_pairs"].max()))

    # Rebuild the combined file from every per-dataset long table on disk,
    # not only from datasets processed in this invocation. This makes
    # one-dataset resume runs safe after a long rolling-origin sweep stops
    # mid-way.
    all_long = []
    for path in sorted(out_dir.glob("*__rolling__long.csv")):
        if path.name == "long_all__rolling.csv":
            continue
        try:
            all_long.append(pd.read_csv(path))
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
    if all_long:
        combined = pd.concat(all_long, ignore_index=True)
        combined.to_csv(out_dir / "long_all__rolling.csv", index=False)
        logger.info("Wrote %s", out_dir / "long_all__rolling.csv")


if __name__ == "__main__":
    main()

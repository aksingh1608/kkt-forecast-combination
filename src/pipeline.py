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

"""End-to-end experiment orchestrator.

This module is the single entry point used by:

* the command-line experiment runner (``scripts/run_full_experiment.py``)
* the Streamlit frontend (``app/streamlit_app.py``)

Given a cleaned, resampled DataFrame and a target column, it:

1. Builds an engineered supervised matrix.
2. Splits chronologically into train/test.
3. Trains every model in the registry, recording wall-clock time.
4. Computes a full metric suite on the held-out test set.
5. Returns predictions, a leaderboard DataFrame and per-model diagnostics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config
from .data.preprocessing import train_test_split_temporal
from .evaluation.metrics import compute_all_metrics
from .features.engineering import build_supervised_frame
from .models import ModelSpec, build_registry
from .models.recursive import recursive_predict

logger = logging.getLogger(__name__)


@dataclass
class ModelRun:
    """Per-model artefact produced by the pipeline."""

    name: str
    family: str
    metrics: dict[str, float]
    predictions: np.ndarray
    train_time_s: float
    predict_time_s: float
    error: str | None = None


@dataclass
class ExperimentResult:
    """Aggregated output of :func:`run_experiment`."""

    target: str
    seasonal_period: int
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    y_train: pd.Series
    y_test: pd.Series
    runs: list[ModelRun] = field(default_factory=list)

    def leaderboard(self, sort_by: str = "RMSE") -> pd.DataFrame:
        """Pivot the ``runs`` list into a sortable leaderboard DataFrame."""
        rows = []
        for r in self.runs:
            row = {"Model": r.name, "Family": r.family,
                   "TrainTime(s)": round(r.train_time_s, 3),
                   "PredictTime(s)": round(r.predict_time_s, 3)}
            row.update({k: round(v, 4) for k, v in r.metrics.items()})
            if r.error:
                row["error"] = r.error
            rows.append(row)
        df = pd.DataFrame(rows)
        if sort_by in df.columns:
            ascending = sort_by != "R2"
            df = df.sort_values(sort_by, ascending=ascending, na_position="last")
        return df.reset_index(drop=True)

    def best_model(self, metric: str = "RMSE") -> ModelRun | None:
        valid = [r for r in self.runs if r.error is None and metric in r.metrics
                 and np.isfinite(r.metrics[metric])]
        if not valid:
            return None
        ascending = metric != "R2"
        return min(valid, key=lambda r: r.metrics[metric]) if ascending \
            else max(valid, key=lambda r: r.metrics[metric])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_experiment(
    df: pd.DataFrame,
    target: str,
    *,
    seasonal_period: int | None = None,
    test_size: float = 0.2,
    feature_cfg: config.FeatureConfig | None = None,
    model_specs: list[ModelSpec] | None = None,
    include_heavy_classical: bool = True,
    include_optional_dl: bool = False,
    optional_dl_horizon: int | None = None,
    optional_dl_max_steps: int = 200,
    optional_dl_input_size: int | None = None,
    n_jobs: int = -1,
    progress_callback: Callable[[str, float], None] | None = None,
    protocol: str = "one_step",
) -> ExperimentResult:
    """Run the full pipeline.

    Parameters
    ----------
    df
        Cleaned, resampled DataFrame indexed by datetime.
    target
        Column to forecast.
    seasonal_period
        Length of the dominant seasonal cycle. Inferred from the index
        frequency when ``None``.
    test_size
        Fraction of the series held out as the test set.
    feature_cfg
        Override for :class:`~src.config.FeatureConfig`.
    model_specs
        Restrict to a subset of models. ``None`` → every model in the registry.
    progress_callback
        Optional ``(message, pct)`` callback. Used by the Streamlit UI to
        render a progress bar; safely ignored from scripts.
    protocol
        Evaluation protocol for supervised (feature-consuming) learners.

        * ``"one_step"`` — teacher-forced: at each test step the model
          receives the true lagged features (paper's Eq. 1).
        * ``"recursive"`` — symmetric recursive multi-step: lag / rolling
          features at step *h* are recomputed from the model's own past
          predictions (paper's Eq. 2 and §V-C). Classical models are
          recursive by definition regardless of this setting.
    """
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in DataFrame columns.")
    if protocol not in ("one_step", "recursive"):
        raise ValueError(f"protocol must be 'one_step' or 'recursive', got {protocol!r}")
    feature_cfg = feature_cfg or config.FeatureConfig()

    if seasonal_period is None:
        from .data.preprocessing import _detect_seasonal_period

        seasonal_period = _detect_seasonal_period(df.index) or 24

    notify = progress_callback or (lambda msg, pct: None)
    notify("Building feature matrix…", 0.05)

    X, y, _feat_names = build_supervised_frame(df, target, cfg=feature_cfg)
    df_target = df[[target]].loc[X.index]
    split_X = train_test_split_temporal(X, test_size=test_size)
    split_y = train_test_split_temporal(df_target, test_size=test_size)

    X_train, X_test = split_X["train"], split_X["test"]
    y_train = split_y["train"][target]
    y_test = split_y["test"][target]

    # Default the rolling-chunk horizon for DL models to the seasonal
    # period (24h on hourly data, 7d on daily data) — never the full test
    # length, which would force absurd input_size values and OOM.
    dl_horizon = optional_dl_horizon or min(seasonal_period, max(24, len(y_test) // 8))
    specs = model_specs or build_registry(
        seasonal_period=seasonal_period,
        n_jobs=n_jobs,
        include_heavy_classical=include_heavy_classical,
        include_optional_dl=include_optional_dl,
        optional_dl_horizon=dl_horizon,
        optional_dl_max_steps=optional_dl_max_steps,
        optional_dl_input_size=optional_dl_input_size,
    )

    result = ExperimentResult(
        target=target,
        seasonal_period=seasonal_period,
        train_index=y_train.index,
        test_index=y_test.index,
        y_train=y_train,
        y_test=y_test,
    )

    for i, spec in enumerate(specs):
        pct_done = 0.10 + 0.85 * (i / max(1, len(specs)))
        notify(f"Training {spec.name}…", pct_done)
        try:
            run = _train_and_eval(
                spec, X_train, y_train, X_test, y_test, seasonal_period,
                protocol=protocol, target=target, feature_cfg=feature_cfg,
            )
        except Exception as exc:  # capture and continue so one bad model can't kill the run
            logger.exception("Model %s failed: %s", spec.name, exc)
            run = ModelRun(
                name=spec.name,
                family=spec.family,
                metrics={},
                predictions=np.full(len(y_test), np.nan),
                train_time_s=0.0,
                predict_time_s=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        result.runs.append(run)

    notify("Done.", 1.0)
    return result


# ---------------------------------------------------------------------------
# Multi-dataset paper runner
# ---------------------------------------------------------------------------
@dataclass
class MultiDatasetResult:
    """Container for a paper-scale experiment over multiple datasets."""

    results_by_dataset: dict[str, ExperimentResult] = field(default_factory=dict)

    def leaderboard(self, sort_by: str = "RMSE") -> pd.DataFrame:
        """Long-format DataFrame: rows = (dataset, model), columns = metrics."""
        frames = []
        for ds, res in self.results_by_dataset.items():
            lb = res.leaderboard(sort_by=sort_by)
            lb.insert(0, "Dataset", ds)
            frames.append(lb)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True)

    def rank_matrix(self, metric: str = "RMSE") -> pd.DataFrame:
        """(datasets × models) rank matrix for Friedman / CD diagram."""
        long = self.leaderboard(sort_by=metric)
        if long.empty:
            return pd.DataFrame()
        pivot = long.pivot_table(
            index="Dataset", columns="Model", values=metric, aggfunc="first"
        )
        ascending = metric != "R2"
        return pivot.rank(axis=1, ascending=ascending, method="average")


def load_project_dataset(name: str, freq: str = "h") -> tuple[pd.DataFrame, str, dict]:
    """Load any of the five PROJECT_DATASETS by name and resample uniformly.

    Returns ``(df_hourly, target_column, meta)``. The UCI dataset is
    subset to the paper's twelve-month window; the four external datasets
    are used in full at their native hourly resolution.
    """
    from .data import (
        clean_dataset, load_uci_household_power, resample_dataset,
    )
    from .data.extra_loaders import load_external

    if name == "UCI":
        df_raw = load_uci_household_power(use_cache=True)
        df = clean_dataset(df_raw)
        df = resample_dataset(df, freq=freq)
        # Twelve-month window used by the paper (16 Dec 2006 → 16 Dec 2007).
        start = df.index.min()
        window_end = start + pd.Timedelta(days=365)
        df_win = df.loc[start:window_end]
        return df_win, config.TARGET_COLUMN, {
            "name": "UCI", "freq": freq, "season": 24,
            "n": len(df_win),
        }
    if name in config.EXTERNAL_DATASETS:
        df, target, meta = load_external(name)
        # These are already hourly; still pass through the cleaner for NaNs.
        df = clean_dataset(df)
        return df, target, {**meta, "name": name}
    raise KeyError(f"Unknown dataset {name!r}. Known: {config.PROJECT_DATASETS}")


def run_multi_dataset(
    datasets: Sequence[str] | None = None,
    *,
    protocol: str = "one_step",
    freq: str = "h",
    test_size: float = 0.2,
    include_heavy_classical: bool = True,
    include_optional_dl: bool = False,
    optional_dl_horizon: int | None = None,
    optional_dl_max_steps: int = 200,
    optional_dl_input_size: int | None = None,
    n_jobs: int = -1,
    progress_callback: Callable[[str, float], None] | None = None,
) -> MultiDatasetResult:
    """Run the full benchmark on every requested dataset under one protocol.

    Parameters
    ----------
    datasets
        Names of the datasets to run. Defaults to
        :data:`src.config.PROJECT_DATASETS` — the five paper datasets.
    protocol
        ``"one_step"`` or ``"recursive"``. See :func:`run_experiment`.
    """
    names = list(datasets) if datasets is not None else list(config.PROJECT_DATASETS)
    out = MultiDatasetResult()
    n_ds = len(names)
    for i, name in enumerate(names):
        pct = i / max(1, n_ds)
        if progress_callback:
            progress_callback(f"[{name}] loading…", pct)
        df, target, meta = load_project_dataset(name, freq=freq)
        season = int(meta.get("season", config.FREQ_TO_SEASON.get(freq, 24)))
        logger.info("[%s] rows=%d, target=%s, season=%d", name, len(df), target, season)

        result = run_experiment(
            df, target=target,
            seasonal_period=season,
            test_size=test_size,
            include_heavy_classical=include_heavy_classical,
            include_optional_dl=include_optional_dl,
            optional_dl_horizon=optional_dl_horizon,
            optional_dl_max_steps=optional_dl_max_steps,
            optional_dl_input_size=optional_dl_input_size,
            n_jobs=n_jobs,
            protocol=protocol,
            progress_callback=(
                lambda msg, sub_pct, _pct=pct, _n=n_ds, _name=name:
                progress_callback(f"[{_name}] {msg}", _pct + sub_pct / _n)
            ) if progress_callback else None,
        )
        out.results_by_dataset[name] = result
    if progress_callback:
        progress_callback("Done.", 1.0)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _train_and_eval(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seasonal_period: int,
    *,
    protocol: str = "one_step",
    target: str | None = None,
    feature_cfg: config.FeatureConfig | None = None,
) -> ModelRun:
    model = spec.factory()
    t0 = time.perf_counter()
    if spec.requires_features:
        model.fit(y_train, X_train=X_train)
    else:
        model.fit(y_train)
    t1 = time.perf_counter()

    if spec.requires_features:
        # CSE / FAHE: classical branches use open-loop multi-step forecasts;
        # tree branches roll out with lag/rolling features rebuilt from the
        # *ensemble* predictions (see FAHEForecaster.predict_recursive).
        cse_like = spec.name.lower().startswith(("fahe", "cse"))
        if protocol == "recursive" and target is not None and cse_like:
            preds = model.predict_recursive(
                X_test, y_train.values,
                target=target, feature_cfg=feature_cfg,
            )
        elif protocol == "recursive" and target is not None:
            preds = recursive_predict(
                model, X_test, y_train.values,
                target=target, feature_cfg=feature_cfg,
            )
        else:
            preds = model.predict(len(y_test), X_test=X_test)
    else:
        preds = model.predict(len(y_test))
    t2 = time.perf_counter()

    preds = np.asarray(preds, dtype=float).ravel()
    if len(preds) != len(y_test):
        raise RuntimeError(
            f"{spec.name}: returned {len(preds)} predictions for {len(y_test)} targets."
        )
    if not np.all(np.isfinite(preds)):
        raise RuntimeError(
            f"{spec.name}: produced non-finite predictions "
            f"({int(np.sum(~np.isfinite(preds)))}/{len(preds)} bad values)."
        )

    metrics = compute_all_metrics(
        y_test.values, preds, y_train=y_train.values, seasonality=seasonal_period
    )
    return ModelRun(
        name=spec.name,
        family=spec.family,
        metrics=dict(metrics),
        predictions=preds,
        train_time_s=t1 - t0,
        predict_time_s=t2 - t1,
    )

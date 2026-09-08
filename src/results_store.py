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

"""Persistence layer for benchmark runs.

A :class:`BenchmarkSnapshot` is a single self-contained, on-disk record of
one full benchmark execution (one dataset × N models). The store lets the
Streamlit frontend accumulate multiple snapshots over time and feed them
to :mod:`src.comparison` for cross-dataset analysis.

Why a dedicated layer
---------------------
* Snapshots **outlive Streamlit sessions** — close the browser, the runs
  stay on disk under ``results/runs/``.
* The CLI scripts and the UI write to the **same store**, so a heavy
  overnight run from ``run_full_experiment.py`` shows up in the UI the
  next morning.
* Each snapshot is a tiny ``joblib`` pickle (≤ 1–2 MB typically) carrying
  only what the comparison views actually need: dataset signature,
  config, leaderboard and the prediction matrix.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

RUNS_DIR: Path = config.RESULTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkSnapshot:
    """Self-contained record of one benchmark execution."""

    snapshot_id: str
    name: str
    dataset_signature: dict
    config: dict
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    created_at: str
    notes: str = ""

    # ------------------------------------------------------------------
    @property
    def models(self) -> list[str]:
        return [c for c in self.predictions.columns if c != "actual"]

    @property
    def metric_columns(self) -> list[str]:
        return [c for c in self.leaderboard.columns
                if c in ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE")]

    def best(self, metric: str = "RMSE") -> pd.Series | None:
        if metric not in self.leaderboard.columns:
            return None
        ascending = metric != "R2"
        cleaned = self.leaderboard.dropna(subset=[metric])
        if cleaned.empty:
            return None
        idx = cleaned[metric].idxmin() if ascending else cleaned[metric].idxmax()
        return cleaned.loc[idx]

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "name": self.name,
            "dataset_signature": self.dataset_signature,
            "config": self.config,
            "created_at": self.created_at,
            "notes": self.notes,
            "n_models": len(self.models),
            "n_test_points": len(self.predictions),
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return slug[:60] or "snapshot"


def snapshot_from_result(
    result,
    *,
    name: str,
    dataset_signature: dict,
    extra_config: dict | None = None,
    notes: str = "",
) -> BenchmarkSnapshot:
    """Wrap an :class:`~src.pipeline.ExperimentResult` into a snapshot."""
    leaderboard = result.leaderboard(sort_by="RMSE").copy()
    leaderboard = leaderboard.set_index("Model")

    pred_df = pd.DataFrame({"actual": result.y_test.values}, index=result.test_index)
    for run in result.runs:
        if run.error is None:
            pred_df[run.name] = run.predictions
    pred_df.index.name = "datetime"

    cfg = {
        "target": result.target,
        "seasonal_period": result.seasonal_period,
        "n_train": len(result.y_train),
        "n_test": len(result.y_test),
    }
    if extra_config:
        cfg.update(extra_config)

    return BenchmarkSnapshot(
        snapshot_id=uuid.uuid4().hex[:12],
        name=name,
        dataset_signature=dict(dataset_signature),
        config=cfg,
        leaderboard=leaderboard,
        predictions=pred_df,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Store API
# ---------------------------------------------------------------------------
def save_snapshot(snapshot: BenchmarkSnapshot, root: Path | None = None) -> Path:
    """Persist ``snapshot`` to ``results/runs/<slug>__<id>/``."""
    root = root or RUNS_DIR
    folder = root / f"{_slugify(snapshot.name)}__{snapshot.snapshot_id}"
    folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(snapshot, folder / "snapshot.joblib")
    snapshot.leaderboard.to_csv(folder / "leaderboard.csv")
    snapshot.predictions.to_csv(folder / "predictions.csv")
    with (folder / "meta.json").open("w") as fh:
        json.dump(snapshot.to_dict(), fh, indent=2, default=str)
    logger.info("Saved snapshot %s to %s", snapshot.snapshot_id, folder)
    return folder


def list_snapshots(root: Path | None = None) -> list[BenchmarkSnapshot]:
    """Return every snapshot currently on disk, newest first."""
    root = root or RUNS_DIR
    if not root.exists():
        return []
    snaps: list[BenchmarkSnapshot] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        pkl = folder / "snapshot.joblib"
        if not pkl.exists():
            continue
        try:
            snaps.append(joblib.load(pkl))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load snapshot at %s: %s", folder, exc)
    snaps.sort(key=lambda s: s.created_at, reverse=True)
    return snaps


def load_snapshot(snapshot_id: str, root: Path | None = None) -> BenchmarkSnapshot | None:
    for snap in list_snapshots(root):
        if snap.snapshot_id == snapshot_id:
            return snap
    return None


def delete_snapshot(snapshot_id: str, root: Path | None = None) -> bool:
    root = root or RUNS_DIR
    if not root.exists():
        return False
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        meta = folder / "meta.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
                if data.get("snapshot_id") == snapshot_id:
                    _rmtree(folder)
                    return True
            except Exception:
                continue
    return False


def _rmtree(folder: Path) -> None:
    for child in folder.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink(missing_ok=True)
    folder.rmdir()

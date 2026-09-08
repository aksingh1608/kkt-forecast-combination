"""Tests for the cross-dataset comparison layer.

We build 3 small synthetic datasets, run the pipeline on each, save them
as snapshots, then verify every comparison primitive (matrix, ranks,
champions, Friedman test) returns sane, typed output.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.comparison import (
    LOWER_IS_BETTER,
    algorithm_summary,
    best_pair_overall,
    build_metric_matrix,
    build_rank_matrix,
    champions_per_dataset,
    friedman_across_datasets,
    global_champion,
)
from src.data import clean_dataset
from src.pipeline import run_experiment
from src.results_store import (
    delete_snapshot,
    list_snapshots,
    save_snapshot,
    snapshot_from_result,
)


def _make_series(seed: int, n: int = 1200, amp: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    y = (
        5.0 + 0.0005 * t
        + amp * np.sin(2 * np.pi * t / 24)
        + 0.6 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.4, size=n)
    )
    return pd.DataFrame({"y": y}, index=idx)


def _make_snapshot(label: str, seed: int) -> "BenchmarkSnapshot":
    df = clean_dataset(_make_series(seed=seed))
    result = run_experiment(
        df, target="y",
        seasonal_period=24, test_size=0.2,
        include_heavy_classical=False,
    )
    snap = snapshot_from_result(
        result, name=label,
        dataset_signature={"freq": "h", "n_rows": len(df),
                           "start": str(df.index.min()),
                           "end": str(df.index.max()),
                           "missing_pct": 0.0, "target": "y"},
    )
    return snap


def test_cross_dataset_comparison_end_to_end(tmp_path):
    """Run 3 synthetic benchmarks, save them, verify every analytic."""
    snaps = []
    for i, name in enumerate(["Synth-A", "Synth-B", "Synth-C"]):
        snap = _make_snapshot(name, seed=i + 1)
        save_snapshot(snap, root=tmp_path)
        snaps.append(snap)

    loaded = list_snapshots(tmp_path)
    assert len(loaded) == 3

    matrix = build_metric_matrix(loaded, metric="MASE")
    assert matrix.shape[0] == 3
    assert matrix.shape[1] >= 6     # ML zoo has many models

    ranks = build_rank_matrix(loaded, metric="MASE")
    assert ranks.shape == matrix.shape
    assert (ranks >= 1).all().all()

    champs = champions_per_dataset(loaded, metric="MASE")
    assert len(champs) == 3
    assert set(champs["Dataset"]) == {"Synth-A", "Synth-B", "Synth-C"}

    algo = algorithm_summary(loaded, metric="MASE")
    assert not algo.empty
    assert "mean_rank" in algo.columns
    # First algorithm by mean_rank wins more or equal vs. last
    assert algo["mean_rank"].iloc[0] <= algo["mean_rank"].iloc[-1]

    g = global_champion(loaded, metric="MASE")
    assert g is not None
    assert g.dataset in {"Synth-A", "Synth-B", "Synth-C"}
    assert g.metric == "MASE"

    fried = friedman_across_datasets(loaded, metric="MASE")
    assert fried is not None
    stat, p = fried
    assert np.isfinite(stat)
    assert 0.0 <= p <= 1.0

    summary = best_pair_overall(loaded, metric="MASE")
    assert summary is not None
    assert summary["n_snapshots"] == 3

    # MASE is in LOWER_IS_BETTER — sanity-check the constant set used by UI.
    assert "MASE" in LOWER_IS_BETTER
    assert "R2" not in LOWER_IS_BETTER

    # Cleanup
    for snap in loaded:
        delete_snapshot(snap.snapshot_id, root=tmp_path)
    assert list_snapshots(tmp_path) == []

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

"""Rerun every standalone model on selected datasets, in place.

``run_paper_experiments.py`` rebuilds ``leaderboards.csv`` from an append-only
partial file, so running it on one dataset would drop the others. This script
instead replaces only the rows of the datasets it reruns:

* ``results/leaderboards.csv`` rows for (dataset, protocol, model),
* ``results/predictions/{dataset}__{protocol}__{model}.parquet`` and
  ``..._y_true.parquet``.

The CSE is left out (``scripts/rerun_cse.py`` handles it) and NLinear /
DLinear are scored with their defaults here and then retuned by
``scripts/rerun_lite_dl_tuned.py``, as in the original pipeline.

Run::

    python scripts/rerun_dataset.py --datasets OPSD_DE_household
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.models.registry import build_registry  # noqa: E402
from src.pipeline import load_project_dataset, run_experiment  # noqa: E402
from src.seed import set_global_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rerun_dataset")

NO_HEAVY = {"OPSD_DE_total"}   # same memory guard as the main benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--protocols", nargs="+", default=["one_step", "recursive"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    set_global_seed(42)
    args = parse_args()
    res = args.results
    pred_dir = res / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    lb_path = res / "leaderboards.csv"
    lb = pd.read_csv(lb_path)
    cols = lb.columns.tolist()

    for dataset in args.datasets:
        df, target, meta = load_project_dataset(dataset)
        season = int(meta.get("season", 24))
        removed = meta.get("removed_readings")
        if isinstance(removed, pd.DataFrame):
            out = res / "tables" / f"{dataset}_removed_readings.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            removed.to_csv(out, index=False)
            logger.info("[%s] wrote %s (%d readings set to missing)", dataset, out, len(removed))
        specs = build_registry(seasonal_period=season, n_jobs=args.n_jobs,
                               include_heavy_classical=dataset not in NO_HEAVY)
        specs = [s for s in specs if not s.name.lower().startswith(("fahe", "cse"))]
        logger.info("[%s] rows=%d models=%d", dataset, len(df), len(specs))
        for protocol in args.protocols:
            result = run_experiment(
                df, target=target, seasonal_period=season, test_size=args.test_size,
                model_specs=specs, include_heavy_classical=dataset not in NO_HEAVY,
                n_jobs=args.n_jobs, protocol=protocol,
            )
            pd.Series(result.y_test.values, index=result.test_index, name="y_true").to_frame().to_parquet(
                pred_dir / f"{dataset}__{protocol}__y_true.parquet")
            for run in result.runs:
                safe = run.name.replace("/", "_").replace(" ", "_")
                if run.error is not None:
                    logger.warning("[%s/%s] %s failed: %s", dataset, protocol, run.name, run.error)
                    stale = pred_dir / f"{dataset}__{protocol}__{safe}.parquet"
                    if stale.exists():
                        stale.unlink()   # never leave an old forecast behind
                    continue
                pd.Series(run.predictions, index=result.test_index, name="y_pred").to_frame().to_parquet(
                    pred_dir / f"{dataset}__{protocol}__{safe}.parquet")
            new = result.leaderboard(sort_by="RMSE")
            new.insert(0, "Protocol", protocol)
            new.insert(0, "Dataset", dataset)
            new = new[new["RMSE"].notna()][[c for c in cols if c in new.columns]]
            rerun_names = [s.name for s in specs]
            keep = ~((lb.Dataset == dataset) & (lb.Protocol == protocol) & lb.Model.isin(rerun_names))
            lb = pd.concat([lb[keep], new], ignore_index=True)[cols]
            lb.to_csv(lb_path, index=False)
            best = new.sort_values("RMSE").iloc[0]
            logger.info("[%s/%s] %d models rescored; best %s RMSE=%.4f",
                        dataset, protocol, len(new), best.Model, best.RMSE)


if __name__ == "__main__":
    main()

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

"""Per-horizon leaderboard at h ∈ {1, 3, 6, 12, 24}.

Consumes the recursive-protocol prediction Parquet files written by
``run_paper_experiments.py`` and produces ``results/multistep/*.csv``
containing (model × horizon) tables per dataset and metric.

For each model on each dataset:

    * y_pred is the recursive multi-step forecast over the full test window.
    * We report RMSE / MAE / R² / MASE on the first h steps for
      h ∈ {1, 3, 6, 12, 24} — the standard IEEE-TSG per-horizon breakdown.

Run::

    python scripts/run_multistep_benchmark.py --in results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.horizon import per_horizon_leaderboard

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("multistep")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=Path,
                   default=Path("results"))
    p.add_argument("--out", type=Path, default=None,
                   help="Default: <in>/multistep/")
    p.add_argument("--protocol", default="recursive",
                   choices=["one_step", "recursive"],
                   help="Which persisted protocol to break down (default: recursive).")
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 3, 6, 12, 24])
    p.add_argument("--metrics", nargs="+",
                   default=["RMSE", "MAE", "R2", "MASE"])
    return p.parse_args()


def _load_predictions(in_dir: Path, protocol: str) -> dict[str, dict[str, tuple[pd.Series, pd.Series]]]:
    """Return {dataset: {model: (y_true, y_pred)}} from the Parquet cache."""
    pred_dir = in_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"{pred_dir} not found — run run_paper_experiments.py first.")

    y_true_files = list(pred_dir.glob(f"*__{protocol}__y_true.parquet"))
    out: dict[str, dict[str, tuple[pd.Series, pd.Series]]] = {}
    for f in y_true_files:
        dataset = f.name.split("__")[0]
        y_true = pd.read_parquet(f)["y_true"]
        out[dataset] = {}
        for pf in pred_dir.glob(f"{dataset}__{protocol}__*.parquet"):
            if pf.name.endswith("__y_true.parquet"):
                continue
            model = pf.stem.split(f"__{protocol}__")[1]
            y_pred = pd.read_parquet(pf)["y_pred"]
            out[dataset][model] = (y_true, y_pred)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out or (args.in_dir / "multistep")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ds = _load_predictions(args.in_dir, args.protocol)
    summary_rows = []
    for dataset, models in per_ds.items():
        if not models:
            continue
        y_true_arr = next(iter(models.values()))[0]
        preds = {name: yp.values for name, (_, yp) in models.items()}
        # Load the corresponding training slice so MASE can be computed.
        y_train_arr = None
        try:
            from src.pipeline import load_project_dataset
            from src.features.engineering import build_supervised_frame
            from src.data.preprocessing import train_test_split_temporal
            df, target, meta = load_project_dataset(dataset)
            X, y, _ = build_supervised_frame(df, target)
            split = train_test_split_temporal(y.to_frame(), test_size=0.2)
            y_train_arr = split["train"][target].values
        except Exception as exc:
            logger.warning("[%s] could not load y_train for MASE: %s", dataset, exc)

        for metric in args.metrics:
            table = per_horizon_leaderboard(
                y_true_arr.values, preds,
                horizons=args.horizons, metric=metric,
                y_train=y_train_arr, seasonality=24,
            )
            table.insert(0, "Dataset", dataset)
            out_path = out_dir / f"{dataset}__{args.protocol}__{metric}.csv"
            table.to_csv(out_path)
            logger.info("Wrote %s", out_path)
            long = table.reset_index().melt(id_vars=["Dataset", "model"],
                                            var_name="horizon", value_name=metric)
            summary_rows.append(long)

    if summary_rows:
        combined = pd.concat(summary_rows, axis=0, ignore_index=True)
        combined.to_csv(out_dir / f"long_all__{args.protocol}.csv", index=False)
        logger.info("Wrote combined long table: %s",
                    out_dir / f"long_all__{args.protocol}.csv")


if __name__ == "__main__":
    main()

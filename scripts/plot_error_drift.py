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

"""Error-drift figures: daily MAE/MAPE and cumulative |error| over the test window.

Reads persisted predictions from ``results/predictions/`` and writes
PDF figures under ``results/figures/``.

Run::

    python scripts/plot_error_drift.py --in results/
    python scripts/plot_error_drift.py --in results/ --datasets UCI --protocol recursive
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("error_drift")

# Keep the figure readable: a few classical + a few trees + CSE.
_DEFAULT_MODELS = (
    "SARIMA", "Holt-Winters", "SeasonalNaive", "ARIMA",
    "ExtraTrees", "CatBoost", "LightGBM", "XGBoost",
    "FAHE_(Ours)", "RandomForest", "Ridge",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--protocol", default="recursive",
                   choices=["one_step", "recursive", "both"])
    p.add_argument("--metric", default="MAE", choices=["MAE", "MAPE"])
    p.add_argument("--models", nargs="*", default=None)
    return p.parse_args()


def _load_dataset_preds(
    pred_dir: Path, dataset: str, protocol: str
) -> tuple[pd.Series, dict[str, np.ndarray]]:
    y_path = pred_dir / f"{dataset}__{protocol}__y_true.parquet"
    if not y_path.exists():
        raise FileNotFoundError(y_path)
    y_true = pd.read_parquet(y_path)["y_true"]
    preds: dict[str, np.ndarray] = {}
    for pf in pred_dir.glob(f"{dataset}__{protocol}__*.parquet"):
        if pf.name.endswith("__y_true.parquet"):
            continue
        name = pf.stem.split(f"__{protocol}__")[1]
        preds[name] = pd.read_parquet(pf)["y_pred"].to_numpy(dtype=float)
    return y_true, preds


def _select_models(
    available: dict[str, np.ndarray], wanted: list[str] | None
) -> dict[str, np.ndarray]:
    if wanted:
        keys = [m for m in wanted if m in available]
    else:
        keys = [m for m in _DEFAULT_MODELS if m in available]
        # Fill up to ~8 models if defaults missing.
        for m in available:
            if m not in keys:
                keys.append(m)
            if len(keys) >= 8:
                break
    return {m: available[m] for m in keys}


def _plot_matplotlib(
    y_true: pd.Series,
    preds: dict[str, np.ndarray],
    *,
    metric: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    yt = y_true.to_numpy(dtype=float)
    idx = y_true.index
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for name, yp in preds.items():
        n = min(len(yt), len(yp))
        err = np.abs(yt[:n] - yp[:n])
        if metric.upper() == "MAPE":
            denom = np.where(np.abs(yt[:n]) < 0.05, 0.05, np.abs(yt[:n]))
            err = 100.0 * err / denom
        s = pd.Series(err, index=idx[:n])
        daily = s.resample("1D").mean()
        axes[0].plot(daily.index, daily.values, label=name, linewidth=1.6)
        axes[1].plot(s.index, s.cumsum().values, label=name, linewidth=1.6)

    ylab = "daily MAPE (%)" if metric.upper() == "MAPE" else "daily MAE"
    axes[0].set_ylabel(ylab)
    axes[0].set_title("Daily-mean absolute error")
    axes[1].set_ylabel("cumulative |error|")
    axes[1].set_title("Cumulative absolute error")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle(title, fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out or (in_dir / "figures")
    pred_dir = in_dir / "predictions"
    protocols = (
        ["one_step", "recursive"] if args.protocol == "both" else [args.protocol]
    )

    y_true_files = list(pred_dir.glob("*__*__y_true.parquet"))
    datasets = sorted({f.name.split("__")[0] for f in y_true_files})
    if args.datasets:
        datasets = [d for d in datasets if d in set(args.datasets)]

    for dataset in datasets:
        for protocol in protocols:
            try:
                y_true, all_preds = _load_dataset_preds(pred_dir, dataset, protocol)
            except FileNotFoundError:
                logger.warning("Missing preds for %s / %s — skip.", dataset, protocol)
                continue
            if not isinstance(y_true.index, pd.DatetimeIndex):
                y_true.index = pd.to_datetime(y_true.index)
            preds = _select_models(all_preds, args.models)
            if not preds:
                continue
            title = f"{dataset} · {protocol} · error drift"
            out_path = out_dir / f"error_drift__{dataset}__{protocol}__{args.metric}.pdf"
            _plot_matplotlib(
                y_true, preds, metric=args.metric, title=title, out_path=out_path
            )


if __name__ == "__main__":
    main()

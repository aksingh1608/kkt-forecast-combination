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

"""Full paper runner: 5 datasets × 23 models × 2 protocols.

Generates every headline artefact needed for the paper:

* ``results/leaderboards.csv`` — long-format (dataset × protocol × model × metrics).
* ``results/cse_weights_by_dataset.csv`` — the KKT weight table.
* ``results/predictions/{dataset}__{protocol}__{model}.parquet`` —
  the full y_true / y_pred series (used later by the significance script).
* ``results/cse_diagnostics/{dataset}.pkl`` — the CSEDiagnostics
  object per dataset (consumed by ``plot_kkt_boundary.py``).

Each (dataset, protocol) pair is checkpointed immediately after completion so
a killed run can be resumed with ``--resume``.

Run::

    python scripts/run_paper_experiments.py --out results/
    python scripts/run_paper_experiments.py --out results/ --resume
    python scripts/run_paper_experiments.py --datasets UCI OPSD_DE_household
    python scripts/run_paper_experiments.py --dl                      # add deep zoo (GPU)
    python scripts/run_paper_experiments.py --smoke                   # fast subset run
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.pipeline import load_project_dataset, run_experiment
from src.seed import set_global_seed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("paper_runner")

# OPSD national load is large; full SARIMA state arrays can exceed ~4 GB RAM.
# Classical models still run inside CSE; standalone SARIMA/ARIMA/HW/Theta are
# skipped on this dataset unless the user overrides with --force-heavy-opsd.
_LARGE_DATASETS_NO_HEAVY = frozenset({"OPSD_DE_total"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"),
                   help="Output artefact directory (default: results/).")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Subset of PROJECT_DATASETS to run. Default: all 5.")
    p.add_argument("--protocols", nargs="*", default=["one_step", "recursive"],
                   choices=["one_step", "recursive"],
                   help="Evaluation protocols to run (default: both).")
    p.add_argument("--no-heavy", action="store_true",
                   help="Skip ARIMA/SARIMA/Holt-Winters/Theta (fast).")
    p.add_argument("--force-heavy-opsd", action="store_true",
                   help="Run heavy classical on OPSD_DE_total despite memory risk.")
    p.add_argument("--resume", action="store_true",
                   help="Skip (dataset, protocol) pairs whose checkpoint marker exists.")
    p.add_argument("--dl", action="store_true",
                   help="Include the optional deep-learning zoo (GPU recommended).")
    p.add_argument("--dl-max-steps", type=int, default=200)
    p.add_argument("--dl-horizon", type=int, default=None)
    p.add_argument("--dl-input-size", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--smoke", action="store_true",
                   help="Fast smoke run: skip heavy classical, small n_jobs.")
    return p.parse_args()


def _checkpoint_marker(out_dir: Path, dataset: str, protocol: str) -> Path:
    return out_dir / "checkpoints" / f"{dataset}__{protocol}.done"


def _is_done(out_dir: Path, dataset: str, protocol: str) -> bool:
    return _checkpoint_marker(out_dir, dataset, protocol).exists()


def _mark_done(out_dir: Path, dataset: str, protocol: str) -> None:
    marker = _checkpoint_marker(out_dir, dataset, protocol)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")


def _persist_predictions(
    out_dir: Path, dataset: str, protocol: str, result
) -> None:
    """Write y_true and every model's y_pred to Parquet."""
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    y_test = pd.Series(result.y_test.values, index=result.test_index, name="y_true")
    y_test.to_frame().to_parquet(
        pred_dir / f"{dataset}__{protocol}__y_true.parquet"
    )

    for run in result.runs:
        if run.error is not None:
            continue
        preds = pd.Series(run.predictions, index=result.test_index, name="y_pred")
        safe = run.name.replace("/", "_").replace(" ", "_")
        preds.to_frame().to_parquet(
            pred_dir / f"{dataset}__{protocol}__{safe}.parquet"
        )


def _append_leaderboard_rows(
    out_dir: Path, dataset: str, protocol: str, result
) -> None:
    """Append one dataset's leaderboard rows to the partial CSV."""
    lb = result.leaderboard(sort_by="RMSE")
    lb.insert(0, "Protocol", protocol)
    lb.insert(0, "Dataset", dataset)
    partial = out_dir / "leaderboards_partial.csv"
    header = not partial.exists()
    lb.to_csv(partial, mode="a", header=header, index=False)


def _rebuild_leaderboards(out_dir: Path) -> pd.DataFrame:
    partial = out_dir / "leaderboards_partial.csv"
    if not partial.exists():
        return pd.DataFrame()
    df = pd.read_csv(partial)
    df.to_csv(out_dir / "leaderboards.csv", index=False)
    return df


def _include_heavy_for_dataset(
    dataset: str, args: argparse.Namespace
) -> bool:
    if args.no_heavy or args.smoke:
        return False
    if dataset in _LARGE_DATASETS_NO_HEAVY and not args.force_heavy_opsd:
        logger.info(
            "[%s] skipping standalone heavy classical (memory guard); "
            "CSE still fits SARIMA/HW internally. Use --force-heavy-opsd to override.",
            dataset,
        )
        return False
    return True


def _persist_cse_diagnostics(out_dir: Path, dataset: str) -> object | None:
    """Refit CSE and pickle validation-slice KKT diagnostics."""
    from src.models.novel import FAHEForecaster
    from src.data.preprocessing import train_test_split_temporal
    from src.features.engineering import build_supervised_frame

    diag_path = out_dir / "cse_diagnostics" / f"{dataset}.pkl"
    if diag_path.exists():
        logger.info("[%s] CSE diagnostics already on disk — skipping refit.", dataset)
        with open(diag_path, "rb") as fh:
            return pickle.load(fh)

    logger.info("[%s] refitting CSE for KKT diagnostics…", dataset)
    df, target, meta = load_project_dataset(dataset)
    feature_cfg = config.FeatureConfig()
    X, y, _ = build_supervised_frame(df, target, cfg=feature_cfg)
    df_target = df[[target]].loc[X.index]
    split_X = train_test_split_temporal(X, test_size=0.2)
    split_y = train_test_split_temporal(df_target, test_size=0.2)
    y_train = split_y["train"][target]
    X_train = split_X["train"]

    cse = FAHEForecaster(seasonal_periods=int(meta.get("season", 24)))
    try:
        cse.fit(y_train, X_train=X_train)
        diag = cse.diagnostics(dataset=dataset)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "wb") as fh:
            pickle.dump(diag, fh)
        logger.info("[%s] CSE diagnostics saved (K=%d members, n_val=%d)",
                    dataset, len(diag.member_names), diag.n_validation)
        return diag
    except Exception as exc:
        logger.exception("[%s] CSE diagnostics failed: %s", dataset, exc)
        return None


def main() -> None:
    set_global_seed(42)
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "cse_diagnostics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(config.PROJECT_DATASETS)
    logger.info(
        "Running datasets=%s  protocols=%s  dl=%s  resume=%s",
        datasets, args.protocols, args.dl, args.resume,
    )

    for protocol in args.protocols:
        logger.info("========== protocol = %s ==========", protocol)
        for dataset in datasets:
            if args.resume and _is_done(out_dir, dataset, protocol):
                logger.info("[%s / %s] checkpoint found — skipping.", dataset, protocol)
                continue

            df, target, meta = load_project_dataset(dataset)
            season = int(meta.get("season", config.FREQ_TO_SEASON.get("h", 24)))
            logger.info("[%s / %s] rows=%d, target=%s", dataset, protocol, len(df), target)

            result = run_experiment(
                df,
                target=target,
                seasonal_period=season,
                test_size=args.test_size,
                include_heavy_classical=_include_heavy_for_dataset(dataset, args),
                include_optional_dl=args.dl,
                optional_dl_horizon=args.dl_horizon,
                optional_dl_max_steps=args.dl_max_steps,
                optional_dl_input_size=args.dl_input_size,
                n_jobs=args.n_jobs,
                protocol=protocol,
            )

            _persist_predictions(out_dir, dataset, protocol, result)
            _append_leaderboard_rows(out_dir, dataset, protocol, result)
            _mark_done(out_dir, dataset, protocol)
            logger.info("[%s / %s] checkpoint saved.", dataset, protocol)

    long_lb = _rebuild_leaderboards(out_dir)
    logger.info("Wrote %s (%d rows)", out_dir / "leaderboards.csv", len(long_lb))

    cse_weights_rows = []
    for ds in datasets:
        diag = _persist_cse_diagnostics(out_dir, ds)
        if diag is None:
            continue
        for i, name in enumerate(diag.member_names):
            cse_weights_rows.append({
                "dataset": ds,
                "member": name,
                "weight": float(diag.weights[i]),
                "sigma": float(diag.sigma[i]),
                "rho_vs_best": float(diag.rho[i, diag.best_member_idx]),
                "is_best": (i == diag.best_member_idx),
                "kkt_residual": float(diag.kkt_residuals[i]),
            })

    cse_df = pd.DataFrame(cse_weights_rows)
    if not cse_df.empty:
        cse_df.to_csv(out_dir / "cse_weights_by_dataset.csv", index=False)
        logger.info("Wrote %s", out_dir / "cse_weights_by_dataset.csv")

    print("\n=== Cross-dataset leaderboard (one_step, sorted by RMSE) ===")
    if not long_lb.empty:
        top = long_lb[long_lb["Protocol"] == "one_step"].sort_values(["Dataset", "RMSE"])
        print(top.head(80).to_string(index=False))


if __name__ == "__main__":
    main()

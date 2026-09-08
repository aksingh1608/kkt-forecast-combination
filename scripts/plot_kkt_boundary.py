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

"""Render the KKT-boundary figure (paper's signature Prop. 1 diagram).

Reads the ``CSEDiagnostics`` pickles under
``results/cse_diagnostics/`` and produces a small-multiples grid
showing each base learner in the (σ_i/σ_best, ρ_{i,best}) plane, overlaid
with the theoretical exclusion curve ρ = σ_best / σ_i.

Run::

    python scripts/plot_kkt_boundary.py --in results/ --out results/figures/kkt_boundary.pdf
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.kkt_analysis import plot_kkt_boundary, kkt_conditions

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("kkt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=Path,
                   default=Path("results"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output PDF. Default: <in>/figures/kkt_boundary.pdf")
    p.add_argument("--table-out", type=Path, default=None,
                   help="Optional CSV with the per-dataset KKT verification table.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    diag_dir = args.in_dir / "cse_diagnostics"
    if not diag_dir.exists():
        raise FileNotFoundError(
            f"{diag_dir} not found — run run_paper_experiments.py first."
        )
    diagnostics = []
    for f in sorted(diag_dir.glob("*.pkl")):
        with open(f, "rb") as fh:
            diagnostics.append(pickle.load(fh))
        logger.info("Loaded diagnostics for %s (K=%d, n_val=%d)",
                    diagnostics[-1].dataset, len(diagnostics[-1].member_names),
                    diagnostics[-1].n_validation)
    if not diagnostics:
        raise RuntimeError("No CSE diagnostics found — cannot plot KKT boundary.")

    fig = plot_kkt_boundary(
        diagnostics,
        title="KKT exclusion boundary (Proposition 1) — per dataset",
        annotate=True,
    )
    out_path = args.out or (args.in_dir / "figures" / "kkt_boundary.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    logger.info("Wrote %s", out_path)

    if args.table_out:
        import pandas as pd
        tables = [kkt_conditions(d).assign(dataset=d.dataset) for d in diagnostics]
        pd.concat(tables, axis=0, ignore_index=True).to_csv(args.table_out, index=False)
        logger.info("Wrote %s", args.table_out)


if __name__ == "__main__":
    main()

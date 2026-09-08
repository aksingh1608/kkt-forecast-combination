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

"""Export manuscript tables as CSV and LaTeX under ``results/tables/``.

Horizon numbers come only from ``results/multistep_rolling/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    tex_path = stem.with_suffix(".tex")
    df.to_csv(csv_path, index=False)
    try:
        tex = df.to_latex(index=False, float_format="%.4f")
    except Exception:
        tex = df.to_string(index=False)
    tex_path.write_text(tex)
    print(f"wrote {csv_path.resolve().relative_to(ROOT)} / {tex_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or (args.results / "tables")

    lb = pd.read_csv(args.results / "leaderboards.csv")
    _write(lb, out / "table_leaderboards_long")

    one = lb[lb.Protocol == "one_step"][
        ["Dataset", "Model", "Family", "MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE"]
    ]
    _write(one, out / "table_onestep_metrics")

    rec = lb[lb.Protocol == "recursive"][
        ["Dataset", "Model", "Family", "MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE"]
    ]
    _write(rec, out / "table_recursive_metrics")

    weights = pd.read_csv(args.results / "cse_weights_by_dataset.csv")
    _write(weights, out / "table_cse_weights")

    rolling = pd.read_csv(args.results / "multistep_rolling" / "long_all__rolling.csv")
    _write(rolling, out / "table_horizons_rolling")

    friedman = pd.read_csv(args.results / "significance" / "friedman_summary.csv")
    _write(friedman, out / "table_friedman")

    print("table build complete")


if __name__ == "__main__":
    main()

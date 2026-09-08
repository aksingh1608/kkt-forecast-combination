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

"""Create a small paper-writing bundle from generated artefacts.

The full project is large because it contains the virtual environment, raw
datasets and cached parquet files. This script copies only the manuscript
artefacts: tables, significance outputs, figures, KKT diagnostics and method
notes.

Run::

    python scripts/create_paper_pack.py --in results/ --out paper_pack/
    zip -r paper_pack.zip paper_pack
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("paper_pack"))
    p.add_argument("--include-predictions", action="store_true")
    return p.parse_args()


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    in_dir = args.in_dir
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for file_name in [
        "leaderboards.csv",
        "leaderboards_partial.csv",
        "cse_weights_by_dataset.csv",
        "paper_run.log",
    ]:
        _copy_file(in_dir / file_name, out / "results" / file_name)

    for dir_name in [
        "figures",
        "significance",
        "multistep",
        "multistep_rolling",
        "cse_diagnostics",
        "tuning",
    ]:
        _copy_tree(in_dir / dir_name, out / "results" / dir_name)

    if args.include_predictions:
        _copy_tree(in_dir / "predictions", out / "results" / "predictions")

    for file_name in ["README.md", "PAPER_README.md", "requirements.txt"]:
        _copy_file(Path(file_name), out / file_name)

    notes = out / "METHOD_NOTES.md"
    if not notes.exists():
        notes.write_text(
            "# Method Notes for Manuscript Writer\n\n"
            "- Use `results/leaderboards.csv` for main one-step/recursive leaderboards.\n"
            "- Use `results/multistep_rolling/` for paper horizon tables; `multistep/` is prefix diagnostic only.\n"
            "- Use `results/cse_weights_by_dataset.csv` and `figures/kkt_boundary.pdf` for CSE collapse/KKT discussion.\n"
            "- Use `results/significance/` for DM, Cohen's d, Cliff's delta and Friedman summaries.\n"
            "- Use `results/figures/error_drift__*.pdf` for daily/cumulative error drift figures.\n"
            "- Do not invent Transformer results unless `--dl` experiments are run and included.\n",
            encoding="utf-8",
        )

    print(f"Wrote paper pack: {out.resolve()}")
    print("Create zip with: zip -r paper_pack.zip", out)


if __name__ == "__main__":
    main()

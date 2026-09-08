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

"""Fast install check: tiny UCI window, no heavy classical, CPU only.

Target runtime: under two minutes on a laptop CPU.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import clean_dataset, load_uci_household_power, resample_dataset
from src.pipeline import run_experiment
from src.seed import set_global_seed

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    set_global_seed(42)
    t0 = time.perf_counter()
    # ~30 days of minute data → hourly (~720 rows). Lag/rolling windows need
    # ≥168 hours of history after dropna; shorter windows leave X empty.
    df_raw = load_uci_household_power(nrows=43_200, use_cache=False)
    df = clean_dataset(df_raw)
    df_h = resample_dataset(df, freq="h")
    print(f"frame={df_h.shape} range={df_h.index.min()}..{df_h.index.max()}")
    result = run_experiment(
        df_h,
        target="Global_active_power",
        seasonal_period=24,
        test_size=0.2,
        include_heavy_classical=False,
        n_jobs=1,
    )
    lb = result.leaderboard(sort_by="RMSE")
    # Require at least one finite RMSE so the install is meaningfully checked.
    ok = lb["RMSE"].notna().any() if "RMSE" in lb.columns else False
    print(lb.head(8).to_string(index=False))
    elapsed = time.perf_counter() - t0
    print(f"smoke_ok={ok} elapsed_s={elapsed:.1f}")
    if not ok:
        raise SystemExit("smoke failed: no finite RMSE rows")
    if elapsed > 120:
        print("WARNING: smoke exceeded two minutes", file=sys.stderr)


if __name__ == "__main__":
    main()

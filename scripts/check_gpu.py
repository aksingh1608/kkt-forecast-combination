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

"""Diagnose the GPU stack and run a tiny benchmark.

Use this after installing CUDA torch / neuralforecast / chronos to
verify everything is wired up correctly. Also useful when shipping a
research-paper run on a new machine — it surfaces the device, the
selected mixed-precision mode, and the actual speed-up over CPU.

Run::

    python scripts/check_gpu.py           # hardware + import probe
    python scripts/check_gpu.py --bench   # also run a 5-min NHITS benchmark
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compute import detect_device, lightning_accelerator


def _probe_imports() -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    for mod, attr in [
        ("torch", "__version__"),
        ("neuralforecast", "__version__"),
        ("chronos", "__version__"),
        ("lightning", "__version__"),
        ("pytorch_lightning", "__version__"),
    ]:
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, attr, "(unknown)")
        except ImportError:
            out[mod] = False
    return out


def _print_header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _hardware_report() -> None:
    info = detect_device()
    accel, devices, precision = lightning_accelerator(info)

    _print_header("Compute device")
    print(f"  backend       : {info.backend}")
    print(f"  name          : {info.name}")
    print(f"  n_devices     : {info.n_devices}")
    print(f"  vram_gb       : {info.vram_gb}")
    print(f"  supports_amp  : {info.supports_amp}")
    print(f"  → Lightning   : accelerator={accel!r}, devices={devices}, "
          f"precision={precision!r}")

    _print_header("Library versions")
    for k, v in _probe_imports().items():
        marker = "OK " if v else "-- "
        print(f"  {marker}{k:18s} {v if v else '(not installed)'}")


def _mini_dataset(n: int = 1500, seed: int = 0) -> pd.Series:
    """Hourly-ish synthetic series with day + week seasonality + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = (
        20
        + 5 * np.sin(2 * np.pi * t / 24)
        + 3 * np.sin(2 * np.pi * t / (24 * 7))
        + 0.001 * t
        + rng.normal(0, 1.0, n)
    )
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(y, index=idx, name="kw")


def _bench_nhits(max_steps: int = 120) -> None:
    """Train an NHITS once on CPU and once on whatever the auto device is."""
    try:
        from src.models.optional_dl import (
            NeuralForecastWrapper, neuralforecast_available,
        )
    except Exception as exc:  # pragma: no cover
        print(f"NeuralForecast wrapper unavailable: {exc}")
        return
    if not neuralforecast_available():
        print("`neuralforecast` is not installed — skipping benchmark.")
        return

    y = _mini_dataset()
    n_test = 96
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    def _run(label: str, force_cpu: bool) -> tuple[float, float]:
        os.environ["FORCE_CPU"] = "1" if force_cpu else "0"
        # NeuralForecast caches some imports; reload our wrapper module so
        # the freshly-set FORCE_CPU is honoured by detect_device().
        importlib.reload(importlib.import_module("src.compute.device"))
        from src.compute import detect_device  # local re-import
        info = detect_device()
        from src.models.optional_dl import NeuralForecastWrapper as NFW

        m = NFW(arch="nhits", horizon=n_test, input_size=168,
                max_steps=max_steps)
        t0 = time.perf_counter()
        m.fit(y_train)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        yhat = m.predict(n_test)
        t_pred = time.perf_counter() - t0
        rmse = float(np.sqrt(np.mean((yhat - y_test.values) ** 2)))
        print(f"  {label:18s} dev={info.name:35s} "
              f"fit={t_fit:6.2f}s  predict={t_pred:5.2f}s  rmse={rmse:.3f}")
        return t_fit, t_pred

    _print_header(f"NHITS micro-benchmark (max_steps={max_steps})")
    cpu_fit, cpu_pred = _run("[CPU forced]", force_cpu=True)
    auto_fit, auto_pred = _run("[Auto device]", force_cpu=False)
    if auto_fit > 0:
        print(f"\n  Speedup (fit)     : {cpu_fit / auto_fit:5.2f}x")
        print(f"  Speedup (predict) : {cpu_pred / max(auto_pred, 1e-6):5.2f}x")
    os.environ.pop("FORCE_CPU", None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", action="store_true",
                   help="Run a small NHITS benchmark (CPU vs auto-device).")
    p.add_argument("--max-steps", type=int, default=120,
                   help="Training steps for the benchmark (default 120).")
    args = p.parse_args()

    _hardware_report()
    if args.bench:
        _bench_nhits(max_steps=args.max_steps)
    print()


if __name__ == "__main__":
    main()

"""The OPSD household target must not jump when a meter starts or stops."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.extra_loaders import opsd_household_series


def _meters():
    idx = pd.date_range("2017-01-01", periods=10, freq="h")
    a = pd.Series(1000.0 + 0.5 * np.arange(10), index=idx)        # 0.5 kWh per hour
    b = pd.Series(5000.0 + 1.0 * np.arange(10), index=idx)        # 1.0 kWh per hour
    b.iloc[:4] = np.nan                                           # starts reporting at hour 4
    return pd.DataFrame({"h1": a, "h2": b})


def test_meter_joining_does_not_create_a_spike():
    s, removed = opsd_household_series(_meters())
    s = s.dropna()
    assert s.max() <= 1.0 + 1e-9
    assert removed.empty
    # the old construction (average, then difference) produces a jump of about 2000 kWh
    old = _meters().mean(axis=1, skipna=True).diff().clip(lower=0).dropna()
    assert old.max() > 1000


def test_impossible_step_is_removed():
    m = _meters()
    m.loc[m.index[6]:, "h1"] += 800.0                             # one-hour jump of 800 kWh
    s, removed = opsd_household_series(m)
    assert s.dropna().max() <= 1.0 + 1e-9
    assert len(removed) == 1 and removed.household.iloc[0] == "h1"

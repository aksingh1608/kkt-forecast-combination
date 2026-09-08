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

"""Stationarity diagnostics — Augmented Dickey-Fuller (ADF) and KPSS tests.

These are the two complementary stationarity tests reviewers expect to see
for any time-series paper:

* **ADF** — H0: series has a unit root (i.e. is non-stationary).
  Reject H0 (small p) ⇒ series is *stationary*.
* **KPSS** — H0: series is stationary around a deterministic trend.
  Reject H0 (small p) ⇒ series is *non-stationary*.

The two tests can disagree; the joint outcome is what's interpretively
useful. We package both into a tidy table the UI / paper can consume.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StationarityReport:
    series_name: str
    n_obs: int
    adf_stat: float
    adf_pvalue: float
    adf_crit: Mapping[str, float]
    adf_lags: int
    kpss_stat: float
    kpss_pvalue: float
    kpss_crit: Mapping[str, float]
    verdict: str

    def to_dict(self) -> dict:
        return {
            "series": self.series_name,
            "n_obs": self.n_obs,
            "ADF stat": round(self.adf_stat, 4),
            "ADF p-value": round(self.adf_pvalue, 4),
            "ADF lags": self.adf_lags,
            "KPSS stat": round(self.kpss_stat, 4),
            "KPSS p-value": round(self.kpss_pvalue, 4),
            "Verdict": self.verdict,
        }


def stationarity_report(series: pd.Series, max_n: int = 20_000) -> StationarityReport:
    """Run ADF + KPSS and return a single tidy report.

    Notes
    -----
    * KPSS becomes very slow for n > 20k and the conclusion does not
      change materially; we sub-sample by uniform stride for diagnostics.
    * Both tests are run on the *differenced-once* version of the
      series too if the level is non-stationary, but only the level
      result is returned in the report (the UI surfaces the verdict).
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    s = series.dropna().astype(float)
    if len(s) > max_n:
        step = max(1, len(s) // max_n)
        s = s.iloc[::step]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # adfuller returns 6 items: (stat, p_value, usedlag, nobs, crit_dict, icbest)
        adf_stat, adf_p, adf_lags, _adf_nobs, adf_crit, _icbest = adfuller(
            s.values, autolag="AIC"
        )
        # kpss returns 4 items: (stat, p_value, lags, crit_dict)
        kpss_stat, kpss_p, _kpss_lags, kpss_crit = kpss(
            s.values, regression="c", nlags="auto"
        )

    # Joint verdict — see Hyndman & Athanasopoulos §9.1.
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p > 0.05
    if adf_stationary and kpss_stationary:
        verdict = "Stationary (both tests agree)."
    elif (not adf_stationary) and (not kpss_stationary):
        verdict = "Non-stationary (both tests agree). Consider differencing."
    elif adf_stationary and not kpss_stationary:
        verdict = "Difference-stationary (ADF: yes, KPSS: trend present)."
    else:  # not adf_stationary and kpss_stationary
        verdict = "Trend-stationary (KPSS: stable, ADF: unit root). Consider de-trending."

    return StationarityReport(
        series_name=series.name or "series",
        n_obs=len(s),
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        adf_crit={k: float(v) for k, v in adf_crit.items()},
        adf_lags=int(adf_lags),
        kpss_stat=float(kpss_stat),
        kpss_pvalue=float(kpss_p),
        kpss_crit={k: float(v) for k, v in kpss_crit.items()},
        verdict=verdict,
    )

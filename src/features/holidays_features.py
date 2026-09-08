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

"""Holiday and calendar-event features.

Uses the `holidays` package (https://pypi.org/project/holidays/) to flag
public holidays for any country. Energy demand drops sharply on holidays,
so this is one of the highest-ROI exogenous features for residential
forecasting (Hong & Fan, 2016).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_holiday_features(
    df: pd.DataFrame,
    *,
    country: str = "FR",
    subdiv: str | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Add ``is_holiday``, ``is_holiday_eve``, ``days_to_holiday`` columns.

    Parameters
    ----------
    df
        Time-indexed frame.
    country
        ISO country code understood by the ``holidays`` package.
        Default 'FR' is correct for the bundled UCI dataset (Sceaux,
        France).
    subdiv
        Optional sub-division (e.g. state) for finer holidays.
    horizon
        Look-ahead window for the *eve* indicator.
    """
    try:
        import holidays as _h
    except ImportError:  # pragma: no cover
        logger.warning("`holidays` package not installed — returning df unchanged.")
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Holiday features require a DatetimeIndex.")
    out = df.copy()

    years = list(range(int(df.index.year.min()) - 1,
                       int(df.index.year.max()) + 2))
    try:
        cal = _h.country_holidays(country, subdiv=subdiv, years=years)
    except (KeyError, NotImplementedError):  # pragma: no cover
        logger.warning("Holidays not available for %s/%s — fallback to FR.",
                       country, subdiv)
        cal = _h.country_holidays("FR", years=years)

    cal_dates = {pd.Timestamp(d).date() for d in cal.keys()}
    dates_only = pd.Series(df.index.date)
    is_holiday = dates_only.isin(cal_dates).astype(int).values
    out["is_holiday"] = is_holiday

    # Eve indicator (next ``horizon`` days are a holiday).
    eve = np.zeros(len(out), dtype=int)
    for k in range(1, horizon + 1):
        shifted = pd.Series((df.index + pd.Timedelta(days=k)).date)
        eve_mask = shifted.isin(cal_dates).astype(int).values
        eve = np.maximum(eve, eve_mask)
    out["is_holiday_eve"] = eve

    # Days until the next holiday (capped at 30 to keep the feature bounded).
    holiday_days = sorted(d for d in cal.keys()
                          if pd.Timestamp(d) >= df.index.min() - pd.Timedelta(days=1))
    if holiday_days:
        holiday_ts = pd.DatetimeIndex(pd.to_datetime(holiday_days))
        next_idx = holiday_ts.searchsorted(df.index)
        next_idx = np.clip(next_idx, 0, len(holiday_ts) - 1)
        next_days = holiday_ts[next_idx]
        days_to = (next_days - df.index).days
        out["days_to_holiday"] = np.clip(days_to, 0, 30)
    else:
        out["days_to_holiday"] = 30

    return out


def list_supported_countries(top_n: int = 20) -> list[str]:
    """Return a small curated list of ISO country codes understood by holidays."""
    return [
        "FR", "DE", "ES", "GB", "IT", "PT", "NL", "BE", "CH", "AT",
        "US", "CA", "MX", "BR", "IN", "JP", "AU", "NZ", "SG", "CN",
    ][:top_n]

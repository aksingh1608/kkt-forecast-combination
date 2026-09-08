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

"""Weather covariate loader using Meteostat (free, no auth).

The UCI Household-Power-Consumption dataset is recorded in
**Sceaux, France (48.7715° N, 2.3009° E)**, a southern suburb of
Paris. We pull hourly temperature, humidity, dew point, wind, and
pressure for that location and re-index them onto the energy
dataset's timeline so they can be used as exogenous covariates.

If the user is benchmarking a different geography, set
``location=(lat, lon)`` accordingly; the helper auto-discovers the
nearest weather station.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

logger = logging.getLogger(__name__)


# UCI dataset is recorded in Sceaux, FR — see the original paper.
SCEAUX_LAT_LON: tuple[float, float] = (48.7715, 2.3009)
WEATHER_CACHE_DIR: Path = config.DATA_DIR / "external" / "weather"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_COLUMNS_KEEP = ("temp", "dwpt", "rhum", "wspd", "pres", "tsun")


@dataclass(frozen=True)
class WeatherSpec:
    name: str
    lat: float
    lon: float
    elevation_m: float | None = None


def meteostat_available() -> bool:
    try:
        import meteostat  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_weather(
    start: datetime, end: datetime,
    *, location: tuple[float, float] = SCEAUX_LAT_LON,
    cache_key: str = "sceaux",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch hourly weather features for the given window and location.

    Returns a frame indexed by UTC timestamp with columns
    ``temp, dwpt, rhum, wspd, pres, tsun`` (NaN-filled to forward).
    """
    if not meteostat_available():
        raise ImportError(
            "Install `meteostat` (pip install meteostat) to use weather covariates."
        )
    cache_path = WEATHER_CACHE_DIR / f"{cache_key}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    from meteostat import Hourly, Point
    pt = Point(*location)
    hourly = Hourly(pt, start, end).fetch()
    if hourly.empty:
        raise RuntimeError(
            f"Meteostat returned no data for {location} {start} … {end}."
        )
    keep = [c for c in WEATHER_COLUMNS_KEEP if c in hourly.columns]
    df = hourly[keep].copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df = df.sort_index().ffill().bfill()
    df.to_parquet(cache_path)
    logger.info("Cached %s weather rows -> %s", len(df), cache_path)
    return df


def attach_weather(
    df: pd.DataFrame,
    *, location: tuple[float, float] = SCEAUX_LAT_LON,
    cache_key: str = "sceaux",
    prefix: str = "wx_",
) -> pd.DataFrame:
    """Merge weather features into ``df`` (datetime-indexed)."""
    if df.index.size == 0:
        return df
    start = df.index.min().to_pydatetime()
    end = df.index.max().to_pydatetime()
    wx = fetch_weather(start, end, location=location, cache_key=cache_key)
    wx = wx.add_prefix(prefix)
    # Reindex weather onto the energy clock; forward-fill the gaps.
    wx = wx.reindex(df.index, method="ffill").ffill().bfill()
    return df.join(wx, how="left")


__all__ = [
    "fetch_weather", "attach_weather", "meteostat_available",
    "SCEAUX_LAT_LON", "WeatherSpec",
]

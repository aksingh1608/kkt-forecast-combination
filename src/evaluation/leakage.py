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

"""Data-leakage report — was this dataset in the foundation model's pre-training?

The published TSFM papers each enumerate their pre-training corpora.
For the residential-energy datasets we use, we cross-reference each
known TSFM corpus to flag potential leakage. This is an over-cautious
*static* check — if a dataset is on the corpus list, the TSFM has
*seen* it and zero-shot numbers should not be interpreted as truly
out-of-distribution.

Sources (correct as of November 2025):
* Chronos-Bolt — https://huggingface.co/amazon/chronos-bolt-base#training-data
* TimesFM-2.0 — https://github.com/google-research/timesfm/blob/main/README.md
* Moirai-2 — https://github.com/SalesforceAIResearch/uni2ts
* TinyTimeMixer (R2) — https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
* Time-MoE — https://github.com/Time-MoE/Time-MoE
* Sundial — https://github.com/thuml/Sundial

If any of these change their training mixes, edit ``_TRAINING_CORPORA``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# Lower-cased dataset shorthand → set of TSFM names whose training mix
# is *known to include* (or near-clones of) that dataset.
_TRAINING_CORPORA: dict[str, set[str]] = {
    # Electricity / energy --------------------------------------------------
    "electricityloaddiagrams": {"Chronos-Bolt", "TimesFM-2.0", "Moirai-2", "Time-MoE"},
    "elec":                    {"Chronos-Bolt", "TimesFM-2.0", "Moirai-2"},
    "ercot":                   {"Chronos-Bolt", "TimesFM-2.0"},
    "ett":                     {"Chronos-Bolt", "TimesFM-2.0", "Moirai-2",
                                "TinyTimeMixer-R2", "Time-MoE", "Sundial"},
    "electricity":             {"Chronos-Bolt", "TimesFM-2.0", "Moirai-2",
                                "TinyTimeMixer-R2", "Time-MoE", "Sundial"},
    "uk_dale":                 {"Chronos-Bolt", "Moirai-2"},
    "individual_household_power": {"Chronos-Bolt", "Moirai-2"},
    "uci_power":               {"Chronos-Bolt", "Moirai-2"},
    # Weather / climate ----------------------------------------------------
    "weather":                 {"Chronos-Bolt", "TimesFM-2.0", "Moirai-2", "Time-MoE", "Sundial"},
    # Traffic / mobility ---------------------------------------------------
    "traffic":                 {"Chronos-Bolt", "TimesFM-2.0", "Moirai-2",
                                "TinyTimeMixer-R2", "Time-MoE", "Sundial"},
    # Other classics -------------------------------------------------------
    "exchange_rate":           {"TimesFM-2.0", "Moirai-2", "Sundial"},
    "ili":                     {"TimesFM-2.0", "Moirai-2", "TinyTimeMixer-R2"},
    "monash":                  {"Chronos-Bolt", "TimesFM-2.0", "Moirai-2",
                                "TinyTimeMixer-R2", "Time-MoE", "Sundial"},
    "lotsa":                   {"Moirai-2"},
    "gift_eval":               {"Moirai-2", "TimesFM-2.0"},
}


@dataclass(frozen=True)
class LeakageHit:
    dataset: str
    aliases_matched: list[str]
    leaked_into: list[str]


def _canonicalise(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def assess_leakage(
    dataset_name: str,
    aliases: Iterable[str] = (),
) -> LeakageHit:
    """Check whether ``dataset_name`` (and any aliases) overlaps a known TSFM corpus."""
    candidates = {_canonicalise(dataset_name)} | {_canonicalise(a) for a in aliases}
    leaked: set[str] = set()
    matched: list[str] = []
    for cand in candidates:
        for known, fams in _TRAINING_CORPORA.items():
            if known in cand or cand in known:
                leaked |= fams
                matched.append(known)
                break
    return LeakageHit(
        dataset=dataset_name,
        aliases_matched=sorted(set(matched)),
        leaked_into=sorted(leaked),
    )


def leakage_report(
    datasets: dict[str, list[str]],
    foundation_models: list[str] | None = None,
) -> pd.DataFrame:
    """Build a tidy DataFrame: dataset × foundation-model = '⚠️' / '✓' / '?'.

    Parameters
    ----------
    datasets
        Mapping ``{display_name: [aliases...]}``.
    foundation_models
        Restrict the comparison columns. ``None`` ⇒ every TSFM we know.
    """
    all_fms = sorted({fm for fams in _TRAINING_CORPORA.values() for fm in fams})
    if foundation_models is None:
        foundation_models = all_fms
    rows = []
    for ds_name, aliases in datasets.items():
        hit = assess_leakage(ds_name, aliases)
        row = {"Dataset": ds_name}
        for fm in foundation_models:
            if fm in hit.leaked_into:
                row[fm] = "leaked"
            elif fm in all_fms:
                row[fm] = "clean"
            else:
                row[fm] = "unknown"
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["assess_leakage", "leakage_report", "LeakageHit"]

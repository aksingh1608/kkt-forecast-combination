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

"""Dataset loading and preprocessing utilities."""

from .loader import load_uci_household_power, load_generic_csv
from .preprocessing import (
    clean_dataset,
    resample_dataset,
    train_test_split_temporal,
    DatasetSchema,
    infer_schema,
)
from .extra_loaders import (
    list_external_datasets, load_external, maybe_download, download_all_external,
)
from .weather import (
    attach_weather, fetch_weather, meteostat_available, SCEAUX_LAT_LON,
)

__all__ = [
    "load_uci_household_power",
    "load_generic_csv",
    "clean_dataset",
    "resample_dataset",
    "train_test_split_temporal",
    "DatasetSchema",
    "infer_schema",
    "list_external_datasets",
    "load_external",
    "maybe_download",
    "download_all_external",
    "attach_weather", "fetch_weather", "meteostat_available", "SCEAUX_LAT_LON",
]

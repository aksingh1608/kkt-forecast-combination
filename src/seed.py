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

"""Single seed helper for reproducible CPU runs.

Call :func:`set_global_seed` once at process start (scripts and smoke tests).
Default seed is ``src.config.RANDOM_STATE`` (42).
"""

from __future__ import annotations

import os
import random

import numpy as np

from src.config import RANDOM_STATE


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and common library RNGs. Returns the seed used."""
    s = int(RANDOM_STATE if seed is None else seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    try:
        import torch

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass
    return s


__all__ = ["set_global_seed"]

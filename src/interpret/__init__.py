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

"""Model interpretability — SHAP, permutation importance, partial dependence."""

from .shap_helpers import shap_summary, shap_values_for_tree_model
from .permutation import permutation_importance_tabular
from .pdp import partial_dependence_grid

__all__ = [
    "shap_summary",
    "shap_values_for_tree_model",
    "permutation_importance_tabular",
    "partial_dependence_grid",
]

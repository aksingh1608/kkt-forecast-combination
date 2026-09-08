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

"""Compute & sustainability profiling utilities."""

from .profile import (
    estimate_param_count,
    measure_inference_latency,
    accuracy_per_second,
    co2_emissions_estimate,
    gpu_memory_tracker,
)
from .device import (
    DeviceInfo, detect_device, lightning_accelerator, vram_budget_ok,
)

__all__ = [
    "estimate_param_count",
    "measure_inference_latency",
    "accuracy_per_second",
    "co2_emissions_estimate",
    "gpu_memory_tracker",
    "DeviceInfo",
    "detect_device",
    "lightning_accelerator",
    "vram_budget_ok",
]

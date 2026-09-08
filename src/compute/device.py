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

"""GPU/CPU device discovery helpers.

Used by the optional deep-learning forecasters (NeuralForecast wrapper,
Chronos zero-shot) to auto-pick the right accelerator without forcing
the rest of the codebase to depend on torch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Summary of the local accelerator (if any)."""

    backend: str           # "cuda", "mps", or "cpu"
    name: str              # "NVIDIA GeForce RTX 3050 …" or "CPU"
    n_devices: int
    vram_gb: float | None
    supports_amp: bool     # mixed-precision (fp16/bf16) recommended

    def is_gpu(self) -> bool:
        return self.backend in {"cuda", "mps"}


def detect_device(prefer: str | None = None) -> DeviceInfo:
    """Detect the best available accelerator.

    ``prefer`` may be ``"cpu"``, ``"cuda"``, ``"mps"`` or ``None``
    (auto). Falls back to CPU if the preferred backend is unavailable.
    """
    # Allow operators to disable GPU via env var (useful in CI).
    if os.environ.get("FORCE_CPU") == "1":
        return DeviceInfo("cpu", "CPU (FORCE_CPU=1)", 1, None, False)

    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", "CPU (torch not installed)", 1, None, False)

    if (prefer in (None, "cuda")) and torch.cuda.is_available():
        idx = 0
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        # Trade a hair of fp32 precision for ~2x matmul throughput on
        # Tensor-Core GPUs (Ampere+). Safe for forecasting workloads.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return DeviceInfo(
            backend="cuda", name=name,
            n_devices=torch.cuda.device_count(),
            vram_gb=round(vram, 2),
            supports_amp=True,
        )

    if (prefer in (None, "mps")) and getattr(torch.backends, "mps", None) \
            and torch.backends.mps.is_available():
        return DeviceInfo(
            backend="mps", name="Apple Silicon (MPS)",
            n_devices=1, vram_gb=None, supports_amp=False,
        )

    return DeviceInfo("cpu", "CPU", 1, None, False)


def lightning_accelerator(info: DeviceInfo | None = None) -> tuple[str, int, str]:
    """Map a :class:`DeviceInfo` to PyTorch-Lightning constructor args.

    Returns ``(accelerator, devices, precision)`` ready to drop into a
    Lightning Trainer or NeuralForecast model.
    """
    info = info or detect_device()
    if info.backend == "cuda":
        # Use bf16 if the GPU supports it (Ampere+), else 16-mixed.
        try:
            import torch
            major = torch.cuda.get_device_capability(0)[0]
            precision = "bf16-mixed" if major >= 8 else "16-mixed"
        except Exception:  # pragma: no cover
            precision = "16-mixed"
        return "gpu", 1, precision
    if info.backend == "mps":
        return "mps", 1, "32-true"
    return "cpu", 1, "32-true"


def vram_budget_ok(min_gb: float = 1.5) -> bool:
    """Return True if we have at least ``min_gb`` of VRAM available."""
    info = detect_device()
    if info.backend != "cuda" or info.vram_gb is None:
        return False
    return info.vram_gb >= min_gb

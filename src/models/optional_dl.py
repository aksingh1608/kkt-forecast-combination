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

"""Optional deep-learning forecasters — auto-detected at import time.

If `neuralforecast` (Nixtla) is installed, we expose
:class:`NeuralForecastWrapper` covering NHITS, NBEATS, PatchTST, LSTM,
GRU, TFT and iTransformer through a single sklearn-compatible interface.

If `chronos-forecasting` is installed, we expose
:class:`ChronosZeroShot` for zero-shot foundation-model forecasting.

When neither is installed, this module is silently a no-op — the rest
of the codebase continues to work with the classical / ML / FAHE stack.

GPU support
-----------
Every wrapper here calls :func:`src.compute.detect_device` and
auto-routes to CUDA / MPS / CPU. On a small (≤ 6 GB) GPU we enable
mixed-precision (bf16 on Ampere+, fp16 on older cards) so the models
fit comfortably without sacrificing accuracy. Override with the
``FORCE_CPU=1`` environment variable.

Enable instructions
-------------------
::

    # GPU (CUDA 12.x):
    pip install --index-url https://download.pytorch.org/whl/cu128 torch

    # CPU-only (smaller download):
    pip install --index-url https://download.pytorch.org/whl/cpu torch

    pip install neuralforecast              # all NHITS / NBEATS / TFT / …
    pip install chronos-forecasting          # foundation models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import ForecastModel
from ..compute.device import detect_device, lightning_accelerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------
def neuralforecast_available() -> bool:
    try:
        import neuralforecast  # noqa: F401
        return True
    except ImportError:
        return False


def chronos_available() -> bool:
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# NeuralForecast wrapper (LSTM / NBEATS / NHITS / PatchTST / TFT / iTrans.)
# ---------------------------------------------------------------------------
@dataclass
class NeuralForecastWrapper(ForecastModel):
    """Wraps a single Nixtla NeuralForecast model.

    The supported architectures are: ``nhits``, ``nbeats``, ``patchtst``,
    ``lstm``, ``gru``, ``tft``, ``itransformer``. Default is ``nhits`` —
    the fastest and most reliable on residential energy data.
    """

    name: str = "NHITS"
    family: str = "deep-learning"
    arch: str = "nhits"
    requires_features: bool = False
    horizon: int = 24                # *rolling chunk* size (not the test length)
    input_size: int = 168            # context window
    max_steps: int = 200
    seasonal_periods: int = 24
    accelerator: str | None = None     # "auto" | "cpu" | "gpu" | "mps"; None => auto
    batch_size: int = 32
    # Covariate columns to feed as `futr_exog_list` (known into the future).
    # Pass e.g. ['wx_temp', 'wx_rhum'] to make weather available to TFT,
    # iTransformer, NHITS, etc. None ⇒ pure univariate.
    futr_exog_cols: tuple[str, ...] | None = None

    nf_: Any | None = field(default=None, init=False)
    target_name_: str = field(default="y", init=False)
    device_used_: str = field(default="cpu", init=False)
    actual_horizon_: int = field(default=24, init=False)
    actual_input_size_: int = field(default=168, init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "NeuralForecastWrapper":
        if not neuralforecast_available():
            raise ImportError("Install `neuralforecast` to use NeuralForecastWrapper.")
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, NBEATS, PatchTST, LSTM, GRU
        # TFT / iTransformer / VanillaTransformer / TimesNet are present in
        # newer NF versions; import lazily so we can advertise them when
        # available without breaking older installs.
        registry: dict[str, Any] = {
            "nhits": NHITS, "nbeats": NBEATS, "patchtst": PatchTST,
            "lstm": LSTM, "gru": GRU,
        }
        for opt_name in ("TFT", "iTransformer", "VanillaTransformer",
                         "TimesNet", "Informer", "Autoformer", "FEDformer"):
            try:
                mod = __import__("neuralforecast.models", fromlist=[opt_name])
                registry[opt_name.lower()] = getattr(mod, opt_name)
            except (ImportError, AttributeError):
                pass

        cls = registry.get(self.arch.lower())
        if cls is None:
            raise ValueError(
                f"Arch {self.arch!r} not supported by this NeuralForecast "
                f"version. Available: {sorted(registry)}"
            )

        self.target_name_ = y_train.name or "y"
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": y_train.index,
            "y": y_train.astype(float).values,
        })
        # ---- Optional future-known exogenous covariates -------------------
        if self.futr_exog_cols and X_train is not None:
            present = [c for c in self.futr_exog_cols if c in X_train.columns]
            for col in present:
                df[col] = X_train[col].astype(float).values
            self._futr_exog_used_ = tuple(present)
        else:
            self._futr_exog_used_ = ()

        # ---- Safe-size the model so it cannot exceed the training set. ----
        n_train = len(y_train)
        # Reserve at least input_size + horizon timestamps for one batch and
        # leave a small safety margin (32 steps) for NF's internal val_size.
        max_input = max(24, (n_train - 32) // 2 - self.horizon)
        safe_input = min(self.input_size, max_input)
        # Some NF models also require input_size + h <= n_train; double-check.
        safe_input = min(safe_input, max(24, n_train - self.horizon - 32))
        if safe_input < 24:
            safe_input = max(8, n_train // 4)
        safe_horizon = min(self.horizon, max(1, n_train // 4))
        self.actual_input_size_ = int(safe_input)
        self.actual_horizon_ = int(safe_horizon)

        # Auto-route to whichever device is available.
        info = detect_device()
        accel, devices, precision = lightning_accelerator(info)
        self.device_used_ = info.name

        # On a small (≤6 GB) GPU, keep batch size modest.
        bs = self.batch_size
        if info.backend == "cuda" and (info.vram_gb or 99) < 6:
            bs = min(bs, 16)

        trainer_kwargs = dict(
            accelerator=accel, devices=devices, precision=precision,
            enable_progress_bar=False, enable_model_summary=False,
            logger=False,
        )
        # val_check_steps=max_steps disables intra-training validation
        # (we have an outer test set already).
        # Some NeuralForecast archs (TimesNet) don't yet support bf16; some
        # (iTransformer, FEDformer, Informer, Autoformer) need ``n_series``;
        # patch the trainer / model kwargs per-arch so they instantiate cleanly.
        if self.arch.lower() == "timesnet" and precision == "bf16-mixed":
            trainer_kwargs["precision"] = "16-mixed"
        kwargs = dict(
            h=safe_horizon, input_size=safe_input,
            max_steps=self.max_steps, scaler_type="standard",
            batch_size=bs, val_check_steps=self.max_steps,
            **trainer_kwargs,
        )
        if self.arch.lower() in {"itransformer", "fedformer",
                                 "informer", "autoformer"}:
            kwargs["n_series"] = 1
        if self._futr_exog_used_:
            kwargs["futr_exog_list"] = list(self._futr_exog_used_)
        # NeuralForecast model classes vary slightly in accepted kwargs;
        # progressively drop the optional ones if rejected.
        drop_sequence = (
            (),
            ("val_check_steps",),
            ("batch_size",),
            ("scaler_type",),
            ("logger",),
            ("val_check_steps", "batch_size"),
            ("val_check_steps", "batch_size", "scaler_type"),
            ("val_check_steps", "batch_size", "scaler_type", "logger"),
        )
        model = None
        last_err: Exception | None = None
        for drop in drop_sequence:
            try:
                kw = {k: v for k, v in kwargs.items() if k not in drop}
                model = cls(**kw)
                break
            except TypeError as exc:
                last_err = exc
                continue
        if model is None:
            # Last resort — bare-minimum kwargs.
            try:
                model = cls(h=safe_horizon, input_size=safe_input,
                            max_steps=self.max_steps)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Could not instantiate {self.arch}: {exc} "
                    f"(last TypeError={last_err})"
                ) from exc

        freq = pd.infer_freq(y_train.index) or "h"
        self.nf_ = NeuralForecast(models=[model], freq=freq)
        # Pass val_size=0 to skip the internal validation split.
        try:
            self.nf_.fit(df, val_size=0)
        except TypeError:
            self.nf_.fit(df)
        self._train_y_ = y_train.copy()
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        if self.nf_ is None:
            raise RuntimeError("Not fitted.")
        forecasts: list[np.ndarray] = []
        history = pd.DataFrame({
            "unique_id": "series",
            "ds": self._train_y_.index,
            "y": self._train_y_.values,
        })
        freq = pd.infer_freq(history["ds"]) or "h"
        step_delta = history["ds"].iloc[-1] - history["ds"].iloc[-2]
        steps_done = 0
        max_iters = n_periods + self.actual_horizon_  # hard safety cap
        it = 0
        while steps_done < n_periods and it < max_iters:
            it += 1
            fcst = self.nf_.predict(df=history)
            yhat = np.asarray(fcst.iloc[:, -1].values, dtype=float)
            take = min(len(yhat), n_periods - steps_done)
            if take <= 0:
                break
            forecasts.append(yhat[:take])
            steps_done += take
            new_idx = pd.date_range(
                start=history["ds"].iloc[-1] + step_delta,
                periods=take,
                freq=freq,
            )
            history = pd.concat([history, pd.DataFrame({
                "unique_id": "series", "ds": new_idx, "y": yhat[:take]
            })], ignore_index=True)
        if not forecasts:
            return np.full(n_periods, np.nan)
        return np.concatenate(forecasts)[:n_periods]


# ---------------------------------------------------------------------------
# Chronos foundation model (zero-shot)
# ---------------------------------------------------------------------------
@dataclass
class ChronosZeroShot(ForecastModel):
    """Amazon's Chronos-Bolt foundation model — zero-shot forecasting."""

    name: str = "Chronos-Bolt-Small"
    family: str = "foundation"
    requires_features: bool = False
    model_id: str = "amazon/chronos-bolt-small"
    seasonal_periods: int = 24
    device: str | None = None     # None => auto, else "cuda" / "cpu" / "mps"
    dtype: str = "auto"           # "auto" picks bf16 on CUDA, fp32 elsewhere

    pipe_: Any | None = field(default=None, init=False)
    history_: pd.Series | None = field(default=None, init=False)
    device_used_: str = field(default="cpu", init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "ChronosZeroShot":
        if not chronos_available():
            raise ImportError("Install `chronos-forecasting` to use ChronosZeroShot.")
        # Chronos exposes BaseChronosPipeline.from_pretrained which handles
        # both vanilla Chronos (T5) and Chronos-Bolt models — pick the right
        # class automatically.
        try:
            from chronos import BaseChronosPipeline as _Pipe
        except ImportError:
            from chronos import ChronosPipeline as _Pipe  # type: ignore

        import torch

        info = detect_device()
        self.device_used_ = info.name
        if self.device is not None:
            target_device = self.device
        else:
            target_device = info.backend if info.backend != "mps" else "cpu"

        if self.dtype == "auto":
            if target_device == "cuda":
                # Ampere+ supports bf16; older cards fall back to fp16.
                try:
                    major = torch.cuda.get_device_capability(0)[0]
                    torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
                except Exception:
                    torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            torch_dtype = getattr(torch, self.dtype, torch.float32)

        # Newer transformers prefer ``dtype=``; older ones still accept
        # ``torch_dtype=``. Try the modern signature first and fall back.
        try:
            self.pipe_ = _Pipe.from_pretrained(
                self.model_id, device_map=target_device, dtype=torch_dtype,
            )
        except TypeError:
            self.pipe_ = _Pipe.from_pretrained(
                self.model_id, device_map=target_device, torch_dtype=torch_dtype,
            )
        self.history_ = y_train.copy()
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        if self.pipe_ is None or self.history_ is None:
            raise RuntimeError("Not fitted.")
        import torch

        ctx = torch.tensor(self.history_.astype(float).values, dtype=torch.float32)

        # Chronos has a max prediction_length per call (64 for Bolt-Small).
        # Roll out in chunks if the user asks for more.
        max_chunk = 64
        forecasts: list[np.ndarray] = []
        history_tensor = ctx
        remaining = n_periods
        while remaining > 0:
            take = min(remaining, max_chunk)
            try:
                fcst = self.pipe_.predict(
                    context=history_tensor, prediction_length=take,
                )
            except TypeError:
                fcst = self.pipe_.predict(history_tensor, take)

            if hasattr(fcst, "numpy"):
                arr = fcst.detach().cpu().to(torch.float32).numpy()
            else:
                arr = np.asarray(fcst, dtype=float)
            if arr.ndim == 3:
                arr = arr[0, arr.shape[1] // 2]   # median quantile
            elif arr.ndim == 2:
                arr = arr[0]

            forecasts.append(arr[:take].astype(float))
            history_tensor = torch.cat([
                history_tensor,
                torch.tensor(arr[:take], dtype=torch.float32),
            ])
            remaining -= take
        return np.concatenate(forecasts)[:n_periods]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _pick_chronos_model_id() -> str:
    """Pick the largest Chronos-Bolt that fits comfortably in VRAM.

    On a 4 GB GPU we use ``tiny``, on 6 GB ``mini``, on 8 GB+ ``small``
    (still leaves room for activations + the rest of the benchmark).
    """
    info = detect_device()
    if info.backend != "cuda" or info.vram_gb is None:
        return "amazon/chronos-bolt-tiny"
    if info.vram_gb >= 10:
        return "amazon/chronos-bolt-base"
    if info.vram_gb >= 6:
        return "amazon/chronos-bolt-small"
    if info.vram_gb >= 4:
        return "amazon/chronos-bolt-mini"
    return "amazon/chronos-bolt-tiny"


def _supported_nf_archs() -> set[str]:
    """Return the set of NeuralForecast architectures importable in this env."""
    if not neuralforecast_available():
        return set()
    out = set()
    for name in ("NHITS", "NBEATS", "PatchTST", "LSTM", "GRU", "TFT",
                 "iTransformer", "VanillaTransformer", "TimesNet",
                 "Informer", "Autoformer", "FEDformer"):
        try:
            mod = __import__("neuralforecast.models", fromlist=[name])
            getattr(mod, name)
            out.add(name.lower())
        except (ImportError, AttributeError):
            pass
    return out


def build_optional_dl_zoo(
    *, seasonal_periods: int = 24, horizon: int = 24,
    max_steps: int = 200,
    input_size: int | None = None,
    include_foundation: bool = True,
    include_archs: tuple[str, ...] | None = None,
) -> list[ForecastModel]:
    """Return whatever optional models are installed.

    ``horizon`` is the **rolling-chunk size** the wrapper trains on, not the
    full test length — the wrapper rolls the chunked forecast forward to
    cover any number of test points.
    """
    supported = _supported_nf_archs()
    info = detect_device()
    if include_archs is None:
        # Default zoo: every transformer / DL family the install supports.
        preferred = ["nhits", "nbeats", "patchtst", "lstm",
                     "tft", "itransformer", "vanillatransformer", "timesnet"]
        # TFT has a huge memory footprint (~3 GB VRAM on a 168-context
        # window). Skip on small GPUs to avoid the OOM that bricks the run.
        if info.backend == "cuda" and (info.vram_gb or 99) < 6:
            preferred = [a for a in preferred if a != "tft"]
        archs = tuple(a for a in preferred if a in supported)
    else:
        archs = tuple(a for a in include_archs if a in supported)

    eff_input = input_size if input_size is not None else max(168, horizon * 4)

    out: list[ForecastModel] = []
    if neuralforecast_available():
        for arch in archs:
            display = {
                "vanillatransformer": "VanillaTransformer",
                "itransformer": "iTransformer",
                "timesnet": "TimesNet",
            }.get(arch, arch.upper())
            out.append(NeuralForecastWrapper(
                name=display,
                arch=arch,
                horizon=horizon,
                input_size=eff_input,
                max_steps=max_steps,
                seasonal_periods=seasonal_periods,
            ))
    if include_foundation:
        if chronos_available():
            model_id = _pick_chronos_model_id()
            nice_name = "Chronos-Bolt-" + model_id.split("-")[-1].capitalize()
            out.append(ChronosZeroShot(
                name=nice_name, model_id=model_id,
                seasonal_periods=seasonal_periods,
            ))
    return out

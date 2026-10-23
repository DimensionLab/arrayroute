# SPDX-License-Identifier: Apache-2.0
"""Channel-wise float32 transforms with explicit execution and buffer ownership."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

NormalizationEngine = Literal["numpy", "numexpr"]


def _backend(engine: str) -> Any:
    if engine == "numpy":
        return np
    if engine != "numexpr":
        raise ValueError(f"Unknown normalization engine: {engine!r}")
    try:
        return importlib.import_module("numexpr")
    except ImportError as exc:
        raise ImportError(
            "NumExpr normalization requires the optional arrayroute[expressions] extra"
        ) from exc


def normalization_backend_info(engine: str) -> dict[str, Any]:
    """Describe this process's engine without changing its global thread settings."""
    backend = _backend(engine)
    return {
        "engine": engine,
        "version": backend.__version__,
        "dtype": "float32",
        "operation": "(values - mean) / scale",
        "optimization": "moderate" if engine == "numexpr" else None,
        "threads": int(backend.get_num_threads()) if engine == "numexpr" else 1,
        "thread_scope": "process" if engine == "numexpr" else "caller",
    }


def normalize_channels(
    values: np.ndarray,
    mean: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray,
    *,
    out: np.ndarray | None = None,
    engine: NormalizationEngine = "numpy",
) -> np.ndarray:
    """Apply recorded statistics to a channels-last native float32 array.

    ``out`` must be contiguous, writable and either identical to ``values`` or
    disjoint from it. The caller owns in-place buffers; this function cannot
    establish whether a storage reader has lent an array to other consumers.
    Parameters are snapshotted before writing, including when they alias output.

    NumExpr is optional. Set NUMEXPR_NUM_THREADS before process startup; its pool
    is process-global and is never reconfigured here. Arithmetic is subtraction
    followed by division, without reciprocal or aggressive expression rewrites.
    Cross-platform bit identity still requires validation of the deployed stack.
    """
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32"):
        raise ValueError("Normalization requires a native float32 array")
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("Normalization requires a nonempty channel axis")
    channels = values.shape[-1]
    means = np.array(mean, dtype=np.float32, copy=True)
    scales = np.array(scale, dtype=np.float32, copy=True)
    if means.shape != (channels,) or scales.shape != (channels,):
        raise ValueError("Mean and scale must have one value per channel")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Means must be finite and scales must be finite and positive")
    if out is not None:
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != values.dtype
            or out.shape != values.shape
        ):
            raise ValueError("Output must match input shape and native float32 dtype")
        if not out.flags.writeable or not (out.flags.c_contiguous or out.flags.f_contiguous):
            raise ValueError("Output must be writable and contiguous")
        if out is not values and np.may_share_memory(values, out):
            raise ValueError("Output must be the input itself or a disjoint buffer")
    backend = _backend(engine)
    if out is None:
        out = np.empty(values.shape, dtype=np.float32)
    if engine == "numpy":
        np.subtract(values, means, out=out)
        np.divide(out, scales, out=out)
    elif values.size:
        backend.evaluate(
            "(values - mean) / scale",
            local_dict={"values": values, "mean": means, "scale": scales},
            global_dict={},
            out=out,
            optimization="moderate",
            casting="no",
        )
    return out

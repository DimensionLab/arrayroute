# SPDX-License-Identifier: Apache-2.0
"""Optional tensor transforms; importing ArrayRoute never imports PyTorch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class TorchChannelNormalizer:
    """Recorded float32 statistics on the consumer's CPU or CUDA stream.

    Input storage is never modified. Statistics are copied at construction and
    operations run on the caller's current stream. No GPU work belongs in forked
    loader workers. A caller handing the result to another stream must establish
    its own wait/record_stream relationship, as for ordinary PyTorch operations.

    Separate subtraction and division match the NumPy operation order. This is
    reproducible within a fixed runtime, not a cross-device bit-identity promise.
    Do not compile/fuse this transform when comparing that arithmetic contract.
    """

    def __init__(
        self, mean: Sequence[float], scale: Sequence[float], *, device: str = "cpu"
    ) -> None:
        import torch

        means = np.array(mean, dtype=np.float32, copy=True)
        scales = np.array(scale, dtype=np.float32, copy=True)
        if (means.ndim != 1 or means.size == 0 or scales.shape != means.shape
                or not np.isfinite(means).all() or not np.isfinite(scales).all()
                or (scales <= 0).any()):
            raise ValueError("Finite channel means and positive finite channel scales are required")
        selected = torch.device(device)
        if selected.type not in {"cpu", "cuda"}:
            raise ValueError("This transform supports CPU and CUDA tensors")
        # Blocking construction makes parameters safe for a subsequent consumer stream.
        self._mean = torch.tensor(means, device=selected)
        self._scale = torch.tensor(scales, device=selected)
        if selected.type == "cuda":
            torch.cuda.current_stream(selected).synchronize()
        self.device = self._mean.device
        self.channels = int(means.size)
        self._statistics = {"mean": tuple(means.tolist()), "scale": tuple(scales.tolist())}

    def __call__(self, values: Any, *, channel_axis: int = -1) -> Any:
        import torch

        if not isinstance(values, torch.Tensor) or values.dtype != torch.float32:
            raise ValueError("Normalization requires a float32 tensor")
        if values.device != self.device:
            raise ValueError(f"Expected input on {self.device}; transfer it explicitly first")
        if (isinstance(channel_axis, bool) or not isinstance(channel_axis, int)
                or not -values.ndim <= channel_axis < values.ndim):
            raise ValueError("Channel axis is outside the tensor rank")
        axis = channel_axis % values.ndim
        if values.shape[axis] != self.channels:
            raise ValueError("Channel count differs from recorded statistics")
        shape = [1] * values.ndim
        shape[axis] = self.channels
        result = values.clone(memory_format=torch.preserve_format)
        result.sub_(self._mean.reshape(shape))
        result.div_(self._scale.reshape(shape))
        return result

    def describe(self) -> dict[str, Any]:
        import torch

        return {
            "engine": "torch", "version": torch.__version__, "device": str(self.device),
            "dtype": "float32", "operation": "(values - mean) / scale",
            "statistics": {key: list(value) for key, value in self._statistics.items()},
            "stream": "caller", "input_modified": False,
            "determinism": "fixed-runtime; cross-device equivalence requires validation",
        }

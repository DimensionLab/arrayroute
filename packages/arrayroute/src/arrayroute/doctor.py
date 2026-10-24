# SPDX-License-Identifier: Apache-2.0
"""Explicit local CPU/CUDA and torchrun collective checks, without allocating cloud compute."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import numpy as np


def probe(
    *, cuda: bool = False, distributed: bool = False, expected_nodes: int = 1,
    collective_bytes: Sequence[int] | None = None, collective_iterations: int = 10,
) -> dict[str, Any]:
    import torch

    from .normalization import normalize_channels
    from .torch_transforms import TorchChannelNormalizer

    if collective_bytes is not None:
        from .collective_probe import validate_request

        validate_request(collective_bytes, collective_iterations)
        if not distributed:
            raise ValueError("Collective measurements require --distributed")

    if expected_nodes < 1 or (not distributed and expected_nodes != 1):
        raise ValueError("Multiple nodes require a distributed probe")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed and world < 2:
        raise ValueError("Launch a distributed check with torchrun and at least two ranks")
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        torch.cuda.set_device(device)
    backend = "nccl" if cuda else "gloo"
    if distributed and cuda and not torch.distributed.is_nccl_available():
        raise RuntimeError(
            "This PyTorch build has no NCCL; use Linux/WSL2 for the CUDA cluster check")
    values = np.random.default_rng(371).normal(size=(7, 11, 3)).astype("float32")
    means, scales = [0.25, -0.75, 1.3], [0.4, 2.1, 9.7]
    transform = TorchChannelNormalizer(means, scales, device=str(device))
    tensor = torch.tensor(values, device=device)
    first, second = transform(tensor), transform(tensor)
    if not torch.equal(first, second):
        raise RuntimeError("Fixed-runtime normalization replay failed")
    reference = normalize_channels(values, means, scales)
    actual = first.cpu().numpy()
    np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-7)
    report: dict[str, Any] = {
        "platform": platform.system(), "platform_release": platform.release(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "cuda_device_count": torch.cuda.device_count() if cuda else None,
        "nccl_available": torch.distributed.is_nccl_available(),
        "normalization_replay_exact": True,
        "max_numpy_difference": float(np.max(np.abs(actual - reference))),
        "host": socket.gethostname(), "world_size": world,
        "rank": int(os.environ.get("RANK", "0")),
        "scope": "Runtime smoke check, not a throughput or physical-node attestation",
    }
    if distributed:
        if torch.distributed.is_initialized():
            raise RuntimeError("Run the doctor in its own process, outside an existing group")
        torch.distributed.init_process_group(backend, timeout=timedelta(seconds=60))
        try:
            marker = torch.tensor(report["rank"] + 1., device=device)
            torch.distributed.all_reduce(marker)
            if marker.item() != world * (world + 1) / 2:
                raise RuntimeError("Collective returned an incorrect sum")
            nodes: list[Any] = [None] * world
            torch.distributed.all_gather_object(nodes, {
                "host": report["host"], "group_rank": os.environ.get("GROUP_RANK", "0"),
                "collective_request": (list(collective_bytes), collective_iterations)
                if collective_bytes is not None else None})
            if len({(n["host"], n["group_rank"]) for n in nodes}) != expected_nodes:
                raise RuntimeError("Observed torchrun nodes differ from the requested topology")
            if any(n["collective_request"] != nodes[0]["collective_request"] for n in nodes):
                raise ValueError("Every rank must request identical collective measurements")
            report.update(collective_passed=True, nodes=nodes, backend=backend)
            if collective_bytes is not None:
                from .collective_probe import measure

                report["collective_measurements"] = measure(
                    collective_bytes, collective_iterations, device)
                report["transport_settings"] = {k: os.environ[k] for k in (
                    "NCCL_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY", "NCCL_NET",
                    "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "GLOO_SOCKET_IFNAME",
                ) if k in os.environ}
        finally:
            torch.distributed.destroy_process_group()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--expected-nodes", type=int, default=1)
    parser.add_argument("--collective-bytes", type=int, action="append",
                        help="Measure this all-reduce payload size; repeat for up to eight sizes")
    parser.add_argument("--collective-iterations", type=int, default=10)
    args = parser.parse_args()
    try:
        print(json.dumps({"ok": True, **probe(cuda=args.cuda, distributed=args.distributed,
                                             expected_nodes=args.expected_nodes,
                                             collective_bytes=args.collective_bytes,
                                             collective_iterations=args.collective_iterations)},
                         sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

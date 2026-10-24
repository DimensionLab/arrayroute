# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicitly requested transport measurements for an initialized process group."""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_request(sizes: Sequence[int], iterations: int) -> tuple[int, ...]:
    sizes = tuple(sizes)
    if not sizes or len(sizes) > 8 or len(set(sizes)) != len(sizes):
        raise ValueError("Choose one to eight distinct collective payload sizes")
    if any(type(n) is not int or n < 4 or n > 64 << 20 or n % 4 for n in sizes):
        raise ValueError("Collective payloads must be multiples of 4 bytes, from 4 B to 64 MiB")
    if type(iterations) is not int or not 3 <= iterations <= 100:
        raise ValueError("Collective iterations must be an integer from 3 to 100")
    return sizes


def measure(sizes: Sequence[int], iterations: int, device: Any) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    sizes = validate_request(sizes, iterations)
    world = dist.get_world_size()
    if world < 2:
        raise ValueError("Collective measurements require at least two ranks")
    requests: list[Any] = [None] * world
    dist.all_gather_object(requests, (sizes, iterations))
    if any(request != requests[0] for request in requests):
        raise ValueError("Every rank must request identical collective sizes and iterations")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for size in sizes:
        payload = torch.empty(size // 4, dtype=torch.float32, device=device)
        elapsed_max = torch.empty((), dtype=torch.float64, device=device)
        durations = []
        for iteration in range(iterations + 2):
            payload.fill_(1)
            sync()
            dist.barrier()
            sync()
            start = time.perf_counter()
            dist.all_reduce(payload)
            sync()
            elapsed = time.perf_counter() - start
            if not torch.all(payload == world).item():
                raise RuntimeError("Collective payload verification failed")
            elapsed_max.fill_(elapsed)
            dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
            if iteration >= 2:
                durations.append(elapsed_max.item())
        median = float(np.median(durations))
        rows.append({"payload_bytes": size, "iterations": iterations, "warmup": 2,
                     "seconds_per_collective_max_rank": durations,
                     "p50_seconds": median, "p95_seconds": float(np.percentile(durations, 95)),
                     "payload_bytes_per_second": size / median,
                     "payload_verified": True})
        del payload, elapsed_max
    return {"operation": "all_reduce_sum_float32", "world_size": world, "rows": rows,
            "scope": "Synchronized wall time, maximum across ranks; barriers and validation "
                     "excluded. Payload rate is algorithm bandwidth, not physical link bandwidth. "
                     "Not a storage or end-to-end training benchmark."}

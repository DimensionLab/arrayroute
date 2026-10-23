# SPDX-License-Identifier: Apache-2.0
"""Sample resident memory across a process and its current descendants."""

from __future__ import annotations

import math
import threading
import time
from typing import Any


class ProcessTreeMemory:
    """Measure in a separate pass from throughput benchmarks.

    RSS sums can count shared pages more than once. Samples are sequential, not
    atomic, and can miss peaks or short-lived children between observations. This
    is neither a memory limit nor a measurement of unique physical memory. The
    monitor includes its own small overhead when it observes the current process.
    One instance supports one context, with bounded retained measurement state.
    """

    def __init__(self, pid: int | None = None, *, interval_s: float = 0.02) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Memory sample interval must be finite and positive")
        try:
            import psutil
        except ImportError as exc:
            raise ImportError("Install arrayroute[profiling] to measure process memory") from exc
        self._psutil = psutil
        self._root = psutil.Process(pid)
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._used = False
        self._samples = self._denied = self._vanished = 0
        self._peak = self._baseline = self._max_processes = 0
        self._started = self._elapsed = 0.0
        self._error: BaseException | None = None

    def _sample(self) -> None:
        total, count = 0, 0
        try:
            # children() guards against root PID reuse; retain Process instances
            # for each observation so a vanished child is not replaced by a new PID.
            processes = [self._root, *self._root.children(recursive=True)]
        except self._psutil.NoSuchProcess:
            self._vanished += 1
            return
        except self._psutil.AccessDenied:
            self._denied += 1
            processes = [self._root]
        for process in processes:
            try:
                if not process.is_running():
                    self._vanished += 1
                    continue
                total += process.memory_info().rss
                count += 1
            except self._psutil.NoSuchProcess:
                self._vanished += 1
            except self._psutil.AccessDenied:
                self._denied += 1
        if not self._samples:
            self._baseline = total
        self._samples += 1
        self._peak = max(self._peak, total)
        self._max_processes = max(self._max_processes, count)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self._interval):
                self._sample()
        except BaseException as exc:
            self._error = exc

    def __enter__(self) -> ProcessTreeMemory:
        if self._used:
            raise RuntimeError("Memory monitor contexts cannot be reused")
        self._used = True
        self._started = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="arrayroute-memory", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        self._elapsed = time.perf_counter() - self._started
        if self._error is not None and exc[0] is None:
            raise RuntimeError("Memory sampling failed") from self._error

    def report(self) -> dict[str, Any]:
        if self._thread is None or self._thread.is_alive():
            raise RuntimeError("Close the memory monitor before reading its report")
        return {
            "metric": "sampled_sum_process_tree_rss_bytes",
            "baseline_bytes": self._baseline,
            "peak_bytes": self._peak,
            "peak_over_baseline_bytes": max(0, self._peak - self._baseline),
            "max_processes": self._max_processes,
            "samples": self._samples,
            "interval_seconds": self._interval,
            "elapsed_seconds": self._elapsed,
            "access_denied_observations": self._denied,
            "vanished_process_observations": self._vanished,
            "sampling_failed": self._error is not None,
            "scope": "Sequential RSS samples of root and current descendants; shared pages "
                     "may be counted multiple times; transient peaks/processes may be missed. "
                     "Includes runtime, caches and monitor overhead; excludes device memory.",
        }

# SPDX-License-Identifier: Apache-2.0
"""Measure ordered normalization on a real, local channel-last Zarr array."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from .memory import peak_process_rss_bytes
from .statistics import CanonicalChannelMoments, ChannelMoments


def main() -> None:
    import zarr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error("output already exists")
    array = zarr.open_array(str(args.store), mode="r")
    if array.ndim < 2 or array.dtype.kind != "f":
        parser.error("expected a time-first, channel-last floating-point array")
    metadata = (args.store / "zarr.json").read_bytes()
    record = {
        "schema": 1, "store": str(args.store.resolve()),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "shape": list(array.shape), "dtype": str(array.dtype),
        "decoded_bytes_per_trial": int(np.prod(array.shape)) * array.dtype.itemsize,
        "numpy": np.__version__, "zarr": zarr.__version__, "platform": platform.platform(),
        "algorithm": CanonicalChannelMoments.ALGORITHM,
        "statistics_source_sha256": hashlib.sha256(
            Path(__file__).with_name("statistics.py").read_bytes()
        ).hexdigest(),
        "cache_state": "uncontrolled local filesystem cache; trials rotated",
        "trials": [],
    }
    baseline = None
    canonical_bits = None
    input_hash = None
    trials = record["trials"]
    configurations = [("legacy", 1), ("canonical", 1), ("canonical", 3)]
    for repeat in range(args.repeats):
        order = configurations[repeat % 3:] + configurations[:repeat % 3]
        for algorithm, frames in order:
            moments = (ChannelMoments if algorithm == "legacy" else CanonicalChannelMoments)(
                array.shape[-1]
            )
            digest = hashlib.sha256()
            read_s = reduce_s = hash_s = 0.0
            start = time.perf_counter()
            for first in range(0, array.shape[0], frames):
                before = time.perf_counter()
                block = np.asarray(array[first : first + frames])
                read_s += time.perf_counter() - before
                before = time.perf_counter()
                digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
                hash_s += time.perf_counter() - before
                before = time.perf_counter()
                moments.update(block)
                reduce_s += time.perf_counter() - before
                del block
            before = time.perf_counter()
            result = np.concatenate((moments.mean, moments.variance))
            reduce_s += time.perf_counter() - before
            elapsed_s = time.perf_counter() - start
            if baseline is None:
                baseline = result.copy()
            np.testing.assert_allclose(result, baseline, rtol=1e-10, atol=1e-12)
            bits = result.tobytes()
            if algorithm == "canonical":
                if canonical_bits is not None and bits != canonical_bits:
                    raise RuntimeError("Canonical statistics failed byte-for-byte replay")
                canonical_bits = bits
            if input_hash is not None and digest.hexdigest() != input_hash:
                raise RuntimeError("Input data changed between trials")
            input_hash = digest.hexdigest()
            if (args.store / "zarr.json").read_bytes() != metadata:
                raise RuntimeError("Array metadata changed during measurement")
            trial = {
                "repeat": repeat, "method": algorithm, "read_frames": frames,
                "read_s": read_s, "reduce_s": reduce_s, "hash_s": hash_s,
                "elapsed_s": elapsed_s, "input_sha256": input_hash,
                "result_sha256": hashlib.sha256(bits).hexdigest(),
                "mean_variance": result.tolist(),
                "max_abs_difference_from_legacy": float(np.max(np.abs(result - baseline))),
                "process_peak_rss_bytes": peak_process_rss_bytes(),
            }
            trials.append(trial)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
            print(json.dumps(trial), flush=True)


if __name__ == "__main__":
    main()

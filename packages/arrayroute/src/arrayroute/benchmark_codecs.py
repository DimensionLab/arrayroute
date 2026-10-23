# SPDX-License-Identifier: Apache-2.0
"""Lossless codec experiments on synthetic or supplied arrays with exact verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PATTERNS = ("velocity", "noisy-velocity", "random-bits", "phase-mask", "connectivity")
CPU_CODECS = ("zstd", "blosc-zstd-shuffle", "blosc-zstd-bitshuffle", "blosc-lz4-bitshuffle")
GPU_CODECS = ("gpu-lz4", "gpu-lz4-bitshuffle", "gpu-bitcomp", "gpu-ans", "gpu-gdeflate")


def fixture(pattern: str, side: int) -> np.ndarray:
    """Analytic inputs, not solver outputs or a surrogate accuracy benchmark."""
    if side < 2:
        raise ValueError("side must be at least 2")
    rng = np.random.default_rng(73)
    if pattern == "random-bits":
        return rng.integers(0, 2**32, size=(side, side, side, 3), dtype=np.uint32)
    if pattern == "connectivity":
        # Two triangles on every XY quad at each Z layer, using flattened vertex ids.
        base = np.arange(side**3, dtype=np.uint32).reshape(side, side, side)[:, :-1, :-1]
        return np.stack(
            (base, base + 1, base + side, base + 1, base + side + 1, base + side), axis=-1
        ).reshape(-1, 3)
    z, y, x = np.ogrid[:side, :side, :side]
    if pattern == "phase-mask":
        sphere = (x - side / 2) ** 2 + (y - side / 2) ** 2 + (z - side / 2) ** 2
        return np.ascontiguousarray(
            np.where(sphere < (side / 5) ** 2, 2, np.broadcast_to(z < side / 2, sphere.shape)),
            dtype=np.uint8,
        )
    if pattern not in ("velocity", "noisy-velocity"):
        raise ValueError(pattern)
    x, y, z = (axis.astype(np.float32) * (2 * np.pi / side) for axis in (x, y, z))
    velocity = np.empty((side, side, side, 3), dtype=np.float32)
    velocity[..., 0] = np.sin(x) * np.cos(y) * np.cos(z)
    velocity[..., 1] = -np.cos(x) * np.sin(y) * np.cos(z)
    velocity[..., 2] = 0
    for wave in (3, 7, 13):
        velocity[..., 0] += np.sin(wave * y + x) * np.cos(z) / wave**2
        velocity[..., 1] += np.cos(wave * z + y) * np.sin(x) / wave**2
        velocity[..., 2] += np.sin(wave * x + z) * np.cos(y) / wave**2
    if pattern == "noisy-velocity":
        velocity += rng.standard_normal(velocity.shape, dtype=np.float32) * np.float32(0.01)
    return velocity


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def cpu_codec(name: str, threads: int):
    import numcodecs

    numcodecs.blosc.set_nthreads(threads)
    if name == "zstd":
        return numcodecs.Zstd(level=1)
    shuffle = {
        "blosc-zstd-shuffle": numcodecs.Blosc.SHUFFLE,
        "blosc-zstd-bitshuffle": numcodecs.Blosc.BITSHUFFLE,
        "blosc-lz4-bitshuffle": numcodecs.Blosc.BITSHUFFLE,
    }[name]
    cname = "lz4" if name == "blosc-lz4-bitshuffle" else "zstd"
    return numcodecs.Blosc(cname=cname, clevel=1, shuffle=shuffle, blocksize=65536)


def verify(source: np.ndarray, restored: object) -> None:
    actual = np.frombuffer(restored, dtype=np.uint8)  # type: ignore[arg-type]
    if not np.array_equal(source.view(np.uint8).reshape(-1), actual):
        raise ValueError("Lossless reconstruction changed source bytes")


def timed(operation, synchronize=lambda: None):
    synchronize()
    started = time.perf_counter()
    result = operation()
    synchronize()
    return result, time.perf_counter() - started


def load_input(path: Path, *, max_bytes: int) -> np.ndarray:
    """Map a numeric NPY without pickle and reject oversized input before copying."""
    if max_bytes < 1:
        raise ValueError("Input byte allowance must be positive")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value, np.ndarray):
        value.close()
        raise ValueError("Input must be one numeric NPY array, not an archive")
    if value.dtype.kind not in "biuf" or value.size == 0 or value.ndim == 0:
        raise ValueError("Input must be a nonempty real numeric array with at least one axis")
    if value.nbytes > max_bytes:
        raise MemoryError(f"Input needs {value.nbytes} bytes; allowance is {max_bytes}")
    return np.ascontiguousarray(value)


def measure(
    name: str,
    pattern: str,
    side: int,
    repeats: int,
    threads: int,
    *,
    input_path: Path | None = None,
    max_input_bytes: int = 256 << 20,
    axis_order: tuple[int, ...] | None = None,
) -> dict:
    source = (
        fixture(pattern, side)
        if input_path is None
        else load_input(input_path, max_bytes=max_input_bytes)
    )
    source_hash = digest(source)
    axes = tuple(range(source.ndim)) if axis_order is None else axis_order
    if any(type(axis) is not int for axis in axes) or sorted(axes) != list(range(source.ndim)):
        raise ValueError("axis_order must be a permutation of all input axes")
    if name in GPU_CODECS and axes != tuple(range(source.ndim)):
        raise ValueError("Axis permutation is currently measured only for CPU codecs")
    inverse = tuple(axes.index(axis) for axis in range(source.ndim))
    packed_shape = tuple(source.shape[axis] for axis in axes)
    records = []
    versions = {"numpy": np.__version__}
    hardware = {}
    if name in CPU_CODECS:
        codec = cpu_codec(name, threads)
        versions["numcodecs"] = importlib.metadata.version("numcodecs")
        options = codec.get_config()

        def encode():
            packed = np.ascontiguousarray(source.transpose(axes))
            return codec.encode(packed)

        def decode(encoded):
            packed = np.frombuffer(codec.decode(encoded), dtype=source.dtype).reshape(packed_shape)
            return np.ascontiguousarray(packed.transpose(inverse))

        for iteration in range(repeats + 1):
            encoded, encode_s = timed(encode)
            restored, decode_s = timed(lambda encoded=encoded: decode(encoded))
            verify(source, restored)
            if iteration:
                records.append(
                    {
                        "host_encode_s": encode_s,
                        "host_decode_s": decode_s,
                        "encoded_bytes": len(encoded),
                    }
                )
    elif name in GPU_CODECS:
        cp = importlib.import_module("cupy")
        nvcomp = importlib.import_module("nvidia.nvcomp")

        options = {
            "algorithm": {
                "gpu-lz4": "LZ4",
                "gpu-lz4-bitshuffle": "LZ4",
                "gpu-bitcomp": "Bitcomp",
                "gpu-ans": "ANS",
                "gpu-gdeflate": "GDeflate",
            }[name],
            "uncomp_chunk_size": 65536,
        }
        if name == "gpu-bitcomp":
            options["data_type"] = source.dtype.str
        if name == "gpu-lz4-bitshuffle":
            # LZ4's bitshuffle accepts integer words, not floating point type tags.
            options["data_type"] = np.dtype(f"<u{source.dtype.itemsize}").str
            options["bitshuffle_mode"] = nvcomp.BitshuffleMode.NVCOMP_BITSHUFFLE_LSB_FIRST
        codec = nvcomp.Codec(**options)
        synchronize = cp.cuda.runtime.deviceSynchronize
        versions.update(nvcomp=nvcomp.__version__, cupy=cp.__version__)
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        hardware = {
            "gpu": properties["name"].decode(),
            "gpu_total_bytes": properties["totalGlobalMem"],
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        }
        options = {k: str(v) if not isinstance(v, (str, int)) else v for k, v in options.items()}
        options.update(bitstream="NVCOMP_NATIVE", execution_policy="default (CONCURRENCY)")
        device = cp.asarray(source)
        synchronize()

        def host_encode():
            uploaded = cp.asarray(source)
            synchronize()  # CuPy and nvCOMP may own different streams.
            compressed = codec.encode(nvcomp.as_array(uploaded))
            synchronize()
            return np.asarray(compressed.cpu())

        def host_decode(encoded):
            uploaded = cp.asarray(encoded)
            synchronize()
            decoded = codec.decode(nvcomp.as_array(uploaded))
            synchronize()
            return np.asarray(decoded.cpu())

        for iteration in range(repeats + 1):
            encoded_device, encode_s = timed(
                lambda: codec.encode(nvcomp.as_array(device)), synchronize
            )
            restored_device, decode_s = timed(
                lambda encoded_device=encoded_device: codec.decode(encoded_device), synchronize
            )
            verify(source, np.asarray(restored_device.cpu()))
            encoded_host, host_encode_s = timed(host_encode, synchronize)
            restored_host, host_decode_s = timed(
                lambda encoded_host=encoded_host: host_decode(encoded_host), synchronize
            )
            verify(source, restored_host)
            if iteration:
                records.append(
                    {
                        "device_encode_s": encode_s,
                        "device_decode_s": decode_s,
                        "host_encode_s": host_encode_s,
                        "host_decode_s": host_decode_s,
                        "encoded_bytes": encoded_host.nbytes,
                    }
                )
    else:
        raise ValueError(name)
    if digest(source) != source_hash:
        raise ValueError("Source bytes changed during codec measurement")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "codec": name,
        "pattern": pattern if input_path is None else "input",
        "side": side if input_path is None else None,
        "shape": source.shape,
        "dtype": source.dtype.str,
        "source_bytes": source.nbytes,
        "source_sha256": source_hash,
        "axis_order": axes,
        "packed_shape": packed_shape,
        "layout_timing": "CPU encode/decode include packing and inverse packing to C order",
        "byte_identical": True,
        "verified_roundtrips": (repeats + 1) * (2 if name in GPU_CODECS else 1),
        "options": options,
        "versions": versions,
        "hardware": hardware,
        "blosc_threads": threads,
        "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "measurements": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--input", type=Path, help="Immutable numeric NPY input instead of fixtures"
    )
    parser.add_argument(
        "--max-input-mib",
        type=int,
        default=256,
        help="Maximum decoded input MiB; codec scratch/output are additional",
    )
    parser.add_argument(
        "--axis-order",
        type=int,
        nargs="+",
        help="CPU-only lossless axis permutation; both rearrangements are timed",
    )
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--process-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS)
    parser.add_argument(
        "--codecs", nargs="+", choices=CPU_CODECS + GPU_CODECS, default=list(CPU_CODECS)
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.input is not None and args.patterns is not None:
        parser.error("Choose --input or --patterns")
    args.patterns = args.patterns or list(PATTERNS)
    if args.max_input_mib < 1:
        parser.error("--max-input-mib must be positive")
    if args.side < 2 or min(args.repeats, args.process_repeats, args.threads) < 1:
        parser.error("side must be >= 2; counts must be positive")
    if args.worker:
        print(
            json.dumps(
                measure(
                    args.codecs[0],
                    args.patterns[0],
                    args.side,
                    args.repeats,
                    args.threads,
                    input_path=args.input,
                    max_input_bytes=args.max_input_mib << 20,
                    axis_order=tuple(args.axis_order) if args.axis_order is not None else None,
                )
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    expected_input = None
    if args.input is not None:
        source = load_input(args.input, max_bytes=args.max_input_mib << 20)
        expected_input = {
            "source_sha256": digest(source),
            "shape": list(source.shape),
            "dtype": source.dtype.str,
        }
        del source
    args.output.mkdir(parents=True, exist_ok=False)
    runs, failures = [], []
    jobs = [
        (pattern, codec)
        for pattern in (["input"] if args.input else args.patterns)
        for codec in args.codecs
    ]
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for repeat in range(args.process_repeats):
        for pattern, name in jobs if repeat % 2 == 0 else list(reversed(jobs)):
            output = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arrayroute.benchmark_codecs",
                    "--worker",
                    "--codecs",
                    name,
                    *(
                        ["--axis-order", *map(str, args.axis_order)]
                        if args.axis_order is not None
                        else []
                    ),
                    *(
                        [
                            "--input",
                            str(args.input.resolve()),
                            "--max-input-mib",
                            str(args.max_input_mib),
                        ]
                        if args.input
                        else ["--patterns", pattern]
                    ),
                    "--side",
                    str(args.side),
                    "--repeats",
                    str(args.repeats),
                    "--threads",
                    str(args.threads),
                ],
                capture_output=True,
                text=True,
                timeout=600,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": str(args.threads),
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
            stem = f"{repeat}-{pattern}-{name}"
            (args.output / f"{stem}.stderr.txt").write_text(output.stderr)
            if output.returncode:
                failures.append(
                    {
                        "process_repeat": repeat,
                        "pattern": pattern,
                        "codec": name,
                        "returncode": output.returncode,
                        "stderr": output.stderr,
                    }
                )
                print(f"FAILED {stem}", flush=True)
                continue
            result = {"process_repeat": repeat, **json.loads(output.stdout)}
            if expected_input is not None and any(
                result[key] != value for key, value in expected_input.items()
            ):
                failures.append(
                    {
                        "process_repeat": repeat,
                        "codec": name,
                        "error": "Input changed between codec processes",
                    }
                )
                continue
            (args.output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
            runs.append(result)
            print(f"Verified {stem}", flush=True)
    report = {
        "method": (
            "Supplied immutable NPY input. "
            if args.input
            else "Synthetic lossless codec benchmark, not solver output. "
        )
        + "Each codec/pattern uses "
        "a fresh process per process-repeat, one untimed warm-up, then repeated warm timings. "
        "Hash and byte comparisons are outside clocks. Device clocks synchronize before/after "
        "encode/decode and exclude host transfers; host clocks include transfers, allocations "
        "and encode or decode. No disk/network I/O is timed. Encoded sizes include codec headers. "
        "CPU timings include any requested axis packing and inverse packing to C order. "
        "RSS includes fixture creation, imports and verification. No peak GPU-memory claim.",
        "benchmark_sha256": source_hash,
        "input_identity": expected_input,
        "axis_order": args.axis_order,
        "input_limit_bytes": args.max_input_mib << 20 if args.input else None,
        "python": sys.version,
        "platform": platform.platform(),
        "runs": runs,
        "failures": failures,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Compare serial/two-slot reads at matched tiles and budgets with a numerical consumer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .plan import ArraySpec, tiles
from .reader import ArrayReader, ReadStats
from .session import LocalReadSession


def reference_digest(spec: ArraySpec, budget: int) -> str:
    """Regenerate source values, independently of any storage reader, before timing."""
    digest = hashlib.sha256()
    for tile in tiles(spec, (slice(None),)*len(spec.shape), buffer_bytes=budget):
        selection = tile.plan.selection
        assert all(isinstance(s, slice) for s in selection)
        time_slice = selection[0]
        assert isinstance(time_slice, slice)
        expected = np.empty(tile.plan.output_shape, dtype=np.float32)
        for i, frame in enumerate(range(time_slice.start, time_slice.stop, time_slice.step)):
            values = np.random.default_rng(742 + frame).standard_normal(
                spec.shape[1:], dtype=np.float32
            )
            expected[i] = values[selection[1:]]
            del values
        digest.update(str(tile.destination).encode())
        digest.update(memoryview(expected).cast('B'))
    return digest.hexdigest()


def run(root: Path, *, edge: int, frames: int, repeats: int, rounds: int) -> None:
    import zarr
    from zarr.codecs import ZstdCodec

    root.mkdir(parents=True, exist_ok=False)
    shape = (frames, edge, edge, edge, 3)
    array = zarr.create_array(
        store=str(root/'field.zarr'), shape=shape,
        chunks=(1, min(32, edge), min(32, edge), min(32, edge), 3),
        shards=(1, edge, edge, edge, 3), dtype='float32',
        compressors=[ZstdCodec(level=3)],
    )
    hashes = []
    for frame in range(frames):
        value = np.random.default_rng(742 + frame).standard_normal(shape[1:], dtype=np.float32)
        array[frame] = value
        hashes.append(hashlib.sha256(memoryview(value).cast('B')).hexdigest())
        del value
    for frame, expected in enumerate(hashes):
        value = np.ascontiguousarray(array[frame])
        if hashlib.sha256(memoryview(value).cast('B')).hexdigest() != expected:
            raise ValueError('Fixture differs from generated data')
        del value
    metadata = {
        'shape': shape, 'frame_sha256': hashes, 'unique_bytes': int(np.prod(shape))*4,
        'rounds': rounds, 'repeats': repeats, 'platform': platform.platform(),
        'serial_tile_budget': 32 << 20, 'prefetch_two_slot_budget': 64 << 20,
        'memory_scope': 'Reader output/decoded estimate; excludes consumer scratch/backend caches',
        'cache_state': 'OS uncontrolled; validated fixture was read before timing',
        'versions': {n: importlib.metadata.version(n) for n in
                     ('numpy', 'zarr', 'tensorstore', 'arrayroute')},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))},
        'scope': 'Synthetic compressed array; CPU nonlinear reduction, not CFD validation',
    }
    spec = ArraySpec(shape, tuple(array.chunks), 4)
    references = {budget: reference_digest(spec, budget << 20) for budget in (32, 64)}
    metadata['independent_tile_digests_by_budget_mib'] = references
    (root/'method.json').write_text(json.dumps(metadata, indent=2))
    work = [(rep, engine, mode) for rep in range(repeats)
            for engine in ('zarr-python', 'tensorstore')
            for mode in ('serial', 'serial-full', 'prefetch')]
    random.Random(17).shuffle(work)
    (root/'order.json').write_text(json.dumps(work, indent=2))
    reports = []
    for rep, engine, mode in work:
        session = LocalReadSession(engine, tensorstore_threads=2, immutable=True)
        backend = session.open(root/'field.zarr', '').backend
        reader = ArrayReader(backend, buffer_bytes=(32 if mode == 'serial' else 64) << 20)
        tile_budget = 64 if mode == 'serial-full' else 32
        selection = (slice(None),)*len(shape)
        stats = ReadStats()
        digest = hashlib.sha256()
        total = 0.
        tiles = 0
        context = (closing(reader.stream(selection, stats=stats)) if mode != 'prefetch'
                   else reader.prefetch(selection, stats=stats))
        began = time.perf_counter()
        with context as blocks:
            for tile, values in blocks:
                digest.update(str(tile.destination).encode())
                digest.update(memoryview(np.ascontiguousarray(values)).cast('B'))
                scratch = values.copy()
                for _ in range(rounds):
                    np.sin(scratch, out=scratch)
                    np.square(scratch, out=scratch)
                total += float(scratch.sum(dtype=np.float64))
                tiles += 1
                del scratch, values
        elapsed = time.perf_counter()-began
        if digest.hexdigest() != references[tile_budget]:
            raise ValueError('Read bytes/order differ from regenerated reference')
        report = {'rep': rep, 'engine': engine, 'mode': mode, 'elapsed_s': elapsed,
                  'digest': digest.hexdigest(), 'reduction': total, 'tiles': tiles,
                  'read_stats': asdict(stats)}
        reports.append(report)
        with (root/'results.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        session.clear()
    if not all(np.isclose(r['reduction'], reports[0]['reduction'], rtol=1e-12, atol=0)
               for r in reports):
        raise ValueError('Consumer reduction changed beyond floating summation tolerance')
    (root/'complete.json').write_text(json.dumps({'verified': True, 'reports': len(reports)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--edge', type=int, default=128)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=8)
    args = parser.parse_args()
    if args.edge < 32 or args.edge % 32 or min(args.frames, args.repeats) < 1 or args.rounds < 0:
        parser.error('edge must be a positive multiple of 32; frames/repeats > 0; rounds >= 0')
    run(args.root, edge=args.edge, frames=args.frames, repeats=args.repeats, rounds=args.rounds)


if __name__ == '__main__':
    main()

# SPDX-License-Identifier: Apache-2.0
"""Measure ordered queue payloads with uneven buffers and a paced consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from .execution import ordered_map


def run_trial(policy: str, unit_bytes: int) -> dict:
    if policy not in {'count', 'bytes'} or unit_bytes < 1:
        raise ValueError('Choose count/bytes admission and a positive buffer unit')
    sizes = [unit_bytes * n for n in [1, 4, 2, 6] * 4]
    expected = []
    for i, size in enumerate(sizes):
        digest = hashlib.sha256()
        block = bytes([i]) * min(size, 65536)
        whole, remainder = divmod(size, len(block))
        for _ in range(whole):
            digest.update(block)
        digest.update(block[:remainder])
        expected.append(digest.hexdigest())

    def factory():
        return lambda i: np.full(sizes[i], i, dtype=np.uint8)

    digests = []
    tracemalloc.start()
    start = time.perf_counter()
    with ordered_map(factory, range(len(sizes)), workers=4, max_pending=8,
                     max_pending_bytes=unit_bytes * 8 if policy == 'bytes' else None,
                     estimate_bytes=(lambda i: sizes[i]) if policy == 'bytes' else None) as outputs:
        for value in outputs:
            digests.append(hashlib.sha256(memoryview(value)).hexdigest())
            time.sleep(.01)
            del value
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if digests != expected:
        raise ValueError('Ordered payload bytes differ from the independent reference')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'policy': policy, 'unit_bytes': unit_bytes, 'payload_bytes': sum(sizes),
        'planned_queue_bytes': unit_bytes * 8 if policy == 'bytes' else None,
        'seconds': seconds, 'peak_traced_bytes': peak,
        'peak_rss_bytes': rss if sys.platform == 'darwin' else rss * 1024,
        'digest_sha256': hashlib.sha256(json.dumps(digests).encode()).hexdigest(),
        'verified': True, 'workers': 4, 'max_pending': 8, 'consumer_delay_seconds': .01,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--unit-mib', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', choices=['count', 'bytes'])
    args = parser.parse_args()
    if args.unit_mib < 1 or args.repeats < 1:
        parser.error('unit-mib and repeats must be positive')
    if args.worker:
        print(json.dumps(run_trial(args.worker, args.unit_mib * (1 << 20))))
        return
    if args.output is None or args.output.exists():
        parser.error('Choose a new --output path')
    rows = []
    for repeat in range(args.repeats):
        policies = ['count', 'bytes']
        random.Random(20261023 + repeat).shuffle(policies)
        for policy in policies:
            completed = subprocess.run(
                [sys.executable, '-m', 'arrayroute.benchmark_admission',
                 '--worker', policy, '--unit-mib', str(args.unit_mib)],
                capture_output=True, text=True, check=True, timeout=300,
            )
            rows.append({'repeat': repeat, **json.loads(completed.stdout)})
    if len({row['digest_sha256'] for row in rows}) != 1:
        raise ValueError('Policies or repeats changed payload bytes/order')
    report = {'scope': 'Synthetic variable-buffer queue; not scientific I/O or training speed',
              'platform': platform.platform(), 'python': platform.python_version(),
              'numpy': np.__version__, 'trials': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

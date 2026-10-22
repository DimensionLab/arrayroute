# SPDX-License-Identifier: Apache-2.0
"""Bounded access traces and held-out byte-capacity cache policy screening.

Replay models serial, exact-selection decoded payloads, not disk or network latency.
It never reads array values from a saved trace or installs a policy automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .plan import ReadPlan, Selection, plan
from .reader import Backend

_MAX_EVENTS = 1_000_000
_MAX_FILE_BYTES = 128 << 20


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(f'{name} must be an integer >= {minimum} and < 2**63')
    return value


@dataclass(frozen=True)
class Access:
    key: str
    payload_bytes: int

    def __post_init__(self):
        if not isinstance(self.key, str) or re.fullmatch('[0-9a-f]{64}', self.key) is None:
            raise ValueError('Trace keys must be SHA-256 hex digests')
        _integer(self.payload_bytes, 'Payload size')


@dataclass(frozen=True)
class AccessTrace:
    accesses: tuple[Access, ...]
    attempted: int
    failures: int = 0
    peak_concurrency: int = 1

    def __post_init__(self):
        if not isinstance(self.accesses, tuple) or len(self.accesses) > _MAX_EVENTS:
            raise ValueError('Provide a bounded tuple of accesses')
        if not all(isinstance(event, Access) for event in self.accesses):
            raise ValueError('Invalid trace access')
        _integer(self.attempted, 'Attempted reads')
        _integer(self.failures, 'Failed reads')
        _integer(self.peak_concurrency, 'Peak concurrency', 1)
        if self.failures + len(self.accesses) > self.attempted:
            raise ValueError('Trace counters are inconsistent')

    @property
    def complete(self) -> bool:
        return self.failures == 0 and len(self.accesses) == self.attempted

    def to_dict(self) -> dict:
        return {'schema': 1, **asdict(self)}

    def save(self, path: str | Path) -> None:
        _save_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> AccessTrace:
        with Path(path).open('rb') as stream:
            content = stream.read(_MAX_FILE_BYTES + 1)
        if len(content) > _MAX_FILE_BYTES:
            raise ValueError('Trace exceeds the 128 MiB input limit')
        value = json.loads(content)
        expected = {'schema', 'accesses', 'attempted', 'failures', 'peak_concurrency'}
        if (not isinstance(value, dict) or set(value) != expected
                or type(value['schema']) is not int):
            raise ValueError('Invalid trace schema')
        if value['schema'] != 1 or not isinstance(value['accesses'], list):
            raise ValueError('Unsupported trace schema')
        if len(value['accesses']) > _MAX_EVENTS:
            raise ValueError('Trace contains too many accesses')
        events = []
        for row in value['accesses']:
            if not isinstance(row, dict) or set(row) != {'key', 'payload_bytes'}:
                raise ValueError('Invalid trace event')
            events.append(Access(**row))
        return cls(tuple(events), value['attempted'], value['failures'], value['peak_concurrency'])


class TraceRecorder:
    """Capture successful reads in invocation order, bounded by max_events.

    One recorder belongs to one process. Shared threads are supported and marked;
    replay cannot recommend a policy from overlapping reads. Trace files contain
    hashed identities and lengths, never array values, paths, credentials or selections.
    Hashes and access patterns are pseudonymous metadata, not anonymization.
    """

    def __init__(self, *, max_events: int = 100_000):
        _integer(max_events, 'Maximum events', 1)
        if max_events > _MAX_EVENTS:
            raise ValueError('Maximum events cannot exceed 1000000')
        self.max_events = max_events
        self._events: list[Access | None] = []
        self._attempted = self._failed = self._active = 0
        self._peak = 1
        self._lock = threading.Lock()

    def _begin(self) -> int | None:
        with self._lock:
            self._attempted += 1
            self._active += 1
            self._peak = max(self._peak, self._active)
            if len(self._events) >= self.max_events:
                return None
            token = len(self._events)
            self._events.append(None)
            return token

    def _end(self, token: int | None, event: Access | None) -> None:
        with self._lock:
            self._active -= 1
            self._failed += int(event is None)
            if token is not None:
                self._events[token] = event

    def snapshot(self) -> AccessTrace:
        with self._lock:
            if self._active:
                raise RuntimeError('Finish outstanding reads before taking a trace snapshot')
            return AccessTrace(tuple(x for x in self._events if x is not None),
                               self._attempted, self._failed, self._peak)

    def backend(self, source: Backend, *, identity: str) -> TracedBackend:
        return TracedBackend(source, self, identity=identity)


class TracedBackend:
    """Wrap the outermost backend so cache hits and misses are both observed.

    Identity must distinguish immutable source revision and representation, including
    units/preprocessing. A trace cannot prove that caller-supplied identity is correct.
    """

    def __init__(self, source: Backend, recorder: TraceRecorder, *, identity: str):
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 4096:
            raise ValueError('Provide a nonempty immutable source/representation identity')
        if isinstance(source, TracedBackend):
            raise ValueError('Do not wrap an already traced backend')
        self.source, self.recorder = source, recorder
        self.spec, self.dtype = source.spec, np.dtype(source.dtype)
        if self.dtype.kind not in 'biufc' or self.spec.itemsize != self.dtype.itemsize:
            raise ValueError('Tracing requires fixed-width numeric or boolean arrays')
        self._identity = hashlib.sha256(json.dumps(
            {'identity': identity, 'spec': asdict(self.spec), 'dtype': self.dtype.str},
            sort_keys=True).encode()).hexdigest()

    def read(self, selection: Selection) -> np.ndarray:
        return self.read_plan(plan(self.spec, selection))

    def read_plan(self, request: ReadPlan) -> np.ndarray:
        selection = [s if isinstance(s, int) else [s.start, s.stop, s.step]
                     for s in request.selection]
        key = hashlib.sha256(json.dumps([self._identity, selection]).encode()).hexdigest()
        token = self.recorder._begin()
        event = None
        try:
            execute = getattr(self.source, 'read_plan', None)
            value = execute(request) if execute is not None else self.source.read(request.selection)
            if value.shape != request.output_shape or value.dtype != self.dtype:
                raise ValueError('Source returned an unexpected shape or dtype')
            event = Access(key, value.nbytes)
            return value
        finally:
            self.recorder._end(token, event)


@dataclass
class Traffic:
    reads: int = 0
    hits: int = 0
    requested_bytes: int = 0
    source_bytes: int = 0
    payload_write_bytes: int = 0
    bypasses: int = 0

    def score(self) -> tuple[int, int]:
        return self.source_bytes, self.payload_write_bytes


def _replay(trace: AccessTrace, policy: str, capacity: int, max_entry: int,
            overhead: int, split: int) -> tuple[Traffic, Traffic]:
    cache: OrderedDict[str, int] = OrderedDict()
    used = 0
    train, validation = Traffic(), Traffic()
    for position, event in enumerate(trace.accesses):
        stats = train if position < split else validation
        size = event.payload_bytes
        stats.reads += 1
        stats.requested_bytes += size
        if event.key in cache:
            stats.hits += 1
            if policy == 'lru':
                cache.move_to_end(event.key)
            continue
        stats.source_bytes += size
        if policy == 'none':
            continue
        charge = size + overhead
        if not 0 < size <= max_entry or charge > capacity:
            stats.bypasses += 1
            continue
        if policy == 'retain' and used + charge > capacity:
            continue
        while used + charge > capacity:
            _, removed = cache.popitem(last=False)
            used -= removed
        cache[event.key] = charge
        used += charge
        stats.payload_write_bytes += size
    return train, validation


def compare_cache_policies(trace: AccessTrace, *, capacities_bytes: tuple[int, ...],
                           max_entry_bytes: int, training_reads: int,
                           entry_overhead_bytes: int = 0) -> dict:
    """Choose on the prefix; reject regressions against LRU on the held-out suffix.

    Capacities are usable payload+declared entry-overhead budgets. They exclude
    filesystem/database overhead. Cache state continues into validation. Source bytes
    are decoded payload bytes, not compressed remote traffic. No latency prediction.
    """
    if not trace.complete:
        raise ValueError('Trace is truncated or contains failed reads; capture a complete pilot')
    if not capacities_bytes or len(capacities_bytes) > 64:
        raise ValueError('Provide between 1 and 64 capacities')
    capacities = sorted({_integer(c, 'Capacity', 1) for c in capacities_bytes})
    _integer(max_entry_bytes, 'Maximum entry bytes', 1)
    _integer(entry_overhead_bytes, 'Entry overhead bytes')
    _integer(training_reads, 'Training reads', 1)
    if training_reads >= len(trace.accesses):
        raise ValueError('Reserve at least one held-out read after the training prefix')
    sizes: dict[str, int] = {}
    for event in trace.accesses:
        if event.key in sizes and sizes[event.key] != event.payload_bytes:
            raise ValueError('One immutable selection key has inconsistent payload sizes')
        sizes[event.key] = event.payload_bytes
    rows = []
    for capacity in capacities:
        policies = {}
        scores = {}
        for policy in ('none', 'lru', 'retain'):
            train, validation = _replay(trace, policy, capacity, max_entry_bytes,
                                        entry_overhead_bytes, training_reads)
            policies[policy] = {'training': asdict(train), 'validation': asdict(validation)}
            scores[policy] = train.score(), validation.score()
        # Ties favor the uncached baseline, then the current general-purpose default.
        selected = min(scores, key=lambda p: scores[p][0])
        validation_passed = scores[selected][1] <= scores['lru'][1]
        serial = trace.peak_concurrency == 1
        rows.append({'capacity_bytes': capacity, 'policies': policies,
                     'selected_on_training': selected,
                     'validation_passed': validation_passed,
                     'recommendation_status': ('overlapping-reads' if not serial else
                                              'validated' if validation_passed else
                                              'validation-regression'),
                     'recommendation': selected if serial and validation_passed else None})
    return {'schema': 1, 'trace_sha256': hashlib.sha256(json.dumps(
                trace.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'reads': len(trace.accesses), 'unique_selections': len(sizes),
            'unique_payload_bytes': sum(sizes.values()), 'training_reads': training_reads,
            'peak_concurrency': trace.peak_concurrency, 'serial_replay_only': True,
            'objective': 'decoded source bytes, then payload write bytes',
            'max_entry_bytes': max_entry_bytes, 'entry_overhead_bytes': entry_overhead_bytes,
            'rows': rows}


def _save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    encoded = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    if len(encoded) > _MAX_FILE_BYTES:
        raise ValueError('Trace/report exceeds the 128 MiB output limit')
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--capacity-bytes', type=int, action='append', required=True)
    parser.add_argument('--max-entry-bytes', type=int, required=True)
    parser.add_argument('--training-reads', type=int, required=True)
    parser.add_argument('--entry-overhead-bytes', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare_cache_policies(AccessTrace.load(args.trace),
            capacities_bytes=tuple(args.capacity_bytes), max_entry_bytes=args.max_entry_bytes,
            training_reads=args.training_reads, entry_overhead_bytes=args.entry_overhead_bytes)
        _save_json(args.output, report)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('Decoded-byte estimates; verify actual latency before applying a policy.')
    print('Capacity (bytes) | Policy | Validation')
    for row in report['rows']:
        print(f"{row['capacity_bytes']} | {row['recommendation'] or 'no recommendation'}"
              f" | {row['recommendation_status']}")
    print(f'Report saved to {args.output}')


if __name__ == '__main__':
    main()

# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import random
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from arrayroute.cache_policy import Access, AccessTrace, TraceRecorder, compare_cache_policies
from arrayroute.plan import ArraySpec
from arrayroute.reader import ArrayReader


def access(key, size=1):
    return Access(hashlib.sha256(str(key).encode()).hexdigest(), size)


def trace(keys):
    events = tuple(access(key) for key in keys)
    return AccessTrace(events, len(events))


def compare(value, capacity=5, training=32, entry=100, overhead=0):
    return compare_cache_policies(value, capacities_bytes=(capacity,), max_entry_bytes=entry,
                                 training_reads=training, entry_overhead_bytes=overhead)['rows'][0]


class Source:
    def __init__(self):
        self.value = np.arange(256, dtype='>f4').reshape(16, 16)
        self.dtype = self.value.dtype
        self.spec = ArraySpec(self.value.shape, (4, 4), self.value.itemsize)

    def read(self, selection):
        return self.value[selection].copy()


def test_real_reads_normalize_identity_preserve_endian_and_do_not_store_values(tmp_path):
    source = Source()
    recorder = TraceRecorder()
    identity = 'private/source@immutable-42:pressure:Pa'
    reader = ArrayReader(recorder.backend(source, identity=identity))
    np.testing.assert_array_equal(reader.read((-1, slice(None))), source.value[-1])
    reader.read((15, slice(0, 16, 1)))
    assert recorder.snapshot().accesses[0] == recorder.snapshot().accesses[1]
    recorder.backend(source, identity=identity + ':new').read((15, slice(None)))
    assert recorder.snapshot().accesses[0] != recorder.snapshot().accesses[2]
    path = tmp_path / 'access.json'
    recorder.snapshot().save(path)
    assert identity not in path.read_text() and 'pressure' not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert AccessTrace.load(path) == recorder.snapshot()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        recorder.snapshot().save(path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['access.json']


def test_preserves_planned_reads():
    class Planned(Source):
        def read(self, selection):
            pytest.fail('Planned backend was bypassed')

        def read_plan(self, request):
            return self.value[request.selection]

    source = Planned()
    reader = ArrayReader(TraceRecorder().backend(source, identity='revision'))
    np.testing.assert_array_equal(reader.read((1, slice(None))), source.value[1])


def test_training_choice_rejects_changed_validation_workload():
    fixed = list(range(16)) * 4
    row = compare(trace(fixed), training=32)
    assert row['selected_on_training'] == row['recommendation'] == 'retain'
    assert row['policies']['retain']['validation']['hits'] == 10
    moved = list(range(16)) * 2 + [40, 41, 42, 43] * 8
    row = compare(trace(moved), training=32)
    assert row['selected_on_training'] == 'retain'
    assert row['recommendation'] is None and not row['validation_passed']
    assert row['policies']['lru']['validation']['source_bytes'] == 4


def test_no_reuse_selects_no_cache_and_empty_arrays_bypass():
    row = compare(trace(range(64)))
    assert row['recommendation'] == 'none'
    value = AccessTrace((access('zero', 0),) * 4, 4)
    row = compare(value, training=2)
    assert row['recommendation'] == 'none'
    assert row['policies']['lru']['training']['bypasses'] == 2


def test_truncation_failure_and_wrong_results_never_recommend():
    recorder = TraceRecorder(max_events=2)
    backend = recorder.backend(Source(), identity='revision')
    for _ in range(4):
        backend.read((0, slice(None)))
    result = recorder.snapshot()
    assert len(result.accesses) == 2 and result.attempted == 4 and not result.complete
    with pytest.raises(ValueError, match='truncated'):
        compare(result, training=1)

    class Broken(Source):
        def read(self, selection):
            raise OSError('source unavailable')

    recorder = TraceRecorder()
    with pytest.raises(OSError, match='source unavailable'):
        recorder.backend(Broken(), identity='revision').read((0, slice(None)))
    assert recorder.snapshot().failures == 1
    with pytest.raises(ValueError, match='failed reads'):
        compare(recorder.snapshot(), training=1)
    source = Source()
    source.value = source.value.astype('i4')
    with pytest.raises(ValueError, match='shape or dtype'):
        recorder.backend(source, identity='revision').read((0, slice(None)))
    assert recorder.snapshot().failures == 2


def test_concurrent_reads_are_counted_but_never_recommended():
    barrier = threading.Barrier(2)

    class Concurrent(Source):
        def read(self, selection):
            barrier.wait(timeout=5)
            with pytest.raises(RuntimeError, match='outstanding'):
                recorder.snapshot()
            barrier.wait(timeout=5)
            return super().read(selection)

    recorder = TraceRecorder()
    backend = recorder.backend(Concurrent(), identity='revision')
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: backend.read((0, slice(None))), range(2)))
    np.testing.assert_array_equal(results[0], results[1])
    value = recorder.snapshot()
    assert value.peak_concurrency == 2 and value.complete
    assert compare(value, training=1)['recommendation'] is None


def test_randomized_replay_matches_independent_list_reference():
    rng = random.Random(83491)
    for _ in range(100):
        sizes = [rng.randrange(0, 12) for _ in range(16)]
        keys = [rng.randrange(16) for _ in range(100)]
        events = tuple(access(key, sizes[key]) for key in keys)
        capacity, limit, overhead = rng.randrange(1, 64), rng.randrange(1, 16), rng.randrange(4)
        report = compare(AccessTrace(events, 100), capacity, 50, limit, overhead)
        for policy in ('none', 'lru', 'retain'):
            live = []
            expected = []
            for group in [keys[:50], keys[50:]]:
                hits = source = writes = bypasses = 0
                for key in group:
                    size = sizes[key]
                    if key in live:
                        hits += 1
                        if policy == 'lru':
                            live.remove(key)
                            live.append(key)
                        continue
                    source += size
                    if policy == 'none':
                        continue
                    charge = size + overhead
                    if not 0 < size <= limit or charge > capacity:
                        bypasses += 1
                        continue
                    if (policy == 'retain'
                            and sum(sizes[k] + overhead for k in live) + charge > capacity):
                        continue
                    while sum(sizes[k] + overhead for k in live) + charge > capacity:
                        live.pop(0)
                    live.append(key)
                    writes += size
                expected.append((hits, source, writes, bypasses))
            for phase, wanted in zip(('training', 'validation'), expected, strict=True):
                got = report['policies'][policy][phase]
                assert tuple(got[k] for k in ('hits', 'source_bytes', 'payload_write_bytes',
                                               'bypasses')) == wanted


@pytest.mark.parametrize('policy', ['lru', 'retain'])
def test_capture_around_real_cache_and_agreement_after_metadata_reserve(tmp_path, policy):
    from arrayroute.disk_cache import ArrayIdentity, TileCache

    source = Source()
    source.value = np.arange(10 * 65536, dtype=np.float32).reshape(10, 65536)
    source.dtype = source.value.dtype
    source.spec = ArraySpec(source.value.shape, (1, 65536), 4)
    cache = TileCache(tmp_path, capacity_bytes=2 << 20, max_entry_bytes=256 << 10,
                      admission=policy)
    cached = cache.backend(source, ArrayIdentity('source', 'rev', 'Pa'))
    recorder = TraceRecorder()
    reader = ArrayReader(recorder.backend(cached, identity='source@rev:Pa'))
    for i in list(range(10)) * 4:
        np.testing.assert_array_equal(reader.read((i, slice(None))), source.value[i])
    row = compare(recorder.snapshot(), capacity=7 * (256 << 10), training=20, entry=256 << 10)
    simulated = row['policies'][policy]
    assert sum(simulated[p]['hits'] for p in ('training', 'validation')) == cached.stats.hits
    assert (sum(simulated[p]['source_bytes'] for p in ('training', 'validation'))
            == cached.stats.source_bytes)
    cache.close()


@pytest.mark.parametrize('invalid', [True, -1, 0, 1.5, 1_000_001])
def test_recorder_bounds(invalid):
    with pytest.raises(ValueError):
        TraceRecorder(max_events=invalid)


@pytest.mark.parametrize('changes', [
    {'schema': True}, {'schema': 2}, {'attempted': True}, {'failures': -1},
    {'peak_concurrency': 0}, {'accesses': [{'key': 'x', 'payload_bytes': 4}]},
    {'extra': 'ignored?'}, {'accesses': [{'key': 'a' * 64, 'payload_bytes': True}]},
])
def test_invalid_persisted_trace(tmp_path, changes):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({**trace([1, 1]).to_dict(), **changes}))
    with pytest.raises(ValueError):
        AccessTrace.load(path)


def test_reject_inconsistent_size_and_invalid_limits():
    value = trace([1, 1, 1, 1])
    with pytest.raises(ValueError, match='inconsistent payload'):
        compare(replace(value, accesses=(access(1, 1), access(1, 2)), attempted=2), training=1)
    for capacity, training, entry, overhead in [(0, 1, 1, 0), (5, 4, 1, 0),
                                               (5, 1, True, 0), (5, 1, 1, -1)]:
        with pytest.raises(ValueError):
            compare(value, capacity, training, entry, overhead)


def test_cli_customer_workflow(tmp_path):
    path = tmp_path / 'trace.json'
    trace(list(range(16)) * 4).save(path)
    output = tmp_path / 'report.json'
    subprocess.run([sys.executable, '-m', 'arrayroute.cache_policy', str(path),
                    '--capacity-bytes', '5', '--capacity-bytes', '8', '--max-entry-bytes', '2',
                    '--training-reads', '32', '--output', str(output)], check=True)
    report = json.loads(output.read_text())
    assert [r['capacity_bytes'] for r in report['rows']] == [5, 8]
    assert all(r['recommendation'] == 'retain' for r in report['rows'])
    assert report['unique_selections'] == 16

# SPDX-License-Identifier: Apache-2.0
import asyncio
import pickle

import numpy as np
import pytest
import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore, WrapperStore

from arrayroute.ranges import RangePolicy
from arrayroute.session import LocalReadSession
from arrayroute.zarr_ranges import BoundedRangeStore

pytestmark = pytest.mark.skipif(not hasattr(zarr.abc.store.Store, 'get_ranges'),
                                reason='Bounded range adapter requires Zarr >= 3.3')

class ObservedStore(WrapperStore):
    def __init__(self, store, reads):
        super().__init__(store)
        self.reads = reads

    def _with_store(self, store):
        return type(self)(store, self.reads)

    async def get(self, key, prototype, byte_range=None):
        self.reads.append((key, byte_range))
        return await self._store.get(key, prototype, byte_range)


def test_real_partial_shards_preserve_values_and_explicit_request_limits(tmp_path):
    values = np.random.default_rng(119).standard_normal((4, 32, 32, 32), dtype='f4')
    stored = zarr.create_array(tmp_path, shape=values.shape, dtype='f4',
                               chunks=(1, 8, 8, 8), shards=(1, 32, 32, 32))
    stored[:3] = values[:3]
    values[3] = 0
    reads = []
    policy = RangePolicy(8192, 16384, max_gap_bytes=256, max_extra_bytes=1024)
    wrapper = BoundedRangeStore(ObservedStore(LocalStore(tmp_path, read_only=True), reads),
                               policy=policy)
    array = zarr.open_array(wrapper, mode='r')
    for selection in [(0, slice(1, 14, 2), slice(2, 29, 3), slice(0, 8)),
                      (3, slice(1, 14, 2), slice(2, 29, 3), slice(0, 8))]:
        assert array[selection].tobytes() == values[selection].tobytes()
    ranges = [r for key, r in reads if isinstance(r, RangeByteRequest)]
    assert ranges and max(r.end - r.start for r in ranges) <= 8192


def test_session_reopens_policy_without_global_settings(tmp_path):
    original_config = zarr.config.to_dict()
    values = np.arange(32 * 32, dtype='f4').reshape(32, 32)
    root = zarr.open_group(tmp_path, mode='w')
    root.create_array('field', data=values, chunks=(4, 4), shards=(16, 16))
    policy = RangePolicy(4096, 4096)
    session = LocalReadSession('zarr-python', range_policy=policy)
    selection = (slice(1, 8), slice(2, 10))
    assert session.read(tmp_path, 'field', selection).tobytes() == values[selection].tobytes()
    clone = pickle.loads(pickle.dumps(session))
    assert clone.range_policy == policy and not clone._readers
    assert clone.read(tmp_path, 'field', selection).tobytes() == values[selection].tobytes()
    with pytest.raises(ValueError, match='Zarr-Python'):
        LocalReadSession('zarrista', range_policy=policy)
    assert zarr.config.to_dict() == original_config


def test_sync_ranges_enforce_total_budget_before_any_io(tmp_path):
    if not hasattr(LocalStore, 'get_sync'):
        pytest.skip('Synchronous store API requires newer Zarr')
    (tmp_path / 'data').write_bytes(bytes(range(64)))
    store = BoundedRangeStore(LocalStore(tmp_path, read_only=True), policy=RangePolicy(8, 16))
    prototype = default_buffer_prototype()
    result = store.get_ranges_sync('data', [RangeByteRequest(8, 16), RangeByteRequest(0, 4)],
                                    prototype=prototype)
    assert {i: b.to_bytes() for i, b in result} == {0: bytes(range(8, 16)), 1: bytes(range(4))}
    with pytest.raises(ValueError, match='Synchronous'):
        store.get_ranges_sync('absent', [RangeByteRequest(i, i + 8) for i in (0, 16, 32)],
                              prototype=prototype)


def test_metadata_limit_and_missing_capability_are_explicit(tmp_path, monkeypatch):
    store = BoundedRangeStore(LocalStore(tmp_path), policy=RangePolicy(8, 16, max_ranges=1))
    with pytest.raises(ValueError, match='Too many'):
        store.get_ranges_sync('absent', [RangeByteRequest(0, 4)] * 2,
                              prototype=default_buffer_prototype())
    monkeypatch.delattr(zarr.abc.store.Store, 'get_ranges')
    with pytest.raises(ImportError, match='3.3'):
        BoundedRangeStore(LocalStore(tmp_path), policy=RangePolicy(8, 16))


def test_unknown_length_batches_are_rejected_and_early_close_is_safe(tmp_path):
    async def run():
        store = BoundedRangeStore(LocalStore(tmp_path), policy=RangePolicy(8, 16))
        stream = store.get_ranges('absent', [None], prototype=default_buffer_prototype())
        with pytest.raises(ValueError, match='explicit'):
            await anext(stream)
        with pytest.raises(FileNotFoundError):
            async for _ in store.get_ranges('absent', [RangeByteRequest(0, 4)],
                                            prototype=default_buffer_prototype()):
                pass
    asyncio.run(run())

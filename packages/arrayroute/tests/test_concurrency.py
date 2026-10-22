# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec
from arrayroute.concurrency import plan_concurrency


class CountedArray:
    def __init__(self, values, chunks):
        self.values = values
        self.spec = ArraySpec(values.shape, chunks, values.dtype.itemsize)
        self.dtype = values.dtype
        self.calls = 0

    def read(self, selection):
        self.calls += 1
        return self.values[selection].copy()


def test_full_frame_budget_warns_about_fragmentation():
    spec = ArraySpec((43691, 128, 128, 128, 3), (1, 32, 32, 32, 3), 4)
    selection = (0, slice(None), slice(None), slice(None), slice(None))
    result = plan_concurrency(spec, selection, total_buffer_bytes=256 << 20, max_workers=8)
    assert result.max_workers_without_tiling == 5
    assert result.untiled_buffer_bytes_per_value == 48 << 20
    candidates = {c.workers: c for c in result.candidates}
    assert candidates[4].materialization == candidates[5].materialization == "direct"
    assert candidates[8].materialization == "tiled"
    assert candidates[8].requests_per_value_lower_bound == 8
    assert candidates[8].request_count_exact


@pytest.mark.parametrize("selection", [
    (slice(None), slice(None)), (slice(1, 19, 3), slice(2, 17, 2)),
    (-1, slice(None)), (slice(0, 0), slice(None)),
])
@pytest.mark.parametrize("budget", [16, 64, 512, 4096])
def test_candidates_match_actual_reader_execution(selection, budget):
    values = np.arange(20 * 18, dtype=np.float32).reshape(20, 18)
    backend = CountedArray(values, (4, 3))
    result = plan_concurrency(backend.spec, selection, total_buffer_bytes=budget, max_workers=8)
    for candidate in result.candidates:
        backend.calls = 0
        reader = ArrayReader(backend, buffer_bytes=candidate.buffer_bytes_per_worker)
        if candidate.materialization in {"stream-required", "chunk-too-large"}:
            with pytest.raises(MemoryError):
                reader.read(selection)
        else:
            actual = reader.read(selection)
            np.testing.assert_array_equal(actual, values[selection])
            assert candidate.request_count_exact
            assert backend.calls == candidate.requests_per_value_lower_bound


def test_petabyte_selection_planning_does_not_enumerate_tiles():
    spec = ArraySpec((10**15,), (1,), 1)
    result = plan_concurrency(spec, (slice(None),), total_buffer_bytes=10**15 + 2,
                              max_workers=1, count_limit=3)
    choice = result.candidates[0]
    assert choice.materialization == "tiled"
    assert choice.requests_per_value_lower_bound == 4
    assert not choice.request_count_exact
    assert result.max_workers_without_tiling == 0


def test_worker_ceiling_is_bounded_without_enumerating_workers():
    result = plan_concurrency(ArraySpec((1,), (1,), 1), (slice(None),),
                              total_buffer_bytes=1024, max_workers=10**15)
    assert max(c.workers for c in result.candidates) == 1024
    assert len(result.candidates) == 11
    assert all(c.buffer_bytes_per_worker >= 1 for c in result.candidates)


@pytest.mark.parametrize("name,value", [
    ("total_buffer_bytes", 0), ("total_buffer_bytes", True), ("max_workers", -1),
    ("max_workers", 1.5), ("count_limit", False), ("count_limit", 0),
])
def test_invalid_controls_rejected(name, value):
    options = {"total_buffer_bytes": 1024, "max_workers": 8, "count_limit": 16, name: value}
    with pytest.raises(ValueError, match="positive integers"):
        plan_concurrency(ArraySpec((8,), (2,), 4), (slice(None),), **options)

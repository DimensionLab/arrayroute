# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.sample import PrioritySample


@pytest.mark.parametrize("batch", [1, 7, 100, 1000])
def test_exact_priority_oracle_and_batch_boundary_replay(batch):
    values = np.arange(3000, dtype="float64").reshape(1000, 3)
    sample = PrioritySample(57, seed=12)
    for start in range(0, len(values), batch):
        sample.update(values[start:start + batch])
    actual, positions = sample.result()
    keys = np.random.PCG64(12).random_raw(len(values))
    expected = np.sort(np.lexsort((np.arange(len(values)), keys))[:57])
    np.testing.assert_array_equal(positions, expected)
    assert actual.tobytes() == values[expected].tobytes()
    assert sample.seen == 1000


def test_small_stream_is_retained_exactly_and_results_are_owned():
    values = np.arange(12, dtype="int16").reshape(4, 3)
    sample = PrioritySample(20)
    sample.update(values)
    values[:] = -1
    rows, positions = sample.result()
    np.testing.assert_array_equal(rows, np.arange(12).reshape(4, 3))
    rows[:] = 100
    positions[:] = 100
    rows, positions = sample.result()
    np.testing.assert_array_equal(rows, np.arange(12).reshape(4, 3))
    np.testing.assert_array_equal(positions, np.arange(4))


def test_empty_batches_and_schema_validation():
    sample = PrioritySample(3)
    with pytest.raises(ValueError):
        sample.result()
    sample.update(np.empty((0, 2)))
    assert sample.result()[0].shape == (0, 2)
    with pytest.raises(ValueError, match="shape and dtype"):
        sample.update(np.zeros((2, 3)))
    assert sample.seen == 0


def test_failed_update_does_not_advance_rng_or_state(monkeypatch):
    values = np.arange(40, dtype="float64").reshape(20, 2)
    sample = PrioritySample(3)
    sample.update(values[:2])
    real = np.partition

    def fail(*args, **kwargs):
        raise MemoryError("injected allocation failure")

    monkeypatch.setattr(np, "partition", fail)
    with pytest.raises(MemoryError):
        sample.update(values[2:])
    assert sample.seen == 2
    monkeypatch.setattr(np, "partition", real)
    sample.update(values[2:])
    oracle = PrioritySample(3)
    oracle.update(values)
    for actual, expected in zip(sample.result(), oracle.result(), strict=True):
        assert actual.tobytes() == expected.tobytes()


def test_retained_rows_remain_bounded_over_many_updates():
    sample = PrioritySample(31)
    for i in range(100):
        sample.update(np.full((100, 2), i, dtype="float64"))
        assert len(sample.result()[0]) <= 31
    assert sample.seen == 10000


@pytest.mark.parametrize("capacity", [0, -1, True])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        PrioritySample(capacity)

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.axis_batches import aligned_axis_batches
from arrayroute.backends import NumpyBackend
from arrayroute.reader import ArrayReader, ReadStats


def test_mixed_axes_prefix_and_budget_cover_exact_source_bytes():
    first = np.arange(3 * 19 * 7, dtype=">f4").reshape(3, 19, 7)
    second = np.asfortranarray(first[0].astype("f8"))
    sources = [
        (ArrayReader(NumpyBackend(first), buffer_bytes=2000), 1),
        (ArrayReader(NumpyBackend(second), buffer_bytes=2000), -2),
    ]
    stats = ReadStats()
    out = list(
        aligned_axis_batches(
            sources, buffer_bytes=1000, workspace_bytes_per_step=60, steps=17, stats=stats
        )
    )
    assert [(b.start, b.stop) for b in out] == [(0, 5), (5, 10), (10, 15), (15, 17)]
    assert (
        np.concatenate([b.values[0] for b in out], axis=1, dtype=first.dtype).tobytes()
        == first[:, :17].tobytes()
    )
    np.testing.assert_array_equal(np.concatenate([b.values[1] for b in out]), second[:17])
    assert stats.requests == 8
    assert stats.output_bytes == 17 * (3 * 7 * 4 + 7 * 8)


def test_rejects_before_read_and_cancels_between_sources():
    class Counting(NumpyBackend):
        def __init__(self):
            super().__init__(np.ones((10, 20)))
            self.calls = 0

        def read(self, selection):
            self.calls += 1
            return super().read(selection)

    a, b = Counting(), Counting()
    sources = [(ArrayReader(a), 0), (ArrayReader(b), 0)]
    with pytest.raises(MemoryError):
        next(aligned_axis_batches(sources, buffer_bytes=319, workspace_bytes_per_step=0))
    assert a.calls == b.calls == 0
    with pytest.raises(InterruptedError):
        next(
            aligned_axis_batches(
                sources,
                buffer_bytes=1000,
                workspace_bytes_per_step=10,
                cancelled=lambda: a.calls > 0,
            )
        )
    assert a.calls == 1 and b.calls == 0


@pytest.mark.parametrize("change", ["axis", "extent", "empty", "steps", "budget", "workspace"])
def test_invalid_contracts(change):
    reader = ArrayReader(NumpyBackend(np.ones((4, 3))))
    sources, kwargs = [(reader, 0)], {"buffer_bytes": 1000, "workspace_bytes_per_step": 10}
    if change == "axis":
        sources = [(reader, True)]
    elif change == "extent":
        sources.append((reader, 1))
    elif change == "empty":
        sources = []
    elif change == "steps":
        kwargs["steps"] = 5
    elif change == "budget":
        kwargs["buffer_bytes"] = True
    else:
        kwargs["workspace_bytes_per_step"] = -1
    with pytest.raises(ValueError):
        next(aligned_axis_batches(sources, **kwargs))

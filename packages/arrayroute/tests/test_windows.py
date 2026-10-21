# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.windows import WindowIndex


def test_compact_index_matches_dense_windows():
    rng = np.random.default_rng(41)
    for _ in range(300):
        lengths = rng.integers(0, 100, size=8).tolist()
        history, unroll, lead, step = rng.integers(1, 8, size=4).tolist()
        dense = [
            (i, t)
            for i, n in enumerate(lengths)
            for t in range(history - 1, n - lead * unroll, step)
        ]
        compact = WindowIndex(lengths, history=history, unroll=unroll, lead=lead, step=step)
        assert len(compact) == len(dense)
        assert list(compact) == dense
        assert [compact[i] for i in range(len(dense))] == dense
        if dense:
            assert compact[-1] == dense[-1]
        with pytest.raises(IndexError):
            _ = compact[len(compact)]


def test_trillion_windows_do_not_allocate_per_window_metadata():
    index = WindowIndex([10**12, 0, 0, 10**12], history=2, unroll=4, lead=3, step=2)
    per_case = (10**12 - 12 - 2) // 2 + 1
    assert len(index) == 2 * per_case
    assert index[per_case] == (3, 1)
    assert index[-1] == (3, 1 + 2 * (per_case - 1))
    assert len(index._ends) == 4

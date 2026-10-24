# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import zarr

from arrayroute.backends import NumpyBackend, ZarrBackend
from arrayroute.composite import MeanBackend, StackBackend
from arrayroute.reader import ArrayReader, ReadStats


def readers(values):
    return [ArrayReader(NumpyBackend(a), buffer_bytes=4096) for a in values]


@pytest.mark.parametrize(
    "selection",
    [
        (slice(None), slice(None), slice(None)),
        (slice(1, 4, 2), slice(None, None, 2), 2),
        (2, 1, 3),
        (slice(None), 1, 3),
        (slice(0, 0), slice(None), slice(None)),
    ],
)
def test_lazy_stack_matches_numpy(selection):
    values = np.arange(4 * 9 * 13, dtype=np.int32).reshape(4, 9, 13)
    r = ArrayReader(StackBackend(readers(values), workspace_bytes=384), buffer_bytes=4096)
    np.testing.assert_array_equal(r.read(selection), values[selection])


def test_mean_precise_and_invariant_to_blocks_chunks_and_layout(tmp_path):
    rng = np.random.default_rng(3)
    values = rng.normal(size=(9, 19, 31, 3)).astype("f4")
    expected = values.astype("f8").mean(0)
    baseline = None
    for budget in [192, 4096, 65536]:
        actual = ArrayReader(MeanBackend(readers(values), workspace_bytes=budget)).read(
            (slice(None), slice(None), slice(None))
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        if baseline is not None:
            assert actual.tobytes() == baseline.tobytes()
        baseline = actual
    group = zarr.open_group(str(tmp_path / "members.zarr"), mode="w")
    sources = []
    for i, a in enumerate(values):
        group.create_array(str(i), data=a, chunks=(3, 7, 3))
        sources.append(ArrayReader(ZarrBackend(group[str(i)]), buffer_bytes=4096))
    actual = ArrayReader(MeanBackend(sources, workspace_bytes=65536)).read(
        (slice(None), slice(None), slice(None))
    )
    assert actual.tobytes() == baseline.tobytes()
    selected = ArrayReader(MeanBackend(readers(np.asfortranarray(values))))
    assert (
        selected.read((slice(None, None, 2), 4, slice(None))).tobytes()
        == baseline[::2, 4].tobytes()
    )


def test_mean_compensation_extremes_and_nonfinite():
    values = np.array([[1e16, 1e308, np.inf, np.inf], [1, 1e308, 2, -np.inf], [-1e16, 1e308, 3, 0]])
    result = ArrayReader(MeanBackend(readers(values))).read((slice(None),))
    assert result[0] == pytest.approx(1 / 3)
    assert result[1] == pytest.approx(1e308)
    assert result[2] == np.inf and np.isnan(result[3])


@pytest.mark.parametrize("backend", [StackBackend, MeanBackend])
def test_cancel_and_validation_before_payload(backend, monkeypatch):
    source = readers(np.ones((2, 100)))[0]
    monkeypatch.setattr(source.backend, "read", lambda *a: pytest.fail("unexpected I/O"))
    with pytest.raises(InterruptedError):
        ArrayReader(backend([source], cancelled=lambda: True)).read(
            (slice(None),) * (2 if backend is StackBackend else 1)
        )
    with pytest.raises(ValueError):
        backend([source], workspace_bytes=191)
    with pytest.raises(ValueError, match="identical"):
        backend([source, readers([np.ones(99)])[0]])


def test_sources_are_bounded_and_accounted_separately():
    values = np.ones((5, 1000), dtype="f4")
    stats = ReadStats()
    backend = MeanBackend(readers(values), workspace_bytes=1920, source_stats=stats)
    r = ArrayReader(backend, buffer_bytes=256)
    output = np.concatenate([part for _, part in r.stream((slice(None),))])
    np.testing.assert_array_equal(output, 1)
    assert stats.output_bytes == values.nbytes
    assert stats.requests >= 500


def test_recursive_shared_budget_reservation_rejected_before_deadlock():
    from arrayroute.budget import ReadBudget
    from arrayroute.derived import VectorMagnitudeBackend

    pool = ReadBudget(4096)
    source = ArrayReader(NumpyBackend(np.ones((10, 3))), shared_budget=pool)
    mean = ArrayReader(MeanBackend([source]))
    for backend in [MeanBackend([source]), StackBackend([mean]), VectorMagnitudeBackend(mean)]:
        with pytest.raises(ValueError, match="shared read pool"):
            ArrayReader(backend, shared_budget=pool)

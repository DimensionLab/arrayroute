# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import zarr

from arrayroute.backends import NumpyBackend, ZarrBackend
from arrayroute.error_curves import error_curves
from arrayroute.reader import ArrayReader, ReadStats


def reader(values):
    return ArrayReader(NumpyBackend(values))


def reference(p, t):
    p, t = p.reshape(len(p), -1, p.shape[-1]), t.reshape(len(t), -1, t.shape[-1])
    n = p.shape[1]
    axes = (1, 2)
    def norm(a):
        return np.linalg.norm(a.reshape(len(a), -1), axis=1)
    rmse = norm(p - t) / np.sqrt(n)
    sigma = np.sqrt(t.var(axis=(0, 1)).sum())
    pa, ta = p - p.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
    return {
        "l2_rel": norm(p - t) / norm(t), "rmse": rmse, "nrmse": rmse / sigma,
        "vrmse": rmse / np.sqrt(t.var(axis=1).sum(1)),
        "corr": (pa * ta).sum(axes) / (norm(pa) * norm(ta)),
        "persistence": norm(t[:1] - t) / norm(t),
        "climatology": norm(t.mean(0, keepdims=True) - t) / norm(t),
        "ms_pred": np.square(p).sum(axes) / n, "ms_true": np.square(t).sum(axes) / n,
        "sigma": np.array(sigma), "finite": np.ones(len(t), bool),
    }


@pytest.mark.parametrize("shape", [(7, 515, 2), (7, 13, 31, 2), (7, 5, 9, 11, 3)])
def test_matches_independent_dense_reference_and_budget_invariant(shape):
    rng = np.random.default_rng(12)
    t = rng.normal(size=shape)
    p = t + rng.normal(size=shape) * .1
    expected = reference(p, t)
    first = None
    for budget in [2 << 20, 8 << 20]:
        actual = error_curves(reader(p), reader(t), buffer_bytes=budget)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=2e-13, atol=2e-13)
            if first is not None:
                assert actual[name].tobytes() == first[name].tobytes()
        first = actual


def test_storage_chunks_and_memory_layout_do_not_change_results(tmp_path):
    rng = np.random.default_rng(17)
    t = rng.normal(size=(7, 13, 31, 2))
    p = t + .1
    expected = error_curves(reader(p), reader(t))
    for i, chunks in enumerate([(1, 4, 8, 2), (3, 13, 31, 2)]):
        group = zarr.open_group(str(tmp_path / str(i)), mode="w")
        group.create_array("p", data=p, chunks=chunks)
        group.create_array("t", data=t, chunks=chunks)
        actual = error_curves(ArrayReader(ZarrBackend(group["p"])),
                              ArrayReader(ZarrBackend(group["t"])))
        for name in expected:
            assert actual[name].tobytes() == expected[name].tobytes()
    actual = error_curves(reader(np.asfortranarray(p)), reader(np.asfortranarray(t)))
    for name in expected:
        assert actual[name].tobytes() == expected[name].tobytes()


def test_masks_preserve_vector_rmse_and_ignore_missing_reference():
    t = np.ones((3, 600, 3))
    p = t.copy()
    mask = np.ones(t.shape, bool)
    mask[0, 1:] = False
    mask[1, :300] = False
    mask[2] = False
    p[0, 0] += [3, 4, 0]
    p[~mask], t[~mask] = np.inf, np.nan
    result = error_curves(reader(p), reader(t), validity=reader(mask))
    assert result["rmse"][0] == pytest.approx(5)
    assert result["rmse"][1] == 0
    assert np.isinf(result["rmse"][2]) and not result["finite"][2]
    assert np.isinf(result["persistence"][1:]).all()
    assert result["sigma"] == 0


def test_nonfinite_prediction_scores_infinite_but_reference_fails():
    t = np.ones((2, 300, 1))
    p = t.copy()
    p[1, 200] = np.nan
    actual = error_curves(reader(p), reader(t))
    assert actual["finite"].tolist() == [True, False]
    assert np.isinf(actual["l2_rel"][1]) and np.isnan(actual["corr"][1])
    with pytest.raises(ValueError, match="Nonfinite reference"):
        error_curves(reader(t), reader(p))


def test_bad_masks_and_shapes_fail():
    t = np.ones((2, 300, 3))
    for value in [2, np.nan, -1]:
        with pytest.raises(ValueError, match="zero/one"):
            error_curves(reader(t), reader(t), validity=reader(np.full(t.shape, value)))
    mask = np.ones_like(t)
    mask[:, 0, 1] = 0
    with pytest.raises(ValueError, match="common validity"):
        error_curves(reader(t), reader(t), validity=reader(mask))
    with pytest.raises(ValueError, match="matching"):
        error_curves(reader(t), reader(t[:, :-1]))


def test_admission_and_cancellation_before_payload(monkeypatch):
    r = reader(np.ones((8, 600, 3)))
    monkeypatch.setattr(r, "read", lambda *a, **kw: pytest.fail("payload I/O"))
    with pytest.raises(MemoryError):
        error_curves(r, r, buffer_bytes=100)
    with pytest.raises(InterruptedError):
        error_curves(r, r, cancelled=lambda: True)


def test_cancel_between_reads_and_reuse_after_cancellation():
    r = reader(np.ones((4, 600, 1)))
    stats = ReadStats()
    with pytest.raises(InterruptedError):
        error_curves(r, r, stats=stats, cancelled=lambda: stats.requests >= 2)
    assert stats.requests == 2
    np.testing.assert_array_equal(error_curves(r, r)["rmse"], 0)


def test_large_offset_variance_stays_resolved():
    rng = np.random.default_rng(21)
    t = 1e10 + rng.normal(size=(8, 600, 3))
    p = t + .1
    actual = error_curves(reader(p), reader(t))
    expected = reference(p, t)
    for key in ["sigma", "corr", "nrmse", "vrmse"]:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-7, atol=1e-7)


def test_coalesced_windows_preserve_reduction_order_and_reduce_reads():
    rng = np.random.default_rng(23)
    truth = rng.normal(size=(7, 25, 31, 19, 2))
    pred = truth + rng.normal(size=truth.shape) * .02
    results = []
    counts = []
    for budget in [2 << 20, 16 << 20]:
        stats = ReadStats()
        results.append(error_curves(reader(pred), reader(truth),
                                    buffer_bytes=budget, stats=stats))
        counts.append(stats.requests)
    assert counts[0] > counts[1]
    assert counts[0] < 4 * 25 * 3  # Uncoalesced: three reduction groups per X row, two passes.
    for key in results[0]:
        assert results[0][key].tobytes() == results[1][key].tobytes()
    expected = reference(pred, truth)
    for key in expected:
        np.testing.assert_allclose(results[0][key], expected[key], rtol=2e-13, atol=2e-13)


def test_random_masked_curves_match_independent_observed_point_calculation():
    rng = np.random.default_rng(24)
    truth = rng.normal(size=(6, 787, 3))
    pred = truth + rng.normal(size=truth.shape) * .1
    valid = np.repeat((rng.random((6, 787, 1)) > .2), 3, axis=2)
    result = error_curves(reader(pred), reader(truth), validity=reader(valid))
    observed = truth[valid].reshape(-1, 3)
    sigma = np.sqrt(observed.var(axis=0).sum())
    assert result['sigma'] == pytest.approx(sigma, rel=1e-13)
    for i in range(6):
        p, t = pred[i][valid[i, :, 0]], truth[i][valid[i, :, 0]]
        rmse = np.sqrt(np.square(p - t).sum() / len(t))
        assert result['rmse'][i] == pytest.approx(rmse, rel=1e-13)
        pa, ta = p - p.mean(0), t - t.mean(0)
        corr = (pa * ta).sum() / np.sqrt(np.square(pa).sum() * np.square(ta).sum())
        assert result['corr'][i] == pytest.approx(corr, rel=1e-13)

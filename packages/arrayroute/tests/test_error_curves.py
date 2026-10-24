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
    assert np.isinf(result["climatology"][2])
    assert np.isnan(result["ms_true"][2])


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
    monkeypatch.setattr(r.backend, "read", lambda *a, **kw: pytest.fail("payload I/O"))
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


@pytest.mark.parametrize('shape', [(1025, 513, 2), (193, 5, 19, 31, 2)])
def test_long_rollouts_match_dense_with_bounded_time_reads(shape, monkeypatch):
    rng = np.random.default_rng(30)
    truth = rng.normal(size=shape)
    pred = truth + rng.normal(size=shape) * .05
    expected = reference(pred, truth)
    results = []
    for budget in [8 << 20, 32 << 20]:
        pr, tr = reader(pred), reader(truth)
        for r in (pr, tr):
            original = r.backend.read
            def bounded(selection, original=original):
                assert selection[0].stop - selection[0].start <= 64
                return original(selection)
            monkeypatch.setattr(r.backend, 'read', bounded)
        results.append(error_curves(pr, tr, buffer_bytes=budget))
        for key in expected:
            np.testing.assert_allclose(results[-1][key], expected[key], rtol=3e-13, atol=3e-13)
    for key in expected:
        assert results[0][key].tobytes() == results[1][key].tobytes()


def test_long_masked_climate_persistence_and_correlation_are_observed_only():
    rng = np.random.default_rng(31)
    truth = rng.normal(size=(137, 599, 3))
    pred = truth + rng.normal(size=truth.shape) * .1
    valid = np.repeat(rng.random((137, 599, 1)) > .4, 3, axis=2)
    valid[0, :300] = False
    valid[65, 300:] = False
    valid[66] = False
    valid[:64, 20:30] = False
    clean_t, clean_p = truth.copy(), pred.copy()
    truth[~valid], pred[~valid] = np.nan, np.inf
    expected_climate = np.where(valid, clean_t, 0).sum(0) / np.maximum(valid.sum(0), 1)
    baseline = None
    for budget in [16 << 20, 48 << 20]:
        result = error_curves(reader(pred), reader(truth), validity=reader(valid),
                              buffer_bytes=budget)
        for i in [0, 63, 64, 65, 66, 128, 136]:
            visible = valid[i, :, 0]
            if not visible.any():
                assert np.isinf(result['climatology'][i])
                continue
            t, p = clean_t[i, visible], clean_p[i, visible]
            climate_error = np.linalg.norm(expected_climate[visible] - t) / np.linalg.norm(t)
            assert result['climatology'][i] == pytest.approx(climate_error, rel=2e-13)
            common = valid[0, :, 0] & visible
            if not common.any():
                assert np.isinf(result['persistence'][i])
            else:
                persistence = np.linalg.norm(clean_t[0, common] - clean_t[i, common])
                persistence /= np.linalg.norm(clean_t[i, common])
                assert result['persistence'][i] == pytest.approx(persistence, abs=2e-13)
            pa, ta = p - p.mean(0), t - t.mean(0)
            corr = (pa * ta).sum() / np.sqrt((pa * pa).sum() * (ta * ta).sum())
            assert result['corr'][i] == pytest.approx(corr, rel=2e-13)
        if baseline is not None:
            for key in result:
                assert baseline[key].tobytes() == result[key].tobytes()
        baseline = result


def test_long_rollout_chunk_and_layout_replay(tmp_path):
    rng = np.random.default_rng(32)
    truth = rng.normal(size=(131, 17, 33, 2))
    pred = truth + .125
    expected = error_curves(reader(pred), reader(truth), buffer_bytes=8 << 20)
    for i, chunks in enumerate([(7, 5, 11, 2), (65, 17, 33, 2)]):
        group = zarr.open_group(str(tmp_path / str(i)), mode='w')
        group.create_array('p', data=pred, chunks=chunks)
        group.create_array('t', data=truth, chunks=chunks)
        actual = error_curves(ArrayReader(ZarrBackend(group['p'])),
                              ArrayReader(ZarrBackend(group['t'])), buffer_bytes=16 << 20)
        for key in expected:
            assert actual[key].tobytes() == expected[key].tobytes()
    actual = error_curves(reader(np.asfortranarray(pred)), reader(np.asfortranarray(truth)))
    for key in expected:
        assert actual[key].tobytes() == expected[key].tobytes()

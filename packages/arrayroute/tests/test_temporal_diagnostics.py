# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.backends import NumpyBackend
from arrayroute.error_curves import error_curves
from arrayroute.reader import ArrayReader


def reader(a):
    return ArrayReader(NumpyBackend(a))


@pytest.mark.parametrize("shape", [(137, 33, 19, 2), (3, 8000, 3), (1, 12, 1)])
def test_diagnostics_match_dense_oracle_and_replay_across_budgets(shape):
    rng = np.random.default_rng(53)
    t = rng.normal(size=shape)
    p = t * 0.9 + rng.normal(size=shape) * 0.05
    pf, tf = (a.reshape(shape[0], -1, shape[-1]) for a in (p, t))
    pm, tm = pf.mean(0), tf.mean(0)
    ps, ts = pf.std(0), tf.std(0)
    probes = 7
    indices = np.unique(np.linspace(0, tf.shape[1] - 1, min(tf.shape[1], probes)).astype(int))
    expected = {
        "time_mean_difference_sq": np.square(pm - tm).sum(),
        "time_mean_reference_sq": np.square(tm).sum(),
        "time_std_difference_sq": np.square(ps - ts).sum(),
        "time_std_reference_sq": np.square(ts).sum(),
        "time_std_prediction_mean": ps.mean(),
        "time_std_reference_mean": ts.mean(),
        "global_std_prediction": np.sqrt(np.mean((pf - pf.mean((0, 1))) ** 2)),
        "probes_prediction": np.concatenate([pf[:, indices].reshape(shape[0], -1), pf.mean(1)], 1),
        "probes_reference": np.concatenate([tf[:, indices].reshape(shape[0], -1), tf.mean(1)], 1),
    }
    results = [
        error_curves(reader(p), reader(t), diagnostics=True, probes=probes, buffer_bytes=b)
        for b in [8 << 20, 32 << 20]
    ]
    for key, value in expected.items():
        np.testing.assert_allclose(results[0][key], value, rtol=3e-13, atol=3e-13)
    for key in results[0]:
        assert results[0][key].tobytes() == results[1][key].tobytes(), key


def test_prefix_does_not_read_tail_and_zero_probes_only_returns_mean(monkeypatch):
    a = np.arange(150 * 300, dtype=float).reshape(150, 300, 1)
    r = reader(a)
    original = r.backend.read

    def read(selection):
        assert selection[0].stop <= 73
        return original(selection)

    monkeypatch.setattr(r.backend, "read", read)
    actual = error_curves(r, r, steps=73, diagnostics=True, probes=0)
    expected = error_curves(reader(a[:73]), reader(a[:73]), diagnostics=True, probes=0)
    for key in actual:
        assert actual[key].tobytes() == expected[key].tobytes(), key
    assert actual["probes_reference"].shape == (73, 1)
    assert actual["exact_copy"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": 3},
        {"steps": True},
        {"steps": 1.5},
        {"probes": -1},
        {"probes": True},
        {"probes": 1025},
        {"diagnostics": 1},
    ],
)
def test_invalid_diagnostics_rejected_before_read(kwargs, monkeypatch):
    r = reader(np.ones((2, 300, 1)))
    monkeypatch.setattr(r.backend, "read", lambda *args: pytest.fail("payload I/O"))
    with pytest.raises(ValueError):
        error_curves(r, r, **kwargs)


def test_diagnostic_state_admission_and_mask_contract(monkeypatch):
    r = reader(np.ones((137, 300, 3)))
    monkeypatch.setattr(r.backend, "read", lambda *args: pytest.fail("payload I/O"))
    with pytest.raises(MemoryError):
        error_curves(r, r, diagnostics=True, buffer_bytes=1 << 20)
    with pytest.raises(ValueError, match="fully observed"):
        error_curves(r, r, validity=r, diagnostics=True)


def test_exact_copy_checks_source_integers_before_float_conversion():
    t = np.full((2, 300, 1), 2**60, dtype=np.uint64)
    p = t.copy()
    p[0, 0] += np.uint64(1)
    assert not error_curves(reader(p), reader(t))["exact_copy"]


def test_long_ragged_spatial_windows_replay_with_temporal_anchors():
    rng = np.random.default_rng(89)
    t = 1e8 + rng.normal(size=(137, 33, 65, 3))
    p = t + .03
    results = [error_curves(reader(p), reader(t), diagnostics=True, probes=7,
                           buffer_bytes=b) for b in [16 << 20, 64 << 20]]
    for key in results[0]:
        assert results[0][key].tobytes() == results[1][key].tobytes(), key
    assert float(results[0]["time_std_reference_mean"]) == pytest.approx(
        t.std(0).mean(), rel=1e-7)

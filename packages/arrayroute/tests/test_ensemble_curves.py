# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from arrayroute.backends import NumpyBackend
from arrayroute.ensemble_curves import ensemble_curves, gaussian_crps
from arrayroute.reader import ArrayReader, ReadStats


def readers(ensemble, truth):
    return ArrayReader(NumpyBackend(ensemble)), ArrayReader(NumpyBackend(truth))


def test_tiled_replay_and_dense_formula_without_full_frames():
    rng = np.random.default_rng(172)
    truth = rng.normal(size=(19, 129, 81, 3)).astype("f4")
    ens = truth + rng.normal(size=(3, *truth.shape)).astype("f4") * 0.2

    class Bounded(NumpyBackend):
        def read(self, selection):
            selected = self.array[selection]
            if budget == 4 << 20:
                leading = 2 if self.array.ndim == ens.ndim else 1
                assert math.prod(selected.shape[leading:-1]) < math.prod(truth.shape[1:-1])
            return super().read(selection)

    # A single uncertainty frame's prior estimate is >10 MiB; 4 MiB suffices here.
    outputs = []
    for budget, order in [(4 << 20, "C"), (16 << 20, "F")]:
        stats = ReadStats()
        out = ensemble_curves(
            ArrayReader(Bounded(np.array(ens, order=order))),
            ArrayReader(Bounded(np.array(truth, order=order))),
            z_nominal=1.645,
            multipliers=np.array([0.5, 2.0]),
            n_scores=37,
            steps=17,
            buffer_bytes=budget,
            stats=stats,
        )
        assert stats.requests > 0
        outputs.append(out)
    for key in outputs[0]:
        assert outputs[0][key].tobytes() == outputs[1][key].tobytes(), key
    e = ens[:, :17].astype("f8").reshape(3, 17, -1, 3)
    t = truth[:17].astype("f8").reshape(17, -1, 3)
    mu, sd = e.mean(0), e.std(0, ddof=1)
    sp = sd * math.sqrt(4 / 3)
    err = t - mu
    expected = {
        "spread": np.sqrt(np.sum(sd**2, axis=(1, 2)) / t.shape[1]),
        "rmse": np.sqrt(np.sum(err**2, axis=(1, 2)) / t.shape[1]),
        "coverage": (np.abs(err) <= 1.645 * sp).mean(axis=(1, 2)),
        "coverage_conformal": (
            np.abs(err) <= np.array([0.5] + [2.0] * 16)[:, None, None] * sp
        ).mean(axis=(1, 2)),
        "crps": gaussian_crps(mu, sp, t).mean(axis=(1, 2)),
    }
    expected["ratio"] = math.sqrt(4 / 3) * expected["spread"] / expected["rmse"]
    index = np.unique(np.linspace(0, t.shape[1] * 3 - 1, 37).astype(int))
    expected["scores"] = np.abs(err).reshape(17, -1)[:, index] / sp.reshape(17, -1)[:, index]
    for key, value in expected.items():
        np.testing.assert_allclose(outputs[0][key], value, rtol=2e-12, atol=2e-12)


def test_gaussian_crps_matches_independent_erf_formula():
    y = np.linspace(-10, 10, 1001)
    z = y / 2
    cdf = (1 + np.array([math.erf(v / math.sqrt(2)) for v in z])) / 2
    expected = 2 * (
        z * (2 * cdf - 1) + 2 * np.exp(-z * z / 2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
    )
    # CRPS multiplies the erf approximation error by |truth - mean|.
    actual = gaussian_crps(np.zeros_like(y), np.full_like(y, 2), y)
    np.testing.assert_array_less(np.abs(actual - expected), np.abs(y) * 1.5e-7 + 1e-14)
    np.testing.assert_array_equal(gaussian_crps(np.zeros_like(y), np.zeros_like(y), y), np.abs(y))


def test_admission_invalid_metadata_and_cancellation_before_io():
    class NoRead(NumpyBackend):
        def read(self, selection):
            pytest.fail("Payload read before admission/cancellation")

    e = ArrayReader(NoRead(np.zeros((3, 20, 33, 31, 2))))
    t = ArrayReader(NoRead(np.zeros((20, 33, 31, 2))))
    with pytest.raises(MemoryError):
        ensemble_curves(e, t, z_nominal=1.645, buffer_bytes=1024)
    with pytest.raises(InterruptedError):
        ensemble_curves(e, t, z_nominal=1.645, cancelled=lambda: True)
    for mult in [np.array([]), np.array([np.nan]), np.array([-1.0]), np.ones((2, 2))]:
        with pytest.raises(ValueError):
            ensemble_curves(e, t, z_nominal=1.645, multipliers=mult)


def test_zero_spread_nonfinite_predictions_and_reference_rejection():
    t = np.ones((9, 17, 2))
    e = np.ones((3, *t.shape))
    out = ensemble_curves(*readers(e, t), z_nominal=1.645, n_scores=4)
    assert np.all(out["spread"] == 0) and np.all(out["rmse"] == 0)
    assert np.all(out["coverage"] == 1) and np.isnan(out["ratio"]).all()
    assert np.isinf(out["scores"]).all()
    e[:, 3, 1] = np.nan
    out = ensemble_curves(*readers(e, t), z_nominal=1.645)
    assert np.isinf(out["spread"][3]) and np.isinf(out["crps"][3])
    t[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="reference"):
        ensemble_curves(*readers(e, t), z_nominal=1.645)


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore"])
def test_chunked_layouts_reproduce_prefix_and_score_bytes(tmp_path, engine):
    import zarr

    from arrayroute.session import LocalReadSession

    rng = np.random.default_rng(177)
    truth = rng.normal(size=(13, 33, 37, 2)).astype("f4")
    ens = truth + rng.normal(size=(3, *truth.shape)).astype("f4") * 0.2
    expected = ensemble_curves(*readers(ens, truth), z_nominal=1.645, n_scores=35, steps=11)
    for i, chunks in enumerate([(1, 11, 37, 2), (5, 7, 9, 1)]):
        path = tmp_path / str(i)
        group = zarr.open_group(str(path), mode="w")
        group.create_array("e", data=ens, chunks=(1, *chunks))
        group.create_array("t", data=truth, chunks=chunks)
        session = LocalReadSession(engine, buffer_bytes=4 << 20, immutable=True)
        actual = ensemble_curves(
            session.open(path, "e"),
            session.open(path, "t"),
            z_nominal=1.645,
            n_scores=35,
            steps=11,
            buffer_bytes=4 << 20,
        )
        for key in expected:
            assert actual[key].tobytes() == expected[key].tobytes(), key

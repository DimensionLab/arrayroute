# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor

import numexpr
import numpy as np
import pytest

from arrayroute import normalization as module
from arrayroute.normalization import normalization_backend_info, normalize_channels


@pytest.mark.parametrize("engine", ["numpy", "numexpr"])
@pytest.mark.parametrize("shape", [(5,), (0, 5), (13, 7, 5), (3, 4, 2, 5)])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
def test_recorded_arithmetic_exact_and_source_unchanged(engine, shape, layout):
    rng = np.random.default_rng(329)
    x = rng.normal(size=shape).astype(np.float32)
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "strided":
        x = x[::2]
    elif layout == "reversed":
        x = x[::-1]
    means = np.array([-3, -0., 0.3, 1.7, 2e-4], dtype=np.float32)
    scales = np.array([0.1, 1.01, 2, 3e4, 0.07], dtype=np.float32)
    # A 1-D strided input changes its channel count, so select matching statistics.
    if x.ndim == 1 and layout == "strided":
        means, scales = means[::2], scales[::2]
    original = x.tobytes()
    x.flags.writeable = False
    expected = (x - means) / scales
    actual = normalize_channels(x, means, scales, engine=engine)
    assert actual.tobytes() == expected.tobytes()
    assert x.tobytes() == original
    assert not np.shares_memory(x, actual)


@pytest.mark.parametrize("engine", ["numpy", "numexpr"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_output_buffers_and_aliasing_parameters(engine, inplace, order):
    x = np.array([[1, 2, 3], [2, 4, 6], [3, 8, 7]], dtype=np.float32, order=order)
    means, scales = x[0], x[1]
    expected = (x - means.copy()) / scales.copy()
    out = x if inplace else np.empty_like(x)
    assert normalize_channels(x, means, scales, out=out, engine=engine) is out
    assert out.tobytes() == expected.tobytes()


@pytest.mark.parametrize("channels", [1, 5, 17])
def test_float32_bit_patterns_include_subnormal_signed_zero_and_nonfinite(channels):
    rng = np.random.default_rng(429)
    bits = rng.integers(0, 2**32, size=(10000, channels), dtype=np.uint32)
    bits[:8] = np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000,
                         0xff800000, 0x7fc00000, 0xffc00000], dtype=np.uint32)[:, None]
    x = bits.view(np.float32)
    means = rng.normal(size=channels).astype(np.float32)
    means[0] = 0
    scales = rng.uniform(0.01, 10, size=channels).astype(np.float32)
    with np.errstate(all="ignore"):
        expected = (x - means) / scales
        actual = normalize_channels(x, means, scales, engine="numexpr")
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("engine", ["numpy", "numexpr"])
@pytest.mark.parametrize("problem", ["dtype", "channels", "scalar", "means", "scale",
                                     "negative", "zero", "nan", "inf", "output-dtype",
                                     "output-shape", "readonly", "overlap", "strided-out"])
def test_invalid_contract_fails_before_writing(engine, problem):
    base = np.arange(60, dtype=np.float32).reshape(20, 3)
    x, out = base[:10], np.full((10, 3), 42, dtype=np.float32)
    means, scales = [0, 0, 0], [1, 1, 1]
    if problem == "dtype":
        x = x.astype(np.float64)
    elif problem == "channels":
        x = x[:, :0]
    elif problem == "scalar":
        x = np.array(1, dtype=np.float32)
    elif problem == "means":
        means = [0]
    elif problem == "scale":
        scales = [1]
    elif problem in {"negative", "zero", "nan", "inf"}:
        scales = [{"negative": -1, "zero": 0, "nan": float("nan"), "inf": float("inf")}[problem]]*3
    elif problem == "output-dtype":
        out = out.astype(np.float64)
    elif problem == "output-shape":
        out = out[:5]
    elif problem == "readonly":
        out.flags.writeable = False
    elif problem == "overlap":
        out = base[1:11]
    elif problem == "strided-out":
        out = base[::2]
    before = out.tobytes()
    with pytest.raises(ValueError):
        normalize_channels(x, means, scales, out=out, engine=engine)
    assert out.tobytes() == before


def test_selected_missing_backend_and_unknown_engine_fail_without_writing(monkeypatch):
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr(module.importlib, "import_module", missing)
    x = np.zeros((4, 2), dtype=np.float32)
    out = np.full_like(x, 7)
    for engine, error in [("numexpr", ImportError), ("automatic", ValueError)]:
        with pytest.raises(error):
            normalize_channels(x, [0, 0], [1, 1], out=out, engine=engine)
        assert (out == 7).all()
        with pytest.raises(error):
            normalization_backend_info(engine)
    assert (normalize_channels(x, [0, 0], [1, 1]) == 0).all()


def test_concurrent_normalizers_do_not_change_process_thread_pool(monkeypatch):
    threads = numexpr.get_num_threads()
    def forbidden(*args):
        raise AssertionError("Must not reconfigure a process-global pool from a transform")
    monkeypatch.setattr(numexpr, "set_num_threads", forbidden)
    def transform(seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(150, 250, 5)).astype(np.float32)
        expected = (x - np.float32(seed)) / np.float32(1.7)
        actual = normalize_channels(x, [seed]*5, [1.7]*5, out=x, engine="numexpr")
        assert expected.tobytes() == actual.tobytes()
        return normalization_backend_info("numexpr")
    with ThreadPoolExecutor(max_workers=4) as pool:
        infos = list(pool.map(transform, range(24)))
    assert all(i["threads"] == threads and i["optimization"] == "moderate" for i in infos)

# SPDX-License-Identifier: Apache-2.0
import pytest

from arrayroute.doctor import probe

torch = pytest.importorskip("torch")


def test_cpu_runtime_probe():
    result = probe()
    assert result["device"] == "cpu"
    assert result["normalization_replay_exact"]
    assert result["max_numpy_difference"] == 0
    assert result["cuda_device_count"] is None


def test_topology_is_not_silently_reduced(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="Multiple nodes"):
        probe(expected_nodes=2)
    with pytest.raises(ValueError, match="at least two ranks"):
        probe(distributed=True)


def test_unavailable_cuda_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        probe(cuda=True)

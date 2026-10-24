# SPDX-License-Identifier: Apache-2.0
import json
import os
import socket
import subprocess
import sys

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


@pytest.mark.parametrize("sizes,iterations", [([], 3), ([0], 3), ([5], 3),
    ([4, 4], 3), ([4] * 9, 3), ([128 << 20], 3), ([4], 2), ([4], 101), ([True], 3)])
def test_collective_limits(sizes, iterations):
    from arrayroute.collective_probe import validate_request

    with pytest.raises(ValueError):
        validate_request(sizes, iterations)


def test_measurement_requires_distributed():
    with pytest.raises(ValueError, match="require --distributed"):
        probe(collective_bytes=[4])


@pytest.mark.skipif(not torch.distributed.is_available()
                    or not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_real_two_rank_collective_measurements():
    env = dict(os.environ, OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        env["MASTER_PORT"] = str(listener.getsockname()[1])
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    workers, reports = [], []
    try:
        for rank in range(2):
            workers.append(subprocess.Popen(
                [sys.executable, "-m", "arrayroute.doctor", "--distributed",
                 "--collective-bytes", "4", "--collective-bytes", "1024",
                 "--collective-iterations", "3"],
                env=dict(env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                         GROUP_RANK="0"), text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE))
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=45)
            assert worker.returncode == 0, stdout + stderr
            reports.extend(json.loads(line) for line in stdout.splitlines()
                           if line.startswith("{"))
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=10)
    assert len(reports) == 2 and {r["rank"] for r in reports} == {0, 1}
    for report in reports:
        assert report["ok"] and report["collective_passed"]
        rows = report["collective_measurements"]["rows"]
        assert [r["payload_bytes"] for r in rows] == [4, 1024]
        assert all(r["payload_verified"] and r["p50_seconds"] > 0 for r in rows)
        assert all(len(r["seconds_per_collective_max_rank"]) == 3 for r in rows)
    assert reports[0]["collective_measurements"] == reports[1]["collective_measurements"]

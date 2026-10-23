# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import threading
from types import SimpleNamespace

import psutil
import pytest

from arrayroute.memory import ProcessTreeMemory


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_sample_interval(interval):
    with pytest.raises(ValueError):
        ProcessTreeMemory(interval_s=interval)


def test_lifecycle_and_thread_cleanup():
    threads = set(threading.enumerate())
    monitor = ProcessTreeMemory()
    with pytest.raises(RuntimeError, match="Close"):
        monitor.report()
    with monitor:
        with pytest.raises(RuntimeError, match="Close"):
            monitor.report()
    report = monitor.report()
    assert report["peak_bytes"] >= report["baseline_bytes"] > 0
    assert report["samples"] >= 2
    assert report["max_processes"] >= 1
    assert set(threading.enumerate()) == threads
    with pytest.raises(RuntimeError, match="reused"):
        with monitor:
            pass


def test_real_child_memory_is_included():
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; x=bytearray(48*1024*1024); "
         "print('ready',flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with ProcessTreeMemory() as monitor:
            pass
        report = monitor.report()
        assert report["max_processes"] >= 2
        assert report["peak_bytes"] >= 48 * 1024 * 1024
    finally:
        child.communicate(timeout=10)


def test_denied_and_vanished_processes_are_visible(monkeypatch):
    monitor = ProcessTreeMemory()
    def child(error):
        def fail():
            raise error(7)
        return SimpleNamespace(is_running=lambda: True, memory_info=fail)
    root = SimpleNamespace(
        children=lambda **_: [child(psutil.AccessDenied), child(psutil.NoSuchProcess)],
        is_running=lambda: True, memory_info=lambda: SimpleNamespace(rss=1234),
    )
    monkeypatch.setattr(monitor, "_root", root)
    with monitor:
        pass
    report = monitor.report()
    assert report["peak_bytes"] == 1234
    assert report["access_denied_observations"] >= 2
    assert report["vanished_process_observations"] >= 2


def test_background_failure_surfaces_and_joins(monkeypatch):
    monitor = ProcessTreeMemory(interval_s=0.001)
    seen = threading.Event()
    original = monitor._sample
    def sample():
        if threading.current_thread().name == "arrayroute-memory":
            seen.set()
            raise OSError("injected observation failure")
        original()
    monkeypatch.setattr(monitor, "_sample", sample)
    with pytest.raises(RuntimeError, match="sampling failed"):
        with monitor:
            assert seen.wait(5)
    assert monitor.report()["sampling_failed"]

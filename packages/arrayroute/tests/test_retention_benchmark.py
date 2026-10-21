# SPDX-License-Identifier: Apache-2.0
import pytest

pytest.importorskip("icechunk")

from arrayroute.benchmark_retention import benchmark  # noqa: E402


def test_lifecycle_benchmark_verifies_retention_and_refuses_existing_roots(tmp_path):
    root = tmp_path / "repository"
    result = benchmark(root, consumers=4)
    assert result["expected_sha256"] == result["actual_sha256"]
    assert result["released_gc_chunks_deleted"] > 0
    assert result["acquire"]["count"] == result["release"]["count"] == 4
    with pytest.raises(FileExistsError):
        benchmark(root, consumers=4)

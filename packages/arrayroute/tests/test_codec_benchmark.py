# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.benchmark_codecs import CPU_CODECS, PATTERNS, digest, fixture, measure, verify


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("codec", CPU_CODECS)
def test_codec_fixture_roundtrips_and_records_actual_size(pattern, codec):
    pytest.importorskip("numcodecs")
    result = measure(codec, pattern, side=8, repeats=2, threads=1)
    source = fixture(pattern, 8)
    assert result["source_bytes"] == source.nbytes
    assert result["source_sha256"] == digest(source) == digest(fixture(pattern, 8))
    assert result["byte_identical"] and result["verified_roundtrips"] == 3
    assert len(result["measurements"]) == 2
    assert all(item["encoded_bytes"] > 0 for item in result["measurements"])
    assert all(item["host_encode_s"] > 0 for item in result["measurements"])
    assert not result["hardware"]


def test_lossless_comparison_includes_nan_payload_and_signed_zero():
    values = np.array([0, 0x80000000, 0x7FC00001, 0x7FC00002, 0x7F800000], dtype="<u4")
    verify(values.view("<f4"), values.tobytes())
    changed = values.copy()
    changed[2] = changed[3]
    with pytest.raises(ValueError, match="source bytes"):
        verify(values.view("<f4"), changed.tobytes())
    with pytest.raises(ValueError, match="source bytes"):
        verify(values, values.tobytes()[:-1])


def test_fixture_rejects_unknown_patterns_and_invalid_sizes():
    with pytest.raises(ValueError):
        fixture("other", 8)
    with pytest.raises(ValueError):
        fixture("velocity", 1)

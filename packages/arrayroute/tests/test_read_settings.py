# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict

import pytest

from arrayroute.settings import ReadSettings


def test_cache_policy_roundtrip_and_distinct_identity():
    configured = ReadSettings(
        engine="tensorstore",
        tensorstore_cache_bytes=128 << 20,
        immutable=True,
        tensorstore_threads=8,
    )
    restored = ReadSettings(**asdict(configured))
    session = restored.session(32 << 20)
    assert session.tensorstore_cache_bytes == 128 << 20
    assert session.buffer_bytes == 32 << 20
    assert session.immutable is True
    assert restored.name == configured.name
    assert configured.name != ReadSettings(engine="tensorstore", tensorstore_threads=8).name


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tensorstore_cache_bytes": -1},
        {"tensorstore_cache_bytes": True},
        {"tensorstore_cache_bytes": 1.5},
        {"tensorstore_cache_bytes": 1},
        {"immutable": "yes"},
    ],
)
def test_invalid_cache_policy_fails_before_opening(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ReadSettings(**kwargs)

# SPDX-License-Identifier: Apache-2.0
from arrayroute.cache import ByteLRU


def test_lru_eviction_replacement_and_oversized_admission():
    cache = ByteLRU[str, str](10, max_entries=2)
    assert cache.put("a", "first", size_bytes=4)
    assert cache.put("b", "second", size_bytes=4)
    assert cache.get("a") == "first"
    assert cache.put("c", "third", size_bytes=4)
    assert cache.get("b") is None
    assert cache.used_bytes == 8
    assert cache.put("c", "replacement", size_bytes=7)
    assert cache.get("a") is None
    assert cache.used_bytes == 7
    assert not cache.put("c", "too large", size_bytes=11)
    assert len(cache) == cache.used_bytes == 0


def test_zero_capacity_disables_cache_and_zero_byte_entries_are_bounded():
    disabled = ByteLRU[str, str](0)
    assert not disabled.put("x", "y", size_bytes=0)
    cache = ByteLRU[int, int](1, max_entries=2)
    for i in range(100):
        cache.put(i, i, size_bytes=0)
    assert len(cache) == 2
    assert cache.get(97) is None
    assert cache.get(99) == 99

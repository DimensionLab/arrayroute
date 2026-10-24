# SPDX-License-Identifier: Apache-2.0
import hashlib
from threading import Event, Thread

import numpy as np
import pytest

from arrayroute.hashing import DigestPipeline


@pytest.mark.parametrize("factory", [hashlib.sha256, lambda: hashlib.blake2b(digest_size=8)])
@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">f8"])
def test_exact_digest_order_and_multiple_streams(factory, dtype):
    values = np.array([1, -0.0, np.nan, np.inf, -np.inf, 3.5], dtype=dtype)
    payload = values.tobytes()
    digests = [factory(), factory()]
    with DigestPipeline(buffer_bytes=8) as pipeline:
        for start in range(0, len(payload), 7):
            for digest in digests:
                pipeline.update(digest, payload[start : start + 7])
    expected = factory()
    expected.update(payload)
    assert all(d.digest() == expected.digest() for d in digests)


def test_payload_copied_before_return_and_one_pending_update():
    started, release, second_returned = Event(), Event(), Event()
    calls = []

    class SlowDigest:
        def update(self, data):
            started.set()
            assert release.wait(5)
            calls.append(bytes(data))

    with DigestPipeline(buffer_bytes=4) as pipeline:
        data = bytearray(b"abcd")
        pipeline.update(SlowDigest(), data)
        assert started.wait(5)
        data[:] = b"xxxx"

        def submit():
            pipeline.update(SlowDigest(), b"efgh")
            second_returned.set()

        producer = Thread(target=submit)
        producer.start()
        try:
            assert not second_returned.wait(0.05)
        finally:
            release.set()
            producer.join(5)
        assert not producer.is_alive() and second_returned.is_set()
    assert calls == [b"abcd", b"efgh"]


def test_failure_is_propagated_and_prevents_more_updates():
    class BrokenDigest:
        def update(self, data):
            raise OSError("hash failed")

    pipeline = DigestPipeline(buffer_bytes=4)
    pipeline.update(BrokenDigest(), b"ab")
    for action in [
        pipeline.flush,
        lambda: pipeline.update(hashlib.sha256(), b"cd"),
        pipeline.close,
    ]:
        with pytest.raises(OSError, match="hash failed"):
            action()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.update(hashlib.sha256(), b"cd")


def test_context_joins_work_and_preserves_producer_failure():
    done = Event()

    class BrokenDigest:
        def update(self, data):
            done.set()
            raise OSError("worker failure")

    with pytest.raises(ValueError, match="producer failure"):
        with DigestPipeline(buffer_bytes=4) as pipeline:
            pipeline.update(BrokenDigest(), b"ab")
            raise ValueError("producer failure")
    assert done.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.__enter__()


@pytest.mark.parametrize("budget", [True, False, 0, -1, 1.5])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="positive integer"):
        DigestPipeline(buffer_bytes=budget)


def test_rejects_oversized_or_noncontiguous_buffers_before_digest_update():
    digest = hashlib.sha256()
    with DigestPipeline(buffer_bytes=4) as pipeline:
        with pytest.raises(MemoryError, match="buffer_bytes"):
            pipeline.update(digest, b"12345")
        with pytest.raises(TypeError):
            pipeline.update(digest, np.arange(8, dtype="u1")[::2])
    assert digest.digest() == hashlib.sha256().digest()

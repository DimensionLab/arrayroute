# SPDX-License-Identifier: Apache-2.0
"""Overlap exact ordered digest updates with other array processing."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Self


def _consume(digest: Any, payload: list[bytes]) -> None:
    # Empty the executor's argument holder before signalling completion.
    data = payload.pop()
    try:
        digest.update(data)
    finally:
        del data


class DigestPipeline:
    """One hashing worker, at most one immutable payload of ``buffer_bytes`` bytes.

    ``update`` copies the submitted contiguous buffer before returning. The producer
    may then change or release it. A subsequent update waits for the previous one,
    even when targeting a different digest. Digests must not be accessed externally
    until ``flush`` or context exit succeeds. Failures propagate without retries;
    discard all affected digests after failure. This object has one producer.

    The bound covers retained payloads, not caller buffers, digest state, executor
    bookkeeping or process RSS. Closing joins active work; native hash calls cannot
    be interrupted. Hash algorithms and byte order are entirely caller-controlled.
    """

    def __init__(self, *, buffer_bytes: int) -> None:
        if isinstance(buffer_bytes, bool) or not isinstance(buffer_bytes, int) or buffer_bytes < 1:
            raise ValueError("buffer_bytes must be a positive integer")
        self.buffer_bytes = buffer_bytes
        self._pending: Future[None] | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arrayroute-digest")

    def update(self, digest: Any, data: Any) -> None:
        if self._closed:
            raise RuntimeError("Digest pipeline is closed")
        view = memoryview(data).cast("B")
        if view.nbytes > self.buffer_bytes:
            raise MemoryError("Digest payload exceeds buffer_bytes")
        self.flush()
        self._pending = self._executor.submit(_consume, digest, [bytes(view)])

    def flush(self) -> None:
        if self._pending is not None:
            # Keep a failed future so subsequent updates cannot silently resume.
            self._pending.result()
            self._pending = None

    def close(self) -> None:
        self._closed = True
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("Digest pipeline is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            # Preserve the producer's exception while joining any active hash call.
            self._closed = True
            self._executor.shutdown(wait=True)

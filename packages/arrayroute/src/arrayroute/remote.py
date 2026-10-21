# SPDX-License-Identifier: Apache-2.0
"""Read-only sessions for immutable Zarr v3 arrays on HTTP and S3-compatible storage."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from operator import index
from typing import Any, Literal
from urllib.parse import urlsplit

from .backends import TensorStoreBackend
from .plan import Selection
from .reader import ArrayReader, ReadStats


def _url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or any(ord(c) <= 32 for c in value)
    ):
        raise ValueError(
            "Use an HTTP(S) URL without user information, query, fragment or whitespace"
        )
    # Trigger port validation without reflecting a potentially secret URL in our error.
    try:
        _ = parts.port
    except ValueError:
        raise ValueError("Invalid endpoint port") from None
    return value.rstrip("/")


def _path(value: str) -> str:
    if value.startswith("/") or "\\" in value or any(ord(c) < 32 for c in value):
        raise ValueError("Array paths must be relative slash-separated names")
    value = value.rstrip("/")
    if value and any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Array paths cannot contain empty or parent/current-directory components")
    return value


@dataclass(frozen=True)
class HttpStore:
    """Trusted HTTP endpoint. Authentication tokens are deliberately not stored in this object."""

    base_url: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _url(self.base_url))

    def spec(self, array_path: str) -> dict:
        path = _path(array_path)
        return {"driver": "http", "base_url": self.base_url, "path": path + "/" if path else ""}


@dataclass(frozen=True)
class S3Store:
    """S3 or a compatible endpoint; credentials are resolved within each worker by TensorStore."""

    bucket: str
    prefix: str = ""
    region: str | None = None
    endpoint: str | None = None
    credentials: Literal["default", "environment", "anonymous"] = "default"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", self.bucket):
            raise ValueError("Expected a DNS-compatible S3 bucket name")
        object.__setattr__(self, "prefix", _path(self.prefix))
        if self.endpoint is not None:
            object.__setattr__(self, "endpoint", _url(self.endpoint))
        if self.credentials not in {"default", "environment", "anonymous"}:
            raise ValueError(
                "Choose a supported credential provider; inline credentials are not accepted"
            )

    def spec(self, array_path: str) -> dict:
        path = "/".join(p for p in (self.prefix, _path(array_path)) if p)
        result: dict[str, Any] = {
            "driver": "s3",
            "bucket": self.bucket,
            "path": path + "/" if path else "",
            "aws_credentials": {"type": self.credentials},
        }
        if self.region is not None:
            result["aws_region"] = self.region
        if self.endpoint is not None:
            result["endpoint"] = self.endpoint
        return result


class RemoteReadSession:
    """Reuse bounded read-only handles for one trusted, immutable remote store.

    This is a client library, not an SSRF boundary: callers must authorize endpoints
    before server-side use. Input objects must remain immutable; clear() drops this
    session's cached handles but cannot create a consistent snapshot of mutable objects.
    Native credential resolution occurs in each worker. Sessions are not thread-safe;
    serialized copies discard native handles/context and must be used with spawn.
    """

    def __init__(
        self,
        store: HttpStore | S3Store,
        *,
        buffer_bytes: int = 128 << 20,
        max_open_arrays: int = 64,
        cache_bytes: int = 0,
        concurrency: int = 8,
        max_retries: int = 3,
    ):
        if not isinstance(store, (HttpStore, S3Store)):
            raise TypeError("Provide an explicit HttpStore or S3Store")
        buffer_bytes, max_open_arrays, cache_bytes, concurrency, max_retries = (
            index(v) for v in (buffer_bytes, max_open_arrays, cache_bytes, concurrency, max_retries)
        )
        if min(buffer_bytes, max_open_arrays, concurrency, max_retries) < 1 or cache_bytes < 0:
            raise ValueError(
                "Positive buffer/handle/concurrency/retry limits and nonnegative cache required"
            )
        self.store = store
        self.buffer_bytes = buffer_bytes
        self.max_open_arrays = max_open_arrays
        self.cache_bytes = cache_bytes
        self.concurrency = concurrency
        self.max_retries = max_retries
        self._context: Any = None
        self._readers: OrderedDict[str, ArrayReader] = OrderedDict()

    def clear(self) -> None:
        self._readers.clear()
        self._context = None

    def __getstate__(self) -> dict:
        return {**vars(self), "_readers": OrderedDict(), "_context": None}

    def open(self, array_path: str = "") -> ArrayReader:
        import tensorstore as ts

        path = _path(array_path)
        if path in self._readers:
            self._readers.move_to_end(path)
            return self._readers[path]
        if self._context is None:
            protocol = "http" if isinstance(self.store, HttpStore) else "s3"
            self._context = ts.Context(
                {
                    "cache_pool": {"total_bytes_limit": self.cache_bytes},
                    "data_copy_concurrency": {"limit": self.concurrency},
                    f"{protocol}_request_concurrency": {"limit": self.concurrency},
                    f"{protocol}_request_retries": {"max_retries": self.max_retries},
                }
            )
        array = ts.open(
            {"driver": "zarr3", "kvstore": self.store.spec(path)},
            read=True,
            write=False,
            recheck_cached_data=False,
            context=self._context,
        ).result()
        reader = ArrayReader(TensorStoreBackend(array), buffer_bytes=self.buffer_bytes)
        while len(self._readers) >= self.max_open_arrays:
            self._readers.popitem(last=False)
        self._readers[path] = reader
        return reader

    def read(self, array_path: str, selection: Selection, *, stats: ReadStats | None = None):
        return self.open(array_path).read(selection, stats=stats)

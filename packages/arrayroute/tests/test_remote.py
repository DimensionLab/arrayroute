# SPDX-License-Identifier: Apache-2.0
import hashlib
import pickle
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
import zarr

from arrayroute.remote import HttpStore, RemoteReadSession, S3Store


@pytest.fixture
def object_server(tmp_path):
    values = np.random.default_rng(92).standard_normal((2, 64, 64, 64), dtype=np.float32)
    root = zarr.open_group(tmp_path, mode="w")
    for name in ("velocity", "pressure"):
        root.create_array(name, data=values, chunks=(1, 16, 16, 16), shards=(1, 64, 64, 64))
    files = {
        "/data/" + str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            key = unquote(urlsplit(self.path).path).removeprefix("/test-bucket")
            data = files.get(key)
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            etag = '"' + hashlib.sha256(data).hexdigest() + '"'
            if self.headers.get("If-None-Match") == etag:
                requests.append({
                    "key": key, "range": self.headers.get("Range"), "bytes": 0,
                    "signed": self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 "),
                })
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            if self.headers.get("If-Match", etag) != etag:
                self.send_response(412)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            requested = self.headers.get("Range")
            start, stop = 0, len(data)
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                assert match is not None
                a, b = match.groups()
                start = int(a) if a else max(0, len(data) - int(b))
                stop = min(len(data), int(b) + 1) if a and b else len(data)
            self.send_response(206 if requested else 200)
            self.send_header("Content-Length", str(stop - start))
            self.send_header("ETag", etag)
            self.send_header("Accept-Ranges", "bytes")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(data)}")
            self.end_headers()
            requests.append(
                {
                    "key": key,
                    "range": requested,
                    "bytes": stop - start if body else 0,
                    "signed": self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 "),
                }
            )
            if body:
                self.wfile.write(data[start:stop])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", values, requests, files
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("protocol", ["http", "s3"])
def test_remote_range_reads_are_exact_and_do_not_download_the_whole_shard(
    object_server, monkeypatch, protocol
):
    import tensorstore as ts

    def body_counter():
        return sum(
            v["value"]
            for m in ts.experimental_collect_matching_metrics(f"/tensorstore/kvstore/{protocol}")
            if m["name"].endswith("/bytes_read")
            for v in m["values"]
        )

    before = body_counter()
    url, values, requests, files = object_server
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    source = (
        HttpStore(url + "/data")
        if protocol == "http"
        else S3Store(
            "test-bucket",
            prefix="data",
            endpoint=url,
            region="us-east-1",
            credentials="environment",
        )
    )
    session = RemoteReadSession(source, max_open_arrays=1, concurrency=2, max_retries=1)
    selection = (1, slice(2, 8), slice(5, 13, 2), slice(3, 14))
    reader = session.open("velocity")
    assert not reader.backend.array.writable
    with pytest.raises(ValueError, match="write|Write"):
        reader.backend.array[0].write(np.zeros_like(values[0])).result()
    np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    assert session.open("velocity") is reader
    transfers = [r for r in requests if "/c/" in r["key"]]
    assert transfers and all(r["range"] for r in transfers)
    assert body_counter() - before == sum(r["bytes"] for r in requests)
    assert sum(r["bytes"] for r in transfers) < sum(
        len(v) for k, v in files.items() if "/velocity/c/1/" in k
    )
    if protocol == "s3":
        assert all(r["signed"] for r in requests)
    clone = pickle.loads(pickle.dumps(session))
    assert not clone._readers and clone._context is None
    assert b"test-secret" not in pickle.dumps(session)
    np.testing.assert_array_equal(clone.read("velocity", selection), values[selection])
    session.open("pressure")
    assert len(session._readers) == 1 and session.open("velocity") is not reader
    session.clear()
    assert not session._readers and session._context is None
    with pytest.raises(ValueError, match="NOT_FOUND|not exist|Not found"):
        session.open("missing")
    assert not session._readers


@pytest.mark.parametrize("protocol", ["http", "s3"])
@pytest.mark.parametrize("cache_bytes", [0, 16 << 20])
def test_fixed_remote_data_cache_avoids_repeat_requests(
    object_server, monkeypatch, protocol, cache_bytes
):
    url, values, requests, _ = object_server
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    store = HttpStore(url + "/data") if protocol == "http" else S3Store(
        "test-bucket", prefix="data", endpoint=url, region="us-east-1", credentials="environment"
    )
    session = RemoteReadSession(store, cache_bytes=cache_bytes)
    selection = (1, slice(0, 16), slice(0, 16), slice(0, 16))
    np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    first = len(requests)
    assert first > 0
    for _ in range(4):
        np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    if cache_bytes:
        assert len(requests) == first
    else:
        assert len(requests) > first
    session.clear()
    np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    assert len(requests) > first


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/data",
        "https://u:p@example.com",
        "https://example.com?token=secret",
        "https://example.com/#fragment",
        "https://example.com:bad",
        "https://example.com/hello world",
    ],
)
def test_endpoint_configuration_does_not_accept_inline_secrets_or_invalid_urls(url):
    with pytest.raises(ValueError):
        HttpStore(url)


@pytest.mark.parametrize("path", ["../private", "/absolute", "a/../b", "a//b", "a\\b"])
def test_remote_paths_cannot_escape_the_named_prefix(path):
    with pytest.raises(ValueError):
        HttpStore("https://example.com/data").spec(path)
    with pytest.raises(ValueError):
        S3Store("bucket", prefix=path)


def test_limits_and_provider_configuration_are_explicit():
    store = S3Store("bucket", prefix="study", credentials="anonymous")
    assert store.spec("fields/p")["path"] == "study/fields/p/"
    with pytest.raises(ValueError):
        S3Store("bucket", credentials="inline")
    with pytest.raises(ValueError):
        RemoteReadSession(store, concurrency=0)
    with pytest.raises(ValueError):
        RemoteReadSession(store, max_retries=-1)
    with pytest.raises(TypeError):
        RemoteReadSession(store, buffer_bytes=1.5)

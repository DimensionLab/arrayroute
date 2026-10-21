# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import pickle
import threading
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from arrayroute.remote import S3Store
from arrayroute.s3_writes import ObjectStoreProbeError, UnsafeObjectStoreError, verify_s3_writes


class Store:
    def __init__(self, mode="correct"):
        self.mode, self.data = mode, {}
        self.lock = threading.Lock()
        self.closed = False
        self.keys = []

    def put_object(self, *, Bucket, Key, Body, **conditions):  # noqa: N803
        with self.lock:
            self.keys.append(Key)
            if self.mode == "forbidden":
                raise ClientError(
                    {
                        "Error": {"Code": "AccessDenied"},
                        "ResponseMetadata": {"HTTPStatusCode": 403},
                    },
                    "PutObject",
                )
            exists = Key in self.data
            mismatch = conditions.get("IfMatch") not in {None, self.etag(Key)}
            rejected = (
                conditions.get("IfNoneMatch") == "*" and exists and self.mode != "ignore-create"
            ) or (mismatch and self.mode != "ignore-update")
            if self.mode == "race-overwrite" and Body.startswith(b"writer-"):
                rejected = False
            if rejected:
                raise ClientError(
                    {
                        "Error": {"Code": "PreconditionFailed"},
                        "ResponseMetadata": {"HTTPStatusCode": 412},
                    },
                    "PutObject",
                )
            if not (self.mode == "lost-update" and Body == b"updated"):
                self.data[Key] = Body
            return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ETag": self.etag(Key)}

    def etag(self, key):
        return '"' + hashlib.sha256(self.data[key]).hexdigest() + '"' if key in self.data else None

    def get_object(self, *, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.data[Key])}

    def delete_object(self, *, Bucket, Key):  # noqa: N803
        if self.mode == "cleanup-fails":
            raise RuntimeError("injected cleanup failure")
        self.data.pop(Key, None)

    def close(self):
        self.closed = True


def probe(monkeypatch, mode):
    import arrayroute.s3_writes as module

    fake = Store(mode)
    monkeypatch.setattr(module, "_client", lambda *_args, **_kwargs: fake)
    return fake


def test_conditional_probe_races_writes_and_cleans_up(monkeypatch):
    fake = probe(monkeypatch, "correct")
    result = verify_s3_writes(S3Store("test-bucket", prefix="project"))
    assert result.contenders == 4
    assert len(result.requests) == 9
    assert (
        sum(status == 200 for label, status in result.requests if label.startswith("contender"))
        == 1
    )
    assert fake.data == {} and fake.closed
    assert len(set(fake.keys)) == 1
    assert fake.keys[0].startswith("project/_arrayroute_probes/")


@pytest.mark.parametrize(
    "mode", ["ignore-create", "ignore-update", "race-overwrite", "lost-update"]
)
def test_unsafe_services_fail_closed_without_leaving_probe_objects(monkeypatch, mode):
    fake = probe(monkeypatch, mode)
    with pytest.raises(UnsafeObjectStoreError):
        verify_s3_writes(S3Store("test-bucket"))
    assert fake.data == {} and fake.closed


@pytest.mark.parametrize("mode", ["forbidden", "cleanup-fails"])
def test_incomplete_probe_is_never_reported_as_support(monkeypatch, mode):
    fake = probe(monkeypatch, mode)
    with pytest.raises(ObjectStoreProbeError):
        verify_s3_writes(S3Store("test-bucket"))
    assert fake.closed


def test_plain_http_requires_opt_in_before_network():
    with pytest.raises(ValueError, match="HTTP"):
        verify_s3_writes(S3Store("test-bucket", endpoint="http://127.0.0.1:9"))


def test_preparation_refuses_unsafe_storage_before_opening_or_mutating_repository(monkeypatch):
    pytest.importorskip("icechunk")
    from arrayroute.preparation import Preparation
    from arrayroute.snapshot import S3Repository, Snapshot

    fake = probe(monkeypatch, "ignore-create")
    location = S3Repository(S3Store("test-bucket"))
    with pytest.raises(UnsafeObjectStoreError):
        Preparation.create(
            Snapshot(location, "nonexistent"),
            branch="work",
            job="test",
            source="u",
            output="result",
            chunks=(4,),
            task_shape=(4,),
            halo=(0,),
            dtype="f4",
            operation="negative-v1",
            parameters={},
            attributes={},
        )
    assert fake.data == {} and fake.closed
    job = Preparation(
        location,
        "work",
        "test",
        {
            "input_snapshot": "nonexistent",
            "shape": (8,),
            "chunks": (4,),
            "task_shape": (4,),
            "halo": (0,),
        },
    )
    job._writes_checked = True
    restored = pickle.loads(pickle.dumps(job))
    with pytest.raises(UnsafeObjectStoreError):
        restored.publish()


@pytest.mark.parametrize("create,update", [(False, True), (True, False), (False, False)])
def test_endpoint_support_cannot_override_unsafe_repository_config(monkeypatch, create, update):
    ic = pytest.importorskip("icechunk")
    import arrayroute.preparation as module
    from arrayroute.snapshot import S3Repository

    config = ic.RepositoryConfig(
        storage=ic.StorageSettings(
            unsafe_use_conditional_create=create, unsafe_use_conditional_update=update
        )
    )
    repo = SimpleNamespace(config=config)
    monkeypatch.setattr(module, "_repository", lambda _location: repo)
    with pytest.raises(UnsafeObjectStoreError, match="configuration disables"):
        module._writable_repository(S3Repository(S3Store("test-bucket")))


@pytest.mark.parametrize("explicit", [False, True])
def test_safe_s3_repository_defaults_and_explicit_settings_are_accepted(monkeypatch, explicit):
    ic = pytest.importorskip("icechunk")
    import arrayroute.preparation as module
    from arrayroute.snapshot import S3Repository

    config = ic.RepositoryConfig()
    if explicit:
        config.storage = ic.StorageSettings(
            unsafe_use_conditional_create=True, unsafe_use_conditional_update=True
        )
    repo = SimpleNamespace(config=config)
    monkeypatch.setattr(module, "_repository", lambda _location: repo)
    assert module._writable_repository(S3Repository(S3Store("test-bucket"))) is repo

# SPDX-License-Identifier: Apache-2.0
"""Probe conditional S3 writes before trusting a service with transaction references.

The probe writes/deletes a unique tiny object under the supplied prefix. Passing is an
observed capability check, not proof of durability, replication or all concurrency cases.
Credentials resolve locally through the SDK; no credential-bearing object is returned.
"""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .remote import S3Store


class UnsafeObjectStoreError(RuntimeError):
    """The endpoint did not enforce a required conditional-write invariant."""


class ObjectStoreProbeError(RuntimeError):
    """Connectivity/authorization/cleanup prevented verification; do not assume support."""


@dataclass(frozen=True)
class WriteCheck:
    requests: tuple[tuple[str, int], ...]
    contenders: int


def _client(store: S3Store, *, allow_http: bool, force_path_style: bool):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if store.endpoint and store.endpoint.startswith("http:") and not allow_http:
        raise ValueError("Plain HTTP requires explicit allow_http=True")
    credentials = {}
    if store.credentials == "environment":
        for variable, parameter in [
            ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
            ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
        ]:
            value = os.environ.get(variable)
            if not value:
                raise ObjectStoreProbeError(
                    "Required S3 credential environment variables are absent"
                )
            credentials[parameter] = value
        if token := os.environ.get("AWS_SESSION_TOKEN"):
            credentials["aws_session_token"] = token
    return boto3.client(
        "s3",
        region_name=store.region,
        endpoint_url=store.endpoint,
        **credentials,
        config=Config(
            signature_version=UNSIGNED if store.credentials == "anonymous" else "s3v4",
            connect_timeout=5,
            read_timeout=10,
            retries={"max_attempts": 0},
            s3={"addressing_style": "path" if force_path_style else "auto"},
        ),
    )


def _verify(client: Any, bucket: str, key: str) -> WriteCheck:
    from botocore.exceptions import ClientError

    requests: list[tuple[str, int]] = []
    touched = False

    def put(label: str, body: bytes, **headers) -> dict:
        nonlocal touched
        touched = True
        try:
            result = client.put_object(Bucket=bucket, Key=key, Body=body, **headers)
            status = result["ResponseMetadata"]["HTTPStatusCode"]
        except ClientError as exc:
            status = exc.response["ResponseMetadata"]["HTTPStatusCode"]
            result = {}
        requests.append((label, status))
        return {"status": status, "etag": result.get("ETag")}

    def matches(body: bytes) -> bool:
        response = client.get_object(Bucket=bucket, Key=key)
        with response["Body"] as stream:
            return stream.read(len(body) + 1) == body

    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise UnsafeObjectStoreError(reason)

    try:
        first = put("create", b"initial", IfNoneMatch="*")
        if first["status"] != 200:
            raise ObjectStoreProbeError(f"Conditional-create probe returned HTTP {first['status']}")
        require(isinstance(first["etag"], str), "Conditional create did not return an ETag")
        duplicate = put("duplicate-create", b"overwrite", IfNoneMatch="*")
        require(
            duplicate["status"] == 412 and matches(b"initial"),
            "Conditional create did not preserve the existing object",
        )
        wrong = put("wrong-etag", b"overwrite", IfMatch='"00000000000000000000000000000000"')
        require(
            wrong["status"] == 412 and matches(b"initial"),
            "Conditional update accepted a nonmatching ETag",
        )
        updated = put("matching-etag", b"updated", IfMatch=first["etag"])
        require(
            updated["status"] == 200 and isinstance(updated["etag"], str) and matches(b"updated"),
            "Conditional update did not store the new value",
        )
        stale = put("stale-etag", b"overwrite", IfMatch=first["etag"])
        require(
            stale["status"] == 412 and matches(b"updated"),
            "Conditional update accepted a stale ETag",
        )
        contenders = 4
        barrier = threading.Barrier(contenders, timeout=10)

        def contend(number: int):
            barrier.wait()
            return put(f"contender-{number}", f"writer-{number}".encode(), IfMatch=updated["etag"])

        with ThreadPoolExecutor(contenders) as pool:
            results = list(pool.map(contend, range(contenders)))
        winners = [i for i, result in enumerate(results) if result["status"] == 200]
        require(
            len(winners) == 1 and all(r["status"] in {200, 409, 412} for r in results),
            "Concurrent conditional updates did not produce exactly one winner",
        )
        require(
            matches(f"writer-{winners[0]}".encode()),
            "Stored value does not match the successful conditional writer",
        )
        return WriteCheck(tuple(requests), contenders)
    finally:
        if touched:
            try:
                client.delete_object(Bucket=bucket, Key=key)
            except Exception:
                raise ObjectStoreProbeError(
                    "Could not remove the temporary S3 write probe"
                ) from None


def verify_s3_writes(
    store: S3Store, *, allow_http: bool = False, force_path_style: bool = False
) -> WriteCheck:
    """Require create-if-absent and compare-and-swap semantics, including a write race.

    Requires get/put/delete access. One random object is created inside the store prefix
    and deleted afterward. Endpoint authorization belongs to the calling application.
    No unsafe conditional-write override is offered. Run again after endpoint changes.
    """
    if not isinstance(store, S3Store):
        raise TypeError("Expected an S3Store")
    client = _client(store, allow_http=allow_http, force_path_style=force_path_style)
    key = "/".join(filter(None, [store.prefix, "_arrayroute_probes", uuid.uuid4().hex]))
    try:
        return _verify(client, store.bucket, key)
    except (UnsafeObjectStoreError, ObjectStoreProbeError):
        raise
    except Exception as exc:
        raise ObjectStoreProbeError(
            f"S3 write capability check failed ({type(exc).__name__})"
        ) from None
    finally:
        client.close()

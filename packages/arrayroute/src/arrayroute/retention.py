# SPDX-License-Identifier: Apache-2.0
"""Durable consumer references using immutable Icechunk tags.

Pins outlive Python objects and processes. They are released explicitly by the owning
application, never by a finalizer, timeout, worker exit or successful read. Administrative
tag deletion can bypass protection; this module is not an access-control boundary.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from .snapshot import (
    RepositoryLocation,
    Snapshot,
    _repository,
    _verify_storage,
    _writable_repository,
)

_PREFIX = "arrayroute-retain-v1-"
_PURPOSES = {"input", "output", "training", "report", "reader"}


class ReleasedPinError(ValueError):
    """A released retention token cannot be reused; allocate a new consumer token."""


@dataclass(frozen=True)
class RetentionPin:
    snapshot: Snapshot
    token: str
    purpose: str

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, Snapshot):
            raise TypeError("Expected a pinned Snapshot")
        if not re.fullmatch(r"[a-f0-9]{32}", self.token):
            raise ValueError("Retention token must be 32 lowercase hexadecimal characters")
        if self.purpose not in _PURPOSES:
            raise ValueError(f"Choose a retention purpose from {sorted(_PURPOSES)}")

    @property
    def tag(self) -> str:
        return f"{_PREFIX}{self.purpose}-{self.token}"


def _retain(repo, pin: RetentionPin) -> None:
    import icechunk as ic

    try:
        repo.create_tag(pin.tag, pin.snapshot.snapshot_id)
    except ic.AlreadyExistsError:
        # An uncertain response may be retried using the same persisted token. Icechunk
        # tombstones deleted tag names, which prevents stale handles releasing a new pin.
        try:
            actual = repo.lookup_tag(pin.tag)
        except ic.RefNotFoundError:
            raise ReleasedPinError("Retention token was released; create a new token") from None
        if actual != pin.snapshot.snapshot_id:
            raise ValueError("Retention token already protects another revision") from None


class RetentionManager:
    """One coordinator protects revisions before distributing readers or publishing reports.

    Persist the returned pin with the job/report. Reuse a caller-generated token to retry
    an uncertain acquire. Losing the pin may leave retained data, never silent deletion.
    Each consumer needs its own token; a copied handle represents the same consumer.
    Filesystem repositories require serialized writers. S3 mutations verify conditional
    writes and repository settings. Endpoint authorization belongs to the application.
    """

    def __init__(self, location: RepositoryLocation):
        self.location = location
        self._writes_checked = False

    def __getstate__(self) -> dict:
        return {**vars(self), "_writes_checked": False}

    def _writer(self):
        if not self._writes_checked:
            _verify_storage(self.location)
            self._writes_checked = True
        return _writable_repository(self.location)

    def retain(self, snapshot: Snapshot, *, purpose: str, token: str | None = None) -> RetentionPin:
        if snapshot.location != self.location:
            raise ValueError("Snapshot belongs to another repository")
        pin = RetentionPin(snapshot, token if token is not None else uuid.uuid4().hex, purpose)
        _retain(self._writer(), pin)
        return pin

    def verify(self, pin: RetentionPin) -> None:
        if pin.snapshot.location != self.location:
            raise ValueError("Pin belongs to another repository")
        if _repository(self.location).lookup_tag(pin.tag) != pin.snapshot.snapshot_id:
            raise ValueError("Retention reference does not match the requested revision")

    def release(self, pin: RetentionPin) -> bool:
        """Release only this consumer's reference; no snapshots or chunk bytes are deleted.

        Returns False if already released. Other consumers, branches and tags continue
        protecting their revisions. The caller must have stopped this consumer's workers.
        """
        import icechunk as ic

        if pin.snapshot.location != self.location:
            raise ValueError("Pin belongs to another repository")
        repo = self._writer()
        try:
            if repo.lookup_tag(pin.tag) != pin.snapshot.snapshot_id:
                raise ValueError("Retention reference does not match the requested revision")
            repo.delete_tag(pin.tag)
        except ic.RefNotFoundError:
            return False
        return True

    def pins(self) -> Iterator[RetentionPin]:
        """Inspect managed references without mutating storage.

        Icechunk materializes its tag-name set; cost scales with reference count, not
        array size. Concurrently released references are omitted from this inventory.
        """
        import icechunk as ic

        repo = _repository(self.location)
        for tag in sorted(repo.list_tags()):
            if not tag.startswith(_PREFIX):
                continue
            purpose, _, token = tag[len(_PREFIX) :].partition("-")
            if purpose not in _PURPOSES or not re.fullmatch(r"[a-f0-9]{32}", token):
                continue
            try:
                sid = repo.lookup_tag(tag)
            except ic.RefNotFoundError:
                continue
            yield RetentionPin(Snapshot(self.location, sid), token, purpose)

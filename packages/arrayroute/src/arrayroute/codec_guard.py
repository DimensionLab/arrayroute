# SPDX-License-Identifier: Apache-2.0
"""Verify bounded lossy blocks independently of the codec that produced them."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .quality import ErrorAssessment, assess_error

Encoder = Callable[[np.ndarray, float], bytes]
Decoder = Callable[[bytes], np.ndarray]
MAX_BLOCK_BYTES = 256 << 20


@dataclass(frozen=True)
class CodecAttempt:
    internal_tolerance: float
    error: ErrorAssessment


class CodecVerificationError(ValueError):
    """A codec failed the declared numerical or byte-identity contract."""


@dataclass(frozen=True)
class VerifiedBlock:
    """Payload and trusted receipt, kept together by the caller's artifact store.

    Hashes detect changed bytes; they do not authenticate an untrusted receipt.
    Codec identity/version and source provenance belong in that artifact store.
    """

    payload: bytes
    shape: tuple[int, ...]
    dtype: str
    unit: str
    absolute_tolerance: float
    attempts: tuple[CodecAttempt, ...]
    source_sha256: str
    payload_sha256: str
    reconstructed_sha256: str

    def decode(self, decoder: Decoder) -> np.ndarray:
        """Reject a changed payload, layout or reconstruction on subsequent reads.

        The decoder is trusted native code, not sandboxed. This checks its output
        after allocation and cannot constrain its internal allocation or runtime.
        """
        if _sha(self.payload) != self.payload_sha256:
            raise CodecVerificationError("Compressed payload hash changed")
        values = decoder(self.payload)
        _check_layout(values, self.shape, self.dtype)
        # Detach from decoder-owned buffers before checking and returning them.
        result = np.array(values, copy=True, order="C")
        if _array_sha(result) != self.reconstructed_sha256:
            raise CodecVerificationError("Reconstructed bytes changed")
        return result


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def _check_layout(values: np.ndarray, shape: tuple[int, ...], dtype: str) -> None:
    if not isinstance(values, np.ndarray) or values.shape != shape or values.dtype.str != dtype:
        raise CodecVerificationError("Codec changed the source shape or dtype")


def encode_verified(
    source: np.ndarray,
    *,
    encoder: Encoder,
    decoder: Decoder,
    absolute_tolerance: float,
    unit: str,
    max_attempts: int = 4,
    max_block_bytes: int = MAX_BLOCK_BYTES,
) -> VerifiedBlock:
    """Encode a finite float block without enlarging the requested error bound.

    A failed numerical check halves the internal codec tolerance, up to the attempt
    limit. A passing result must also reproduce identical encoded and decoded bytes
    on a second call. This checks repeatability here, not across all environments;
    ``VerifiedBlock.decode`` checks the retained reconstruction identity elsewhere.

    Original data should remain in immutable lossless storage. This opt-in derived
    representation bounds pointwise error only, not gradients or model predictions.
    Validation, retries and repeatability checks are part of its total write cost.
    Memory usage exceeds the input size: assessment uses float64 working arrays.
    Callbacks are trusted code; callback errors propagate without numerical retries.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 16:
        raise ValueError("max_attempts must be an integer between 1 and 16")
    if type(max_block_bytes) is not int or not 1 <= max_block_bytes <= MAX_BLOCK_BYTES:
        raise ValueError("max_block_bytes must be an integer between 1 and 256 MiB")
    if not isinstance(source, np.ndarray) or source.dtype.kind != "f" or source.dtype.itemsize > 8:
        raise TypeError("Source must be a float16, float32 or float64 array")
    if not source.size or source.nbytes > max_block_bytes:
        raise ValueError("Source must be nonempty and fit the block byte limit")
    if not np.isfinite(source).all():
        raise ValueError("Source must contain finite values")
    bound = float(absolute_tolerance)
    if isinstance(absolute_tolerance, bool) or not math.isfinite(bound) or bound <= 0:
        raise ValueError("Use a finite positive tolerance; use a lossless codec for zero")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Supply the physical unit, or '1' for dimensionless data")

    # The bytes owner makes this snapshot read-only even if a callback tries to
    # re-enable NumPy writes. The caller's original array is never passed to it.
    snapshot = source.tobytes(order="C")
    reference = np.frombuffer(snapshot, dtype=source.dtype).reshape(source.shape)
    shape, dtype = reference.shape, reference.dtype.str
    source_hash = _sha(snapshot)

    def round_trip(tolerance: float) -> tuple[bytes, np.ndarray]:
        values = np.frombuffer(snapshot, dtype=source.dtype).reshape(shape)
        payload = encoder(values, tolerance)
        if _sha(snapshot) != source_hash:
            raise CodecVerificationError("Codec mutated the source snapshot")
        if not isinstance(payload, bytes) or not payload:
            raise CodecVerificationError("Codec must return nonempty bytes")
        restored = decoder(payload)
        _check_layout(restored, shape, dtype)
        return payload, np.array(restored, copy=True, order="C")

    attempts = []
    internal = bound
    for _ in range(max_attempts):
        payload, restored = round_trip(internal)
        error = assess_error(reference, restored, absolute_tolerance=bound)
        attempts.append(CodecAttempt(internal, error))
        if error.passed:
            decoded_hash = _array_sha(restored)
            payload_again, restored_again = round_trip(internal)
            if payload_again != payload or _array_sha(restored_again) != decoded_hash:
                raise CodecVerificationError("Codec is not byte-repeatable")
            return VerifiedBlock(
                payload, shape, dtype, unit.strip(), bound, tuple(attempts), source_hash,
                _sha(payload), decoded_hash,
            )
        internal /= 2
        if internal == 0:
            break
    raise CodecVerificationError(
        f"No encoding met absolute tolerance {bound} {unit.strip()} after "
        f"{len(attempts)} attempts; last maximum error {attempts[-1].error.max_absolute_error}"
    )

# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import numpy as np
import pytest

from arrayroute.codec_guard import CodecVerificationError, encode_verified


def codec(source, factor=1.1):
    calls = []

    def encode(values, bound):
        calls.append(bound)
        return (values + factor * bound).astype(values.dtype).tobytes()

    def decode(payload):
        return np.frombuffer(payload, dtype=source.dtype).reshape(source.shape)

    return encode, decode, calls


def test_tightens_internal_bound_preserves_user_tolerance_and_original():
    source = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    original = source.copy()
    encode, decode, calls = codec(source)
    block = encode_verified(source, encoder=encode, decoder=decode,
                            absolute_tolerance=0.1, unit="m/s")
    assert calls == [0.1, 0.05, 0.05]
    assert block.absolute_tolerance == 0.1
    assert not block.attempts[0].error.passed
    assert block.attempts[1].error.passed
    assert block.shape == source.shape and block.dtype == source.dtype.str
    assert block.unit == "m/s"
    assert np.max(np.abs(block.decode(decode) - source)) <= 0.1
    np.testing.assert_array_equal(source, original)
    again = encode_verified(source, encoder=encode, decoder=decode,
                            absolute_tolerance=0.1, unit="m/s")
    assert again == block


def test_exhaustion_does_not_accept_an_overrun():
    source = np.zeros(2)
    encode, decode, calls = codec(source, factor=100)
    with pytest.raises(CodecVerificationError, match="after 3 attempts"):
        encode_verified(source, encoder=encode, decoder=decode,
                        absolute_tolerance=0.1, unit="K", max_attempts=3)
    assert calls == [0.1, 0.05, 0.025]


@pytest.mark.parametrize("drift", ["encoded", "decoded"])
def test_repeatability_rejects_drift_even_when_within_bound(drift):
    source = np.zeros(2)
    count = 0

    def encode(values, bound):
        nonlocal count
        count += 1
        return str(count).encode() if drift == "encoded" else b"constant"

    def decode(payload):
        return source + (count * 0.01 if drift == "decoded" else 0)

    with pytest.raises(CodecVerificationError, match="byte-repeatable"):
        encode_verified(source, encoder=encode, decoder=decode,
                        absolute_tolerance=0.1, unit="1")


def test_callback_cannot_enable_writes_to_snapshot():
    source = np.zeros(2)

    def mutate(values, bound):
        values.setflags(write=True)
        return b"never"

    with pytest.raises(ValueError, match="WRITEABLE"):
        encode_verified(source, encoder=mutate, decoder=lambda _: source,
                        absolute_tolerance=0.1, unit="K")
    assert np.all(source == 0)


@pytest.mark.parametrize("output", [None, np.zeros(3), np.zeros(2, dtype=np.float32)])
def test_malformed_decoder_outputs(output):
    with pytest.raises(CodecVerificationError, match="shape or dtype"):
        encode_verified(np.zeros(2), encoder=lambda a, b: b"data", decoder=lambda _: output,
                        absolute_tolerance=0.1, unit="K")


@pytest.mark.parametrize("output", [b"", "string", bytearray(b"x")])
def test_nonimmutable_or_empty_payload_rejected(output):
    with pytest.raises(CodecVerificationError, match="nonempty bytes"):
        encode_verified(np.zeros(2), encoder=lambda a, b: output, decoder=lambda _: np.zeros(2),
                        absolute_tolerance=0.1, unit="K")


def test_subsequent_read_detects_payload_and_reconstruction_drift():
    source = np.zeros(2)
    encode, decode, _ = codec(source, factor=0)
    block = encode_verified(source, encoder=encode, decoder=decode,
                            absolute_tolerance=0.1, unit="K")
    with pytest.raises(CodecVerificationError, match="payload hash"):
        replace(block, payload=b"changed").decode(decode)
    with pytest.raises(CodecVerificationError, match="Reconstructed bytes"):
        block.decode(lambda _: source + 0.01)
    with pytest.raises(CodecVerificationError, match="shape or dtype"):
        block.decode(lambda _: np.zeros(3))


def test_decoder_shared_buffer_is_detached():
    source = np.zeros(2)
    shared = source.copy()
    block = encode_verified(source, encoder=lambda a, b: b"data", decoder=lambda _: shared,
                            absolute_tolerance=0.1, unit="K")
    result = block.decode(lambda _: shared)
    shared[:] = 99
    assert np.all(result == 0)


@pytest.mark.parametrize("bound", [0, -1, float("nan"), float("inf"), True])
def test_invalid_bound(bound):
    with pytest.raises(ValueError, match="positive tolerance"):
        encode_verified(np.zeros(2), encoder=lambda a, b: b"data", decoder=lambda _: np.zeros(2),
                        absolute_tolerance=bound, unit="K")


@pytest.mark.parametrize("source", [np.zeros(0), np.array([np.nan]), np.array([np.inf])])
def test_invalid_source(source):
    with pytest.raises(ValueError):
        encode_verified(source, encoder=lambda a, b: b"data", decoder=lambda _: source,
                        absolute_tolerance=0.1, unit="K")


@pytest.mark.parametrize("key,value", [
    ("max_attempts", 0), ("max_attempts", 1.5), ("max_attempts", True),
    ("max_block_bytes", 0), ("max_block_bytes", True), ("max_block_bytes", 2**29),
    ("unit", " "), ("unit", None),
])
def test_invalid_options(key, value):
    options = {"absolute_tolerance": 0.1, "unit": "K", key: value}
    with pytest.raises(ValueError):
        encode_verified(np.zeros(2), encoder=lambda a, b: b"data", decoder=lambda _: np.zeros(2),
                        **options)


def test_input_byte_limit():
    with pytest.raises(ValueError, match="byte limit"):
        encode_verified(np.zeros(2), encoder=lambda a, b: b"data", decoder=lambda _: np.zeros(2),
                        absolute_tolerance=0.1, unit="K", max_block_bytes=8)


def test_nonfloat_input():
    with pytest.raises(TypeError, match="float"):
        encode_verified(np.zeros(2, dtype=np.int64), encoder=lambda a, b: b"data",
                        decoder=lambda _: np.zeros(2), absolute_tolerance=0.1, unit="K")


def test_nonfinite_reconstruction_rejected():
    with pytest.raises(CodecVerificationError, match="No encoding"):
        encode_verified(np.zeros(2), encoder=lambda a, b: b"data",
                        decoder=lambda _: np.full(2, np.nan), absolute_tolerance=0.1,
                        unit="K", max_attempts=1)


def test_subnormal_bound_does_not_retry_with_zero():
    calls = []

    def encode(values, bound):
        calls.append(bound)
        return b"data"

    with pytest.raises(CodecVerificationError, match="after 1 attempts"):
        encode_verified(np.zeros(2), encoder=encode, decoder=lambda _: np.ones(2),
                        absolute_tolerance=float(np.nextafter(0.0, 1.0)), unit="K")
    assert len(calls) == 1 and calls[0] > 0


def test_callback_failures_propagate_without_retry():
    calls = []

    def encode(values, bound):
        calls.append(bound)
        raise RuntimeError("device failed")

    with pytest.raises(RuntimeError, match="device failed"):
        encode_verified(np.zeros(2), encoder=encode, decoder=lambda _: np.zeros(2),
                        absolute_tolerance=0.1, unit="K")
    assert len(calls) == 1

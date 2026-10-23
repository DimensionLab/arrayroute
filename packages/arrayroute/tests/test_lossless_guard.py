# SPDX-License-Identifier: Apache-2.0
import zlib
from dataclasses import replace

import numpy as np
import pytest

from arrayroute.codec_guard import CodecVerificationError, encode_lossless_verified


def compress(values):
    return zlib.compress(values.tobytes(order="C"))


def decoder(source):
    return lambda p: np.frombuffer(zlib.decompress(p), dtype=source.dtype).reshape(source.shape)


@pytest.mark.parametrize("dtype", ["?", "i1", "u8", ">i8", "f2", "f4", ">f8", "c16"])
@pytest.mark.parametrize("layout", ["C", "F", "slice", "reverse", "scalar"])
def test_numeric_layouts_are_byte_preserving(dtype, layout):
    source = np.arange(24).astype(dtype).reshape(4, 6)
    if layout == "F":
        source = np.asfortranarray(source)
    elif layout == "slice":
        source = source[:, ::2]
    elif layout == "reverse":
        source = source[::-1, ::-1]
    elif layout == "scalar":
        source = source[:1, :1].reshape(())
    source.setflags(write=False)
    before = source.tobytes(order="C")
    block = encode_lossless_verified(source, encoder=compress, decoder=decoder(source), unit="1")
    assert block.decode(decoder(source)).tobytes() == before
    assert block.absolute_tolerance == 0 and block.attempts == ()
    assert block.source_sha256 == block.reconstructed_sha256
    assert block.dtype == source.dtype.str and block.shape == source.shape
    assert source.tobytes(order="C") == before
    assert block == encode_lossless_verified(
        source, encoder=compress, decoder=decoder(source), unit="1")


def test_preserves_nan_payloads_signed_zero_and_infinities():
    source = np.array([0x7FC00001, 0x7FC00002, 0xFFC00003, 0x80000000,
                       0x00000000, 0x7F800000, 0xFF800000], dtype="u4").view("f4")
    block = encode_lossless_verified(source, encoder=compress, decoder=decoder(source), unit="K")
    assert block.decode(decoder(source)).tobytes() == source.tobytes()


@pytest.mark.parametrize("bits", [[0x7FC00001, 0x7FC00002], [0x80000000, 0x00000000]])
def test_rejects_numerically_equivalent_but_different_bits(bits):
    source = np.array(bits[:1], dtype="u4").view("f4")
    changed = np.array(bits[1:], dtype="u4").view("f4")
    with pytest.raises(CodecVerificationError, match="changed source bytes"):
        encode_lossless_verified(source, encoder=compress, decoder=lambda _: changed, unit="1")


@pytest.mark.parametrize("second", [False, True])
def test_rejects_bad_reconstruction_on_either_round_trip(second):
    source = np.zeros(2)
    calls = 0

    def decode(payload):
        nonlocal calls
        calls += 1
        return source + (1 if calls == (2 if second else 1) else 0)

    with pytest.raises(CodecVerificationError, match="changed source bytes"):
        encode_lossless_verified(source, encoder=compress, decoder=decode, unit="1")


def test_rejects_nonrepeatable_payload_even_with_identical_reconstruction():
    source = np.zeros(2)
    calls = 0

    def encode(values):
        nonlocal calls
        calls += 1
        return str(calls).encode()

    with pytest.raises(CodecVerificationError, match="byte-repeatable"):
        encode_lossless_verified(source, encoder=encode, decoder=lambda _: source, unit="1")


def test_immutable_snapshot_and_decoder_buffer_detachment():
    source = np.zeros(2)

    def mutate(values):
        values.setflags(write=True)
        return b"unreachable"

    with pytest.raises(ValueError, match="WRITEABLE"):
        encode_lossless_verified(source, encoder=mutate, decoder=decoder(source), unit="1")
    assert source.tobytes() == np.zeros(2).tobytes()
    shared = source.copy()
    block = encode_lossless_verified(source, encoder=compress, decoder=lambda _: shared, unit="1")
    result = block.decode(lambda _: shared)
    shared[:] = 99
    assert result.tobytes() == source.tobytes()
    with pytest.raises(CodecVerificationError, match="Reconstructed bytes"):
        block.decode(lambda _: shared)
    with pytest.raises(CodecVerificationError, match="payload hash"):
        replace(block, payload=b"changed").decode(decoder(source))


@pytest.mark.parametrize("output", [None, np.zeros(3), np.zeros(2, dtype="f4")])
def test_rejects_decoder_layout_changes(output):
    with pytest.raises(CodecVerificationError, match="shape or dtype"):
        encode_lossless_verified(np.zeros(2), encoder=compress, decoder=lambda _: output, unit="1")


@pytest.mark.parametrize("payload", [b"", "text", bytearray(b"mutable")])
def test_rejects_invalid_payload(payload):
    source = np.zeros(2)
    with pytest.raises(CodecVerificationError, match="nonempty bytes"):
        encode_lossless_verified(source, encoder=lambda _: payload,
                                 decoder=decoder(source), unit="1")


@pytest.mark.parametrize("source", [np.zeros(0), np.array([object()]),
                                   np.array(["a"]), np.zeros(2, dtype=[("x", "f4")])])
def test_rejects_unsupported_sources(source):
    with pytest.raises((TypeError, ValueError)):
        encode_lossless_verified(source, encoder=compress, decoder=decoder(source), unit="1")


@pytest.mark.parametrize("options", [
    {"max_block_bytes": 1}, {"max_block_bytes": True}, {"max_block_bytes": 0},
    {"max_block_bytes": 2**29}, {"unit": None}, {"unit": " "},
])
def test_limits_before_calling_codec(options):
    def unexpected(_):
        pytest.fail("Invalid input reached codec")

    with pytest.raises(ValueError):
        encode_lossless_verified(np.zeros(2), encoder=unexpected, decoder=unexpected,
                                 **({"unit": "1"} | options))

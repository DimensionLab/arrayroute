# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from arrayroute.statistics import CanonicalChannelMoments

CONTRACT = hashlib.sha256(
    b'immutable test fixture, ordered rows, mask v1, pinned runtime',
).hexdigest()


def _bits(m):
    s = m.snapshot()
    return b''.join(a.tobytes() for a in (s.count, s.mean, s.m2, s.variance))


def _rewrite(blob, **updates):
    n = struct.unpack_from('<I', blob, 8)[0]
    meta = json.loads(blob[12:12+n])
    meta.update(updates)
    header = json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()
    body = blob[:8] + struct.pack('<I', len(header)) + header + blob[12+n:-32]
    return bytes(body) + hashlib.sha256(body).digest()


def _resign(body):
    return bytes(body) + hashlib.sha256(body).digest()


@pytest.mark.parametrize('stop', [0, 1, 4095, 4096, 4097, 65535, 65536, 69639])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('offset', [0.0, 1e12])
def test_restart_preserves_exact_outputs_and_cursor(stop, masked, offset):
    x = np.random.default_rng(39).normal(size=(69639, 3)) + offset
    valid = np.random.default_rng(40).random(x.shape) > .35 if masked else None
    if masked:
        valid[:9000, 1] = False
        valid[4096:8192] = False
        x[~valid] = np.nan
    expected = CanonicalChannelMoments(3)
    expected.update(x, valid)
    current = CanonicalChannelMoments(3)
    current.update(x[:stop], None if valid is None else valid[:stop])
    before = current.checkpoint(contract_digest=CONTRACT)
    current.snapshot()
    assert current.checkpoint(contract_digest=CONTRACT) == before
    restored = CanonicalChannelMoments.from_checkpoint(before, contract_digest=CONTRACT)
    assert restored.rows_seen == stop
    assert restored.checkpoint(contract_digest=CONTRACT) == before
    for start in range(stop, len(x), 1381):
        end = start + 1381
        restored.update(x[start:end], None if valid is None else valid[start:end])
    assert restored.rows_seen == len(x)
    assert _bits(restored) == _bits(expected)
    assert restored.checkpoint(contract_digest=CONTRACT) == expected.checkpoint(
        contract_digest=CONTRACT,
    )


def test_masked_nan_payload_and_signed_zero_survive():
    values = np.array([[-0., 0.], [3., np.nan]])
    values.view(np.uint64)[1, 1] = 0x7ff8000000000042
    valid = np.array([[True, True], [True, False]])
    state = CanonicalChannelMoments(2)
    state.update(values, valid)
    data = state.checkpoint(contract_digest=CONTRACT)
    restored = CanonicalChannelMoments.from_checkpoint(data, contract_digest=CONTRACT)
    assert restored._values[:2].tobytes() == values.tobytes()
    assert restored.checkpoint(contract_digest=CONTRACT) == data


def test_fresh_process_restores_and_continues(tmp_path):
    values = np.arange(50001, dtype=np.float64).reshape(-1, 3)
    values[:, 1] += 1e12
    np.save(tmp_path / 'input.npy', values)
    state = CanonicalChannelMoments(3)
    state.update(values[:5137])
    (tmp_path / 'state.bin').write_bytes(state.checkpoint(contract_digest=CONTRACT))
    script = '''
import sys
from pathlib import Path
import numpy as np
from arrayroute.statistics import CanonicalChannelMoments
p = Path(sys.argv[1])
m = CanonicalChannelMoments.from_checkpoint(
    (p / 'state.bin').read_bytes(), contract_digest=sys.argv[2],
)
x = np.load(p / 'input.npy', mmap_mode='r')
for first in range(m.rows_seen, len(x), 2000):
    m.update(x[first:first+2000])
(p / 'final.bin').write_bytes(m.checkpoint(contract_digest=sys.argv[2]))
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path), CONTRACT], check=True,
                   env={**os.environ, 'PYTHONHASHSEED': '987654'})
    state.update(values[5137:])
    assert (tmp_path / 'final.bin').read_bytes() == state.checkpoint(contract_digest=CONTRACT)


def test_repeated_resume_with_no_valid_leaves():
    state = CanonicalChannelMoments(2)
    for _ in range(11):
        state.update(np.full((9811, 2), np.nan), np.zeros((9811, 2), bool))
        state = CanonicalChannelMoments.from_checkpoint(
            state.checkpoint(contract_digest=CONTRACT), contract_digest=CONTRACT,
        )
    assert state.rows_seen == 107921
    assert not state.count.any()
    state.update(np.ones((5000, 2)))
    assert (state.mean == 1).all()
    assert (state.variance == 0).all()


@pytest.fixture
def checkpoint():
    state = CanonicalChannelMoments(2)
    state.update(np.arange(17002).reshape(-1, 2))
    return state.checkpoint(contract_digest=CONTRACT)


@pytest.mark.parametrize('index', [0, 8, 12, -40, -1])
def test_corruption_rejected(checkpoint, index):
    changed = bytearray(checkpoint)
    changed[index] ^= 1
    with pytest.raises(ValueError):
        CanonicalChannelMoments.from_checkpoint(bytes(changed), contract_digest=CONTRACT)


@pytest.mark.parametrize('digest', ['', 'x'*64, 'A'*64, 1, None])
def test_invalid_contract_rejected(digest):
    with pytest.raises(ValueError, match='SHA-256'):
        CanonicalChannelMoments(1).checkpoint(contract_digest=digest)


@pytest.mark.parametrize(('update', 'error'), [
    ({'algorithm': 'unknown'}, 'algorithm mismatch'),
    ({'contract_digest': '0'*64}, 'contract mismatch'),
    ({'runtime': {}}, 'runtime mismatch'),
    ({'channels': True}, 'dimensions'),
    ({'channels': 0}, 'dimensions'),
    ({'channels': 1_000_000_000}, 'allocation'),
    ({'rows_seen': -1}, 'dimensions'),
    ({'rows_seen': 2**64}, 'payload length|row position'),
    ({'extra': 1}, 'metadata fields'),
])
def test_incompatible_metadata_rejected(checkpoint, update, error):
    with pytest.raises(ValueError, match=error):
        CanonicalChannelMoments.from_checkpoint(_rewrite(checkpoint, **update),
                                               contract_digest=CONTRACT)


@pytest.mark.parametrize('limit', [0, -1, True, 200])
def test_size_limit_rejected(checkpoint, limit):
    with pytest.raises(ValueError, match='limit|positive integer'):
        CanonicalChannelMoments.from_checkpoint(checkpoint, contract_digest=CONTRACT,
                                               max_state_bytes=limit)


def test_empty_pending_cannot_bypass_allocation_limit():
    data = CanonicalChannelMoments(200).checkpoint(contract_digest=CONTRACT)
    assert len(data) < 10000
    with pytest.raises(ValueError, match='allocation'):
        CanonicalChannelMoments.from_checkpoint(data, contract_digest=CONTRACT,
                                               max_state_bytes=10000)


def test_truncation_append_and_invalid_payload_rejected(checkpoint):
    for data in (checkpoint[:10], checkpoint[:-1], checkpoint + b'0',
                 _resign(checkpoint[:-33]), _resign(checkpoint[:-32] + b'0')):
        with pytest.raises(ValueError):
            CanonicalChannelMoments.from_checkpoint(data, contract_digest=CONTRACT)
    n = struct.unpack_from('<I', checkpoint, 8)[0]
    for offset, data, error in [
        (12+n, struct.pack('<q', -1), 'counts'),
        (12+n+16, struct.pack('<d', float('nan')), 'moments'),
        (12+n+32, struct.pack('<d', -1), 'variance'),
        (len(checkpoint)-33, b'\x02', 'boolean'),
    ]:
        body = bytearray(checkpoint[:-32])
        body[offset:offset+len(data)] = data
        with pytest.raises(ValueError, match=error):
            CanonicalChannelMoments.from_checkpoint(_resign(body), contract_digest=CONTRACT)


def test_failed_validation_does_not_advance_cursor():
    state = CanonicalChannelMoments(1)
    state.update(np.ones((4321, 1)))
    before = state.checkpoint(contract_digest=CONTRACT)
    with pytest.raises(ValueError):
        state.update(np.full((5001, 1), np.inf))
    assert state.rows_seen == 4321
    assert state.checkpoint(contract_digest=CONTRACT) == before


@pytest.mark.parametrize('header', [b'{', b'[]', b'{"runtime":NaN}', b'\xff', b'{}'])
def test_malformed_metadata_rejected(checkpoint, header):
    n = struct.unpack_from('<I', checkpoint, 8)[0]
    body = checkpoint[:8] + struct.pack('<I', len(header)) + header + checkpoint[12+n:-32]
    with pytest.raises(ValueError):
        CanonicalChannelMoments.from_checkpoint(_resign(body), contract_digest=CONTRACT)


def test_declared_header_length_is_bounded(checkpoint):
    for length in (0, 8193, 2**32-1):
        body = checkpoint[:8] + struct.pack('<I', length) + checkpoint[12:-32]
        with pytest.raises(ValueError, match='metadata length'):
            CanonicalChannelMoments.from_checkpoint(_resign(body), contract_digest=CONTRACT)


def test_extra_whitespace_and_duplicate_keys_rejected(checkpoint):
    n = struct.unpack_from('<I', checkpoint, 8)[0]
    original = checkpoint[12:12+n]
    for header in (b' '+original, b'{"channels":2,'+original[1:]):
        body = checkpoint[:8] + struct.pack('<I', len(header)) + header + checkpoint[12+n:-32]
        with pytest.raises(ValueError, match='canonical JSON'):
            CanonicalChannelMoments.from_checkpoint(_resign(body), contract_digest=CONTRACT)


def test_nonfinite_valid_pending_observation_rejected(checkpoint):
    n = struct.unpack_from('<I', checkpoint, 8)[0]
    body = bytearray(checkpoint[:-32])
    position = 12+n+2*32
    body[position:position+8] = struct.pack('<d', float('nan'))
    with pytest.raises(ValueError, match='Nonfinite checkpoint'):
        CanonicalChannelMoments.from_checkpoint(_resign(body), contract_digest=CONTRACT)

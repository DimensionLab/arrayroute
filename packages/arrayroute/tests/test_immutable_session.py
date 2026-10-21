# SPDX-License-Identifier: Apache-2.0
import pickle

import numpy as np
import pytest
import zarr

from arrayroute.session import LocalReadSession


@pytest.mark.parametrize('immutable', [False, True])
def test_cached_data_freshness_is_explicit_and_clear_reopens(tmp_path, immutable):
    values = np.arange(32, dtype=np.float32)
    array = zarr.create_array(tmp_path / 'a', data=values, chunks=(32,))
    session = LocalReadSession('tensorstore', tensorstore_cache_bytes=1 << 20,
                               immutable=immutable)
    np.testing.assert_array_equal(session.read(tmp_path, 'a', (slice(None),)), values)
    array[:] = values + 100
    # Modifying an immutable source violates its contract; this checks the documented
    # cache behavior and that clear() is sufficient to start a new read session.
    expected = values if immutable else values + 100
    np.testing.assert_array_equal(session.read(tmp_path, 'a', (slice(None),)), expected)
    clone = pickle.loads(pickle.dumps(session))
    assert clone.immutable is immutable and clone._context is None
    session.clear()
    np.testing.assert_array_equal(session.read(tmp_path, 'a', (slice(None),)), values + 100)


def test_immutable_requires_a_boolean():
    with pytest.raises(TypeError):
        LocalReadSession(immutable='false')

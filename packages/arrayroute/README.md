# ArrayRoute

Experimental Apache-2.0 selection planning for chunked scientific arrays, usable without
Frontier. Zarr-Python, TensorStore and Zarrista adapters implement the same backend protocol.
No cloud credentials, scheduler or application account is required.

From a source checkout, install the package and the backends you need:

```sh
python -m pip install './packages/arrayroute[benchmark,test]'
python -m pytest packages/arrayroute/tests
```

The `zarr`, `tensorstore` and `zarrista` extras are also available individually. NumPy is
the only core runtime dependency. Native engines perform the storage reads and codecs;
ArrayRoute plans selections and controls its own buffers.

```python
import zarr
from arrayroute import ArrayReader, ReadStats
from arrayroute.backends import ZarrBackend

reader = ArrayReader(ZarrBackend(zarr.open_array("velocity.zarr", mode="r")))
selection = (10, slice(100, 180), slice(0, 100), slice(0, 100), slice(None))
print(reader.plan(selection))
stats = ReadStats()
velocity = reader.read(selection, stats=stats)
```

For a selection too large to materialize, `stream()` yields `(tile, values)` pairs;
`tile.destination` locates the values within the selected output. Consume and release each
buffer before requesting the next. `cancelled` accepts a callable checked between reads.

Planning is O(array rank), even for a petabyte logical shape. It does not enumerate the
global chunk grid. Subdivision is lazy and prefers existing chunk boundaries. The buffer
budget covers the conservative sum of decoded chunks and output buffers for each request,
including the destination when materializing. Backend caches, encoded I/O buffers, decoder
scratch and caller-retained buffers are additional memory; this is not a process RSS cap.
An individual chunk that cannot fit is rejected. Set backend cache/concurrency limits too.

Supported selections are integers and positive-stride slices on fixed-size arrays. No
lossy conversion is performed. Decoded-byte estimates assume full inner-chunk decoding;
partial codecs can do better. Transport bytes must be measured by the backend, not inferred
from those estimates. A large logical shape is not evidence of large physical I/O throughput.

`arrayroute.statistics.ChannelMoments` accumulates or merges channel-last population
statistics from bounded buffers, with optional per-channel validity masks. It keeps integer
counts and centred float64 moments; missing observations do not contaminate the result.
This is useful for normalization over training splits or distributed preprocessing. The
caller controls partition assignment, read buffering and the reduction order.

`arrayroute.windows.WindowIndex(lengths, history=2, unroll=4, lead=1)` maps a training-window
number to its trajectory and last input frame. Its metadata scales with trajectory count,
not window count, and random lookup takes logarithmic time. It does not shuffle or pad
distributed batches; those policies belong to the training sampler.

`arrayroute.cache.ByteLRU` limits retained values by caller-reported bytes and entry count.
It is intended for worker-local caches. Python bookkeeping and references retained outside
the cache are additional memory, and admitted values must not grow after insertion.

## Reproduce a storage pilot

```sh
python -m arrayroute.benchmark --root /tmp/arrayroute-pilot --create --frames 32 --writer zarrista
python -m arrayroute.benchmark --root /tmp/arrayroute-pilot --scan-engine tensorstore
```

The generator refuses an existing fixture directory. It stores deterministic synthetic
float32 vector fields in frame, brick and sharded layouts. Query results and full-scan
frames are SHA-256 checked; reports retain versions, source hashes and raw timings. These
are synthetic scientific buffers, not solver output. OS caches are uncontrolled, decoded
byte estimates are not transport measurements, and engine thread budgets must be matched
before using the results for a controlled comparison. See the report's method metadata.

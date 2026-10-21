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

`arrayroute.permutation.Permutation(size, seed=73, epoch=4)` supplies a reproducible
random-access order without an index array. `order[position]` returns a unique example
number and `order.inverse(example)` returns its position. Save the algorithm identifier,
size, seed, epoch and next position for replay. The canonical dataset order must remain
fixed. `order.indices(start=rank, step=world_size)` partitions positions without duplicate
padding; shard lengths can differ, so synchronized trainers must handle their tails.

The implementation uses eight keyed BLAKE2b Feistel rounds and cycle walking, based on
[Black and Rogaway's arbitrary-domain construction](https://www.cs.ucdavis.edu/~rogaway/papers/subset.pdf).
It is a sampling primitive, not encryption, and is not a uniform draw from all possible
permutations. Lookup latency varies with cycle walking. A resident NumPy/PyTorch index
array has faster per-index iteration; this trades that speed for bounded memory and fast
startup on very large datasets. No cryptographic or training-convergence claim is made.

`arrayroute.cache.ByteLRU` limits retained values by caller-reported bytes and entry count.
It is intended for worker-local caches. Python bookkeeping and references retained outside
the cache are additional memory, and admitted values must not grow after insertion.

`arrayroute.session.LocalReadSession` reuses a bounded number of array handles on local
or mounted filesystems. Choose one engine explicitly; unavailable engines fail instead
of silently changing the implementation:

```python
from arrayroute.session import LocalReadSession

session = LocalReadSession("tensorstore", buffer_bytes=256 << 20, max_open_arrays=64)
values = session.read("/data/case.zarr", "fields/velocity", (slice(10, 14), slice(None), 2))
```

The selection must match your array's rank. Reuse a session within one worker. Input
metadata stays fixed until `clear()`; serialized copies discard handles and native state.
Use multiprocessing **spawn** for native engines. Sessions are not thread-safe. The shared
TensorStore cache defaults to zero bytes; its I/O and copy concurrency default to eight
threads each across the session's handles. Both are configurable. Other engines retain
their process-level threading policies. Metadata is bounded by handle count, not bytes;
decoder scratch and library state remain outside the read-buffer budget.

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

To scan a **completed** fixture with multiple independent reader processes:

```sh
python -m arrayroute.benchmark_parallel --root /tmp/arrayroute-pilot \
  --output /tmp/arrayroute-scan-2-workers --engine tensorstore \
  --workers 2 --threads-per-worker 2
```

The output directory must be new. Spawned processes read disjoint positions in one
reproducible permutation, without padding or a shuffled index array. Every source frame
is checked against the fixture's SHA-256 manifest. A failed worker cancels its peers
between reads and prevents publication of a success report; partial per-frame logs remain.
The aggregate logical throughput includes startup, metadata, reads, checking and log writes.
It does not measure physical network bytes or claim multi-host scaling. Each worker owns
its buffer budget; hash copies, backend scratch, caches and engine threads are additional.
TensorStore copy/I/O limits apply per worker; set other engines' process-level thread
policies explicitly. Keep the fixture immutable and record CPU quotas and cache conditions
when comparing worker counts. The hash manifest itself occupies memory per process.

Measure sampling separately from physical I/O:

```sh
python -m arrayroute.benchmark_permutation --output /tmp/arrayroute-index-benchmark
# Add --torch to compare with PyTorch RandomSampler if PyTorch is installed.
```

The default benchmark samples 10,000 positions from million-, ten-million-, and
trillion-entry index domains. The trillion-entry case uses ArrayRoute only. Each repeat
runs in a fresh process; reports include raw times, RSS, versions, order hashes and source
hashes. Peak RSS includes imported libraries, which differ between engines.

Compare lossless compression on synthetic float fields, phase masks, mesh connectivity
and incompressible random bits:

```sh
python -m pip install './packages/arrayroute[codecs]'
python -m arrayroute.benchmark_codecs --output /tmp/arrayroute-codecs
# On a compatible NVIDIA CUDA 12 host, install [gpu-codecs] and include GPU candidates:
python -m arrayroute.benchmark_codecs --output /tmp/arrayroute-gpu-codecs \
  --codecs blosc-lz4-bitshuffle blosc-zstd-bitshuffle gpu-lz4-bitshuffle gpu-bitcomp
```

Each combination runs in a fresh process, warms up once, then measures repeated
encode/decode operations. GPU reports separate device-resident time from host-to-host
time including transfers. Every reconstruction is compared byte-for-byte. These are
codec experiments without disk/network timing; random bits are an incompressibility
control. Analytic fields are not simulated CFD. CUDA dependencies are optional and
subject to their upstream platform and licensing requirements.

## Export a selection larger than RAM

```python
from arrayroute.export import save_npy

save_npy(reader, selection, "velocity-region.npy", max_output_bytes=8 << 30)
```

This writes an ordinary `.npy` file through bounded tiles, preserving dtype, shape,
endianness and numerical bytes. The explicit output limit prevents accidentally
exporting a much larger region. A temporary file is published only after a successful
write; existing files and symlinks are never replaced. Cancellation and errors remove
the temporary file. Publication requires same-directory hard links; unsupported
filesystems fail. Abrupt process termination can leave a hidden `.partial` file.

Explicit buffers use the reader's budget plus a configurable 1 MiB write buffer; backend
scratch and OS page caches are additional. No whole-output memory map or materialization
is used. NPY stores numerical metadata, not physical units or provenance; retain your
scientific manifest alongside the export. The API accepts `stats` and a `cancelled`
callback like the streaming reader.

## Validate reconstruction accuracy

```python
from arrayroute.quality import ErrorAccumulator

errors = ErrorAccumulator(absolute_tolerance=0.001)  # same units as the source field
for reference_block, decoded_block in paired_blocks:
    errors.update(reference_block, decoded_block)
print(errors.result.passed, errors.result.max_absolute_error, errors.result.relative_l2)
```

This independently checks the requested absolute tolerance without percentage slack.
Shapes/dtypes and nonfinite positions must agree; NaN payloads and signed zero are not
byte-preservation requirements in this numerical assessment. For lossless storage use
byte comparisons or hashes as well. Measurements use float64, not interval arithmetic.
Overflow in error measurement raises an error. Relative L2 is `None` when undefined or
unrepresentable; counts expose empty/all-nonfinite regions.

Accumulators retain statistics, not input arrays. Workers can merge assessments computed
with the same tolerance over disjoint blocks. Comparison scratch is proportional to the
supplied block, so keep blocks bounded. This checks field reconstruction only; accuracy
of gradients, integrals, spectra or other derived quantities needs its own assessment.

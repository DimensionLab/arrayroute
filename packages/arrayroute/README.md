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
budget covers peak live decoded chunks and output buffers for each request,
including the destination when materializing. Backend caches, encoded I/O buffers, decoder
scratch and caller-retained buffers are additional memory; this is not a process RSS cap.
An individual chunk that cannot fit is rejected. Set backend cache/concurrency limits too.

Supported selections are integers and positive-stride slices on fixed-size arrays. No
lossy conversion is performed. Decoded-byte estimates assume full inner-chunk decoding;
partial codecs can do better. Transport bytes must be measured by the backend, not inferred
from those estimates. A large logical shape is not evidence of large physical I/O throughput.

`ReadPlan.decoded_bytes_upper_bound` measures total decode work; it is separate from
`peak_decoded_bytes_upper_bound`, which determines memory admission together with the output.
By default all touched chunks are charged as live. A backend may declare an
`ArraySpec(..., decode_window=DecodeWindow(max_chunks=1, strided_only=True))` only when its
implementation releases decoded chunks before requesting successors. ArrayRoute's Zarrista
adapter supplies this guarantee for strided partial-shard reads. Contiguous bulk reads,
unsharded reads and outer-codec shards retain conservative accounting. Other adapters do
not inherit this guarantee. Total decode work and amplification remain unchanged.

Zarrista strided reads gather only touched inner chunks. When partial shard decoding is
available, the adapter reuses one shard index while gathering those chunks and discards
it before moving to another shard or returning from the read. It completes all selected
chunks within each shard before visiting the next, retaining the original selection's
stride phase and output order. Shard regions are generated lazily without a global chunk
inventory. Index bytes and native
bookkeeping are additional to the read-buffer allowance; no field payload is cached by
this mechanism. Reads after a completed external write create fresh indexes. Concurrent
source mutation still requires snapshot isolation supplied by the caller. An outer
shard codec uses whole-shard accounting and the ordinary subset reader.

`arrayroute.statistics.ChannelMoments` accumulates or merges channel-last population
statistics from bounded buffers, with optional per-channel validity masks. It keeps integer
counts and centred float64 moments; missing observations do not contaminate the result.
This is useful for normalization over training splits or distributed preprocessing. The
caller controls partition assignment, read buffering and the reduction order.
Its floating-point result depends on grouping; it is not a bitwise-reproducible
distributed reduction.

For ordered normalization, use `CanonicalChannelMoments(channels)` from the same
module. It reduces fixed 4096-row leaves in stream order, regardless of the sizes
of buffers passed to `update(values, valid=None)`. It exposes `count`, `mean`,
`variance` and `mean_square`; `snapshot()` returns an independent result without
changing subsequent reduction boundaries. Validity is per channel. C-contiguous,
Fortran-contiguous and strided inputs follow the same C row order. The persistent
leaf buffer is 4096 × channels × 9 bytes, plus the channel accumulators; conversion
and validation of a caller buffer may need additional temporary memory.
Accumulation stays relative to each channel's first valid observation, avoiding
loss of small between-leaf differences when physical values have a large offset.
Complete leaves are processed in batches of at most 16, with an 8 MiB converted
value-buffer target (one complete leaf is the minimum). Temporary arrays can use
several multiples of that buffer; this is not an RSS cap. Fully observed input
uses a mask-free path. Both paths preserve each leaf's arithmetic and its ordered
fold, so execution batching does not change the numerical algorithm identifier.
Masked input skips leaves with no observations and shares the fast path when the
remaining leaves are fully observed.

Save `CanonicalChannelMoments.ALGORITHM`, the ordered immutable input identities,
NumPy version and numerical execution environment. This contract guarantees read
partition independence in the same numerical runtime, not arbitrary input order
or unverified CPU/GPU equivalence. Parallel readers must deliver global stream
order before reduction. Do not merge independently accumulated partial streams
or restore a reduction from its snapshot: a snapshot does not contain the pending
leaf needed to continue exactly.

For restartable reductions, `checkpoint(contract_digest=...)` returns a canonical
binary artifact containing the committed moments, first-observation origins,
pending values and validity mask, and the consumed `rows_seen` cursor. The digest
must be the lowercase SHA-256 of your saved pipeline contract: include immutable
input revisions/content hashes, row order, channel selection, units/transforms,
mask rules and the pinned numerical environment. A path or random run ID is not
an input identity. The reducer checks the supplied digest; it does not independently
inspect your sources or construct that contract for you.

```python
checkpoint_bytes = moments.checkpoint(contract_digest=pipeline_digest)
# Atomically publish checkpoint_bytes to durable storage before acknowledging progress.
# After interruption, load only the last committed object, not a temporary upload.
moments = CanonicalChannelMoments.from_checkpoint(
    checkpoint_bytes, contract_digest=pipeline_digest,
)
next_row = moments.rows_seen  # includes masked rows; excludes unsubmitted prefetch
# Resume the same transformed stream at next_row, using any convenient read size.
```

Restoration checks the format, complete-artifact SHA-256, algorithm, contract,
NumPy/Python versions, OS, machine architecture, byte order, dimensions and state
invariants before returning independent writable arrays. The basic runtime fields
are early mismatch checks, not a full environment fingerprint: pin library builds,
kernel dispatch and relevant settings in your contract. Different machines are
not automatically certified as bitwise equivalent. There is no bypass flag for a
contract/runtime mismatch. The default `max_state_bytes=64 << 20` bounds input
length and persistent accumulator storage independently; increase it explicitly
for unusually wide channel vectors. It is not a total-process memory limit.

The format stores little-endian integer/float arrays and canonical JSON, never
pickle or executable objects. It excludes unused buffer capacity and timestamps;
equal state under the same contract/runtime has equal checkpoint bytes. SHA-256
detects accidental corruption; authenticate storage separately if writers are
untrusted. The caller owns atomic storage publication, source verification and
cursor-based stream replay. This API does not checkpoint a whole training job.

Measure a real local time-first, channel-last Zarr v3 array with
`python -m arrayroute.benchmark_statistics --store /path/to/array.zarr --output results.json`.
The benchmark hashes every input pass, checks exact canonical replay across read
sizes, and records read, reduction and hashing time separately. It does not flush
the filesystem cache or establish cross-platform reproducibility.

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

`arrayroute.permutation.BatchPartition` makes that tail policy explicit for batched
consumers without materializing an index array:

```python
from arrayroute.permutation import BatchPartition

partition = BatchPartition(
    size=10**12, rank=7, world_size=128, batch_size=8,
    seed=73, epoch=4, tail="drop", start_batch=0,
)
for indices in partition:
    batch = dataset.read(indices)  # your dataset adapter
    train(batch)
```

`tail="drop"` excludes only the incomplete global batch; every rank has equally many
full batches, without duplicate padding. `dropped_samples` reports the global excluded
count. `tail="uneven"` preserves every sample once but can leave unequal rank batch
counts: distributed collective callers must handle uneven participation explicitly.
If the entire dataset is smaller than a global batch, drop mode yields zero batches.

The iterator can also be a PyTorch `DataLoader(batch_sampler=partition)` without a
Torch dependency in ArrayRoute. All ranks must use the same canonical dataset order,
source revision, size, seed, epoch, world size, batch size and tail policy. Construct a
new plan for a new epoch. Save `partition.state_dict()` with the checkpoint and restore
using `BatchPartition.from_state_dict(...)`. Set `start_batch` to the number of
**consumed** per-rank global steps; prefetching does not advance it. Save dataset identity,
optimizer state and framework RNG state separately. Replay does not support changing
rank topology mid-epoch. Uneven ranks that finish early can still restore the shared
global consumed-step position. `sample_count` and `len(partition)` describe this rank's
remaining samples and batches.

The plan has constant retained state; iteration retains at most one batch of indices.
`python -m arrayroute.benchmark_partition --output /tmp/rank-index-benchmark` measures
large-domain rank planning and late resume, explicitly **not physical I/O throughput**.

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

`immutable=True` explicitly asserts that stored values also stay fixed for the session's
lifetime. TensorStore then skips freshness checks when reusing cached chunks. It defaults
to `False`; metadata/handle reuse alone does not opt into stale data. Call `clear()` after
changing inputs. This flag provides neither a write lock nor a snapshot; use pinned
snapshot sessions when a concurrent writer must be supported. The native cache size is a
**soft** eviction limit, not a hard process-memory bound. See TensorStore's
[cache and freshness options](https://google.github.io/tensorstore/python/api/tensorstore.TensorStore.spec.html).

### Share native resources across reader threads

`TensorStoreReadGroup` gives independent worker sessions one native cache and I/O/copy
thread budget. Create the group once, then create a session inside each worker:

```python
from arrayroute.execution import OrderedPool
from arrayroute.session import TensorStoreReadGroup

group = TensorStoreReadGroup(cache_bytes=128 << 20, threads=8, immutable=True)

def make_reader():
    session = group.session(buffer_bytes=64 << 20, max_open_arrays=16)
    return lambda selection: session.read("/data/case.zarr", "fields/velocity", selection)

with OrderedPool(make_reader, workers=8, max_pending=8) as pool:
    with pool.map(selections) as values:
        for value in values:
            consume(value)
```

The 128 MiB native cache soft limit applies to the whole group. Each worker still owns
its handle cache and output buffers; retained results, decoder scratch and Python
metadata are additional memory. This uses TensorStore's documented
[shared resource context](https://google.github.io/tensorstore/context.html).
It does not share mutable Python session state or promise a speedup for every workload.

Create groups after spawning a process. Groups and their sessions reject serialization
and cross-process reuse. Clearing one session drops its handles while preserving the
group's native cache. To refresh changed inputs, release all old sessions and construct
a new group; `immutable=True` remains a caller assertion, not snapshot isolation.

## Plan reader concurrency before a pilot

More workers divide a fixed memory allowance into smaller buffers. That can turn one
full-region read into many tiled requests. Inspect the tradeoff using metadata only:

```python
from arrayroute import ArraySpec
from arrayroute.concurrency import plan_concurrency

spec = ArraySpec((43691, 128, 128, 128, 3), (1, 32, 32, 32, 3), 4)
options = plan_concurrency(
    spec, (0, slice(None), slice(None), slice(None), slice(None)),
    total_buffer_bytes=256 << 20, max_workers=8,
)
for choice in options.candidates:
    print(choice.workers, choice.materialization, choice.requests_per_value_lower_bound)
```

For this layout, a frame needs 48 MiB for output plus the decoded-chunk estimate.
Four or five readers can each issue one full-frame request; eight readers need eight
tile requests per frame. Candidates include powers of two and the untiled boundary.
Benchmark the candidates on the intended storage; this is not a throughput prediction.
For variable requests, plan the largest memory requirement, including chunk alignment
and layout as well as shape, or plan each request bucket separately.

Planning does not read values or enumerate the dataset. Tile counts stop after a
bounded prefix and report a lower bound when `request_count_exact` is false. A selection
whose output cannot fit is marked `stream-required`; a decoder chunk that cannot fit
is marked `chunk-too-large`. Cache, decoder scratch, encoded transport and values retained
outside the reader are additional memory. The plan assumes one active value per reader.

## Read remote arrays

Install the optional `tensorstore` extra to read immutable Zarr v2/v3 arrays over HTTP or
S3-compatible storage:

```python
from arrayroute.remote import RemoteReadSession, S3Store

session = RemoteReadSession(
    S3Store("engineering-data", prefix="snapshots/revision-42", region="eu-central-1"),
    buffer_bytes=256 << 20, concurrency=8, cache_bytes=0,
)
reader = session.open("case-17/velocity")
region = reader.read((10, slice(20, 60), slice(30, 70), slice(10, 50), slice(None)))
```

The default is Zarr format 3. Set `zarr_format=2` for format 2 archives; the directory
name or URL suffix does not determine the stored format. Selection, byte budgeting,
read-only access and worker serialization have the same contract for both formats.

For a private S3-compatible service, supply `endpoint="https://storage.example.org"`
and its region. Credentials are resolved by TensorStore inside each worker using its
default chain, or explicitly `credentials="environment"` or `"anonymous"`. Credential
values are not stored in the session or its serialized state. Use least-privilege read
access. Array handles are opened read-only; this API never creates or writes objects.

`HttpStore("https://data.example.org/revision-42")` reads public/static endpoints. URLs
with embedded credentials or query tokens are rejected; authenticated HTTP headers are
not supported by this adapter. Prefer HTTPS; HTTP is available for explicitly trusted
local/on-premises endpoints. These are client-side connectors, not an SSRF protection
boundary. An application accepting endpoint configuration from others must authorize
destinations before opening them server-side.

Handles are bounded and worker-local, with one shared native cache/concurrency context.
These remote sessions require immutable source objects. With a nonzero cache soft limit,
repeated cached data reads skip freshness validation requests. `clear()` drops the cache;
this is not a consistency mechanism for a concurrently changing prefix.
Use spawn for separate workers; serialization drops handles and native context. Retry
limits must be positive on the tested TensorStore runtime. Cancellation remains between
reads and does not interrupt an in-flight native request. Use immutable, versioned
prefixes: cache clearing does not create a snapshot of changing objects.

Sharded range reads require a server that correctly implements HTTP ranges and validators.
The tests compare native received-byte counters with an instrumented server and verify
that a small selection transfers less than a full shard. Those counters measure response
bodies, not TCP/TLS overhead, and are process-wide. Actual transfer depends on codec,
layout, endpoint behavior, caching and retries. Read-buffer budgets do not include native
encoded-response buffers or decoder scratch.

Driver contracts: [HTTP](https://google.github.io/tensorstore/kvstore/http/index.html)
and [S3](https://google.github.io/tensorstore/kvstore/s3/index.html).

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

For a large engine comparison on a shared layout, select only the representation you
intend to read instead of paying to materialize three copies:

```sh
python -m arrayroute.benchmark --root /data/sharded-study --create --create-only \
  --frames 43691 --edge 128 --layouts sharded --writer zarrista --frame-workers 4
```

This materializes just over 1 TiB of logical float32 values in one sharded array; disk
bytes depend on compression and filesystem allocation. It is a requested experiment
size, not a previously measured throughput claim. All supported read engines can use
the same array. Receipts record the selected layouts and actual writer concurrency.
Invalid or duplicate layout choices fail before creating the output directory.

The generator saves `materialization.json` before counting physical files. For very
large fixtures, numerical verification can proceed independently of that inventory:

```sh
python -m arrayroute.benchmark_materialized --root /tmp/arrayroute-pilot \
  --output /tmp/arrayroute-independent-scan --engine tensorstore
```

This command is specific to the synthetic generator above. It validates the completed
write progress and array shape, then regenerates each frame from its seed and checks
the stored bytes. It retains per-frame expected/actual hashes, read times, generation
times and hashing times separately, with a constant-state traversal permutation.
`--limit` creates an explicitly partial pilot; `--deadline` stops between frames at
the supplied Unix timestamp. A partial or failed scan never reports full verification.
An existing output directory is refused. Keep the source immutable and suspend competing
I/O for comparative measurements; the command does not stop other processes or control
the OS cache. Its memory policy covers two frame arrays, not native decoder scratch.

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

## Spatial grid storage policy

`arrayroute.layout.GridLayoutPolicy` chooses bounded spatial chunks and optional Zarr v3
shards without inspecting values or enumerating the chunk grid:

```python
import zarr
from arrayroute.layout import GridLayoutPolicy

layout = GridLayoutPolicy().plan(
    (20, 128, 128, 128, 3),
    itemsize=4,
    spatial_axes=(1, 2, 3),
    time_axis=0,
)
# chunks=(1,32,32,32,3), shards=(1,128,128,128,3)
group = zarr.open_group("velocity.zarr", mode="w-")
array = group.create_array(
    "velocity", shape=(20,128,128,128,3), dtype="float32",
    chunks=layout.chunks, shards=layout.shards,
)
```

The defaults target 32-cell chunk edges, 128-cell shard edges, at most 512 KiB decoded
per chunk and 32 MiB decoded per shard, including edge padding. Byte limits take priority
over edge targets and requested time steps per chunk. Small arrays remain unsharded.
The policy reports an upper bound on storage-object count; it does not predict compressed
size, temporary codec memory or transport bytes. This is a configurable balanced heuristic,
not an automatically optimal layout. Full-volume training, sparse regions, object stores
and local filesystems can favor different choices. The policy neither converts values nor
selects a compressor; lossless versus lossy encoding remains explicit.

## Pin reads to a dataset revision

Install `arrayroute[snapshots]` on **Python 3.12 or newer** for the optional Icechunk 2.3
adapter. The base package still supports Python 3.11; the snapshot extra does not install
Icechunk there. Resolve a branch once before dispatching training/preprocessing workers:

```python
from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession

revision = Snapshot.resolve(FilesystemRepository("simulation.icechunk"), branch="main")
print(revision.snapshot_id)  # save this in the job's input manifest
session = SnapshotReadSession(revision, buffer_bytes=64 << 20, max_open_arrays=16)
velocity = session.read("velocity", (10, slice(20, 40), slice(None), slice(None), slice(None)))
```

`Snapshot(location, saved_id)` reopens a known revision. Cache eviction, `clear()` and
spawned worker serialization retain that ID; they do not follow a newer branch. Reads are
read-only. `session.attributes(path)` reads metadata from that same revision. Use one
session per worker and multiprocessing **spawn**, not fork.

Local snapshot readers use a 1 MiB target before splitting one object's read into parallel
ranges. This reduces small-file operations; it does not enlarge the requested selection or
read extra field values. Remote readers retain the storage backend's policy. Override with
`SnapshotReadSession(revision, range_request_bytes=...)` when measuring another workload;
`effective_range_request_bytes` reports the local setting or `None` for the backend policy.
The setting is session-local, survives worker serialization, and does not modify repository
configuration. It is separate from the decoded-buffer and cache budgets.

For S3-compatible storage, use `S3Repository(S3Store(...))` from `arrayroute.snapshot`
and `arrayroute.remote`. Credentials resolve inside each worker; descriptors contain no
keys. Plain HTTP requires explicit opt-in. This is a trusted client adapter, not an
endpoint authorization boundary. External virtual chunks are denied: repository snapshots
do not make mutable external objects immutable. The repository owner must retain the
referenced revision; this adapter does not provide backup, reader leases or garbage
collection protection. Icechunk's local filesystem backend must not have concurrent
committers; use an object store with the required conditional operations for that workflow.

Snapshots add version semantics, not an automatic speedup. The portable
`python -m arrayroute.benchmark_snapshots OUTPUT` experiment compares fresh-process reads
with plain Zarr-Python and TensorStore, checks every frame against independent hashes, and
records the extra storage for one changed chunk. It requires the `benchmark` extra on
Python 3.12+, never overwrites an existing output directory, and uses synthetic values.

`python -m arrayroute.benchmark_snapshot_pipelines --fixture OUTPUT --output NEW_DIRECTORY`
compares codec batch sizes and Zarr's experimental fused pipeline using that fixture.
It requires Zarr 3.3+, isolates each configuration in a fresh process, and verifies every
returned frame. Configuration is global to each benchmark worker; this command does not
change the reader library's defaults. Compare the recorded read times; open times have
different optional-import costs. Neither experiment establishes performance at cluster scale.

## Prepare a revision with recoverable parallel tasks

`arrayroute.preparation` uses the same `snapshots` extra. It is an experimental,
shape-preserving numerical preprocessing API. A task owns complete output chunks; read
halos can overlap for stencils such as gradients. A single coordinator commits completed
output chunks and their SHA-256 receipts together. A separate work branch keeps partial
output out of the published revision. Failed or lost tasks have no committed receipt and
are retried after reopening the job.

For a four-dimensional scalar field `(time, x, y, z)`, this computes its x derivative with
2 m spacing. Put this in an importable Python script with a `__main__` guard for spawn:

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from arrayroute.preparation import Preparation
from arrayroute.snapshot import FilesystemRepository, Snapshot


def velocity_gradient(values):
    return np.gradient(values, 2, axis=1)


def main():
    source = Snapshot.resolve(FilesystemRepository("simulation.icechunk"))
    job = Preparation.create(
        source,
        branch="prepare-velocity-gradient", job="velocity-gradient",
        source="velocity_x", output="du_dx",
        chunks=(1, 32, 32, 32), task_shape=(1, 64, 64, 64), halo=(0, 1, 0, 0),
        dtype="float32", operation="numpy-gradient-x-v1", parameters={"spacing_m": 2},
        attributes={"units": "1/s", "dims": ["time", "x", "y", "z"]},
    )
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
        job.run(velocity_gradient, pool,
                batch_tasks=16, buffer_bytes=128 << 20)
    revision = job.publish()
    print(revision.snapshot_id)  # use this immutable ID as a downstream dataset input


if __name__ == "__main__":
    main()
```

After interruption, use `Preparation.resume(location, branch=..., job=...,
operation="numpy-gradient-x-v1", parameters={"spacing_m": 2})`, then `run` with the same
operator. `pending()` streams task IDs in bounded progress blocks. `PreparationError`
reports failed IDs and the checkpoint retaining successful tasks. `run` returns the number
of newly completed tasks. A cancellation callback is checked between batches, without
killing native operations mid-write. Executors and worker cleanup belong to the caller.
Thread executors are also supported and can be faster than processes for small operations.

Each operator must return the halo-expanded input shape and the declared dtype. The
framework crops the owned core. You must choose appropriate halos, spacing, output units,
and physical boundary treatment; this API does not infer boundary conditions. Workers
reuse one pinned input reader per thread, with bounded handles and disabled chunk caches.
Budget checks cover planned read buffers, transform output and write copies; custom
operator scratch, native codec caches and total process RSS require additional headroom.
Publication also has a `buffer_bytes` limit and rereads every output core to verify its
receipt. Hashes verify output integrity, not the scientific correctness of the operator.

**Operational contract:** one trusted coordinator per work branch and trusted operator
code/workers. This is not an access-control boundary, sandbox, distributed lock service,
or code fingerprinting system. Operation name/configuration are caller-supplied provenance.
New outputs are unsharded so task ownership is unambiguous. All workers need access to the
repository; local cooperative writes have been tested, multi-host object-store writes
remain to be validated. Never run concurrent committers against a filesystem repository.
A failed initialization may leave a reserved staging branch; inspect it before cleanup.
Uncommitted failed worker chunks can remain unreachable until repository maintenance.
Preparation now retains its input and verified output using durable revision tags.
Persist `job.input_pin` and `job.output_pin(published)` with the owning job/dataset before
removing its work branch. References remain until the owner explicitly releases them.

Run `python -m arrayroute.benchmark_preparation --root NEW_DIRECTORY` for a reproducible
synthetic pilot comparing commit batches and thread/spawn executors, plus a forced failure
and restart. It retains raw timings, revision IDs, and independently checked hashes. This
is a local experiment; it does not claim TB/PB throughput or a universal executor winner.

### Check S3 write semantics before preprocessing

S3 read compatibility does not establish transaction safety. `Preparation.create` now
probes the target S3 endpoint **before opening or mutating the repository**. A resumed
preparation checks again before its first write. Filesystem preparations retain their
single-committer contract. There is no switch to silently disable conditional writes.
Writes also reject repositories whose stored configuration explicitly disables conditional
creation or updates, even when the endpoint passes. Read-only inspection remains available.

The check creates one random tiny object under `<prefix>/_arrayroute_probes/`, exercises
create-if-absent, matching/stale ETags, and four competing conditional updates, verifies
the winning bytes, and deletes its probe. It requires get/put/delete access and uses Boto3
from the `snapshots` extra. It is also available independently:

```python
from arrayroute.remote import S3Store
from arrayroute.s3_writes import verify_s3_writes

check = verify_s3_writes(S3Store("simulation-data", prefix="research", region="eu-west-1"))
print(check.requests)  # labels and HTTP statuses; no credentials
```

`UnsafeObjectStoreError` means a required invariant failed. `ObjectStoreProbeError` means
verification was unavailable, including access/connectivity or cleanup failure. Both block
preparation. One live test found Garage 2.4.1 accepted an overwrite despite `If-None-Match: *`;
that tested configuration was refused for transactional preparation. Read-only adapters are
unaffected. Passing this probe is an observation, not proof of persistence, replicated
consensus, all race conditions or multipart-upload semantics. Recheck after endpoint changes;
the result is cached only in the current preparation object, never persisted as a certificate.

### Keep revisions available to jobs and reports

An immutable snapshot ID alone does not prevent cleanup. Retention references protect
the exact snapshot and its reachable chunks using Icechunk v2 tags. Older v1 repositories
remain readable, but must be migrated with the engine's tools before using retention:

```python
from arrayroute.retention import RetentionManager
from arrayroute.snapshot import SnapshotReadSession

retention = RetentionManager(snapshot.location)
pin = retention.retain(snapshot, purpose="training")
# Persist pin with the training job before distributing snapshot to workers.
reader = SnapshotReadSession(pin.snapshot)
values = reader.read("velocity", (0, slice(None), slice(None), slice(None), 0))
retention.verify(pin)
# Release only after this consumer and all its workers no longer need the revision.
retention.release(pin)
```

Each consumer uses its own token. Multiple jobs/reports can protect the same revision;
releasing one does not release the others. A reference survives process exit and pickle
round trips. Nothing releases it implicitly. For retryable orchestration, allocate and
persist a 32-character lowercase hex token before calling `retain(..., token=token)`.
Reusing that token for the same snapshot is idempotent; retargeting is refused. Deleted
tokens cannot be reused because Icechunk tombstones tag names. `pins()` inventories managed
references; it scales with tag count, and the underlying engine materializes the tag set.
The application stores owner names, tenant permissions and lifecycle state in its catalog.

Preparation configuration version 2 stores separate input/output tokens. Input protection
is acquired before creating the work branch. Publication verifies all output bytes,
commits completion, then creates the output tag; an interrupted tag response can be retried.
The work branch protects that interval. A failed initialization can leave an input pin;
inspect the inventory when reconciling failed jobs rather than expiring it automatically.
Older preparation configurations must be recreated; completed ordinary snapshots are readable.

Cleanup must preserve tags and branch tips (`delete_expired_tags=False`,
`delete_expired_branches=False`). Physical garbage collection is a separate administrative
operation. Its cutoff must predate all in-flight writes; coordinate active writers before
reclamation. These references do not authorize unsafe cutoff selection, bypassing tags,
deleting external virtual objects, or deleting a bucket. They are not backups or expiring
leases. ArrayRoute does not start a background destructive cleaner.

## Overlap reads with processing

For a synchronous consumer, `ArrayReader.prefetch()` overlaps the next tile's read
with processing the current tile. It preserves tile order and exact values. Use a
context so early exit, cancellation or consumer exceptions join the worker and release
the generator. Native blocking I/O must finish before the context can close.

```python
from arrayroute.session import LocalReadSession

session = LocalReadSession("tensorstore", buffer_bytes=64 << 20, immutable=True)
reader = session.open("simulation.zarr", "velocity")
with reader.prefetch((slice(None),) * len(reader.backend.spec.shape)) as blocks:
    for tile, values in blocks:
        process(tile.destination, values)
        del values  # release the consumer reference before requesting another tile
```

The explicit budget is divided into two equal read slots, each covering returned
values plus the decoded-chunk estimate. At least one decoded chunk must fit in a slot.
This excludes backend caches, codec scratch, consumer processing and arrays retained
after their iteration. One worker owns backend reads; do not concurrently use that
backend/session elsewhere. Cancellation callbacks must be thread-safe. No unbounded task queue is created. Read statistics count
completed prefetches even if their values are not consumed. Storage exceptions propagate
when the consumer reaches the failed read.

This is optional; storage engines already have internal concurrency, so added overlap
may help a compute-heavy consumer or may add overhead. `python -m
arrayroute.benchmark_prefetch --root /tmp/read-overlap` compares identical serial and
prefetched tiles, verifies fixture hashes and every resulting stream digest/reduction,
and records first-party Zarr/TensorStore results without a universal speed claim.

## Bounded parallel frame processing

`arrayroute.execution.ordered_map` runs independent reads or transforms with one
worker-owned callable per active thread. The input is consumed lazily. A slow first
frame cannot be overtaken, errors are propagated at their input position, and an
infinite input cannot grow an unbounded submission queue.

```python
from arrayroute.execution import ordered_map
from arrayroute.session import LocalReadSession


def make_processor():
    session = LocalReadSession("tensorstore", buffer_bytes=64 << 20, immutable=True)
    reader = session.open("simulation.zarr", "velocity")

    def process(frame):
        selection = (frame,) + (slice(None),) * (len(reader.backend.spec.shape) - 1)
        values = reader.read(selection)
        return frame, values.mean(axis=tuple(range(values.ndim - 1)))

    return process


with ordered_map(make_processor, range(frame_count), workers=4, max_pending=4) as results:
    for frame, mean_velocity in results:
        record(frame, mean_velocity)
```

The factory constructs native handles on the thread that uses them. Do not return a
shared mutable reader from the factory. At most `max_pending` submitted tasks/results
are owned by the context, including its currently yielded result. Explicit array
buffers scale with the worker budgets and task outputs; backend caches, codec scratch,
and values retained by the caller are additional. This is not a process-RSS limit.
Factories and cancellation callbacks must be thread-safe.

Use the context manager on early exit. Closing cancels queued tasks and joins running
threads, which releases their worker callables. A blocking native call cannot be killed
by Python thread cancellation; use a process deadline for a hard wall-clock bound.
Results and errors are never silently skipped. A source-iterator error stops iteration.

The synthetic materialization benchmark accepts `--frame-workers 4`. Each task owns an
entire time shard and generates its own seeded frame; workers never update the same
shard. Layouts are written sequentially within each task. The independent scanner accepts
`--workers 4`; its 256 MiB read-buffer budget is divided across workers, with one
regenerated reference frame per worker added separately. Reports record concurrency and
memory policy. Component times sum worker time; `verified_wall_GB_s` uses actual elapsed
time including reference generation, hashing and evidence writes. Neither this benchmark
nor parallel execution establishes a universal speedup.

## Indexed metadata snapshots

`arrayroute.catalog` stores keyed JSON records in a standard SQLite file. Use it when
opening a complete JSON manifest becomes expensive: a lookup uses the unique-key index,
and pagination uses the integer primary key without scanning earlier pages. It has no
solver-specific schema and does not change or compress the arrays referenced by records.

```python
from arrayroute.catalog import Catalog, write_catalog

receipt = write_catalog(
    "simulation-catalog.sqlite",
    ((case.id, case.metadata) for case in cases),
    metadata={"schema": "my-cases-v1"},
)
with Catalog(receipt.path, expected_revision=receipt.revision) as catalog:
    first_page = catalog.page(limit=20, max_bytes=8 << 20)
    one_case = catalog.get("case-123")
```

Writers consume the iterable once, use an 8 MiB SQLite cache by default, and publish
using a same-directory hard link after committing and syncing the file. The destination
must not exist. Duplicate keys, interrupted iterators, invalid JSON and oversized records
abort publication. Individual JSON records are limited to 1 MiB, keys to 4,096 UTF-8 bytes,
and pages to 4,096 records and 64 MiB of encoded payloads. Decoded Python objects, SQL
engine overhead and the OS cache are additional memory; this is not a process RSS limit.

A reader holds a read transaction for its context lifetime. Open separate readers in
separate threads/processes. Do not modify a published snapshot in place: publish a new
file and distribute its receipt. Reads use `mode=ro`, normal SQLite locking, a validated
schema, `trusted_schema=OFF` and no memory mapping. No extension loading or user-supplied
SQL is exposed. This is a local-filesystem catalog, not a remote range-read index or a
multi-writer distributed database. Use a supported, security-maintained SQLite runtime.

Per-record hashes detect accidental damage. The receipt contains a logical revision and
SHA-256 of the entire file. `expected_revision` checks the header identity; it does not
prove every unread record. Call `catalog.verify()` for a complete logical integrity scan,
or verify the file against a trusted receipt before use. A malicious party able to rewrite
records and hashes can fabricate a new catalog; authenticate receipts outside this library.

The deterministic benchmark creates unique metadata records, compares fresh-process
first-page and last-key lookup against full JSON loading, checks equal results, and records
publication time, file size, latency and peak RSS:

```sh
python -m arrayroute.benchmark_catalog --out /tmp/catalog-million --records 1000000 --repeats 5
```

It measures local metadata access, not numerical throughput or petabyte storage performance.

## Global scalar ordering and quantiles

`arrayroute.numeric_catalog` publishes an immutable index for one scalar column.
Use it to select the highest/lowest values or inspect distribution groups across
the whole dataset without loading every metadata record. Ordinals refer to records
in the source snapshot; its revision is required when opening the index.

```python
from arrayroute.numeric_catalog import NumericIndex, write_numeric_index

receipt = write_numeric_index(
    "inlet-velocity.sqlite",
    (case.get("inlet_velocity") for case in cases_in_source_order),
    source_revision=catalog_revision,
    field="inlet_velocity",
    units="m/s",
)
with NumericIndex(receipt.path, source_revision=catalog_revision,
                  expected_revision=receipt.revision) as index:
    fastest = index.page(descending=True, limit=64)
    global_quartile_edges = index.quantiles(4)
    if fastest.next_cursor is not None:
        following = index.page(descending=True, cursor=fastest.next_cursor, limit=64)
```

Equal values retain source order in both directions. `None` sorts last; quantiles
exclude missing values. Signed zeros compare equally. Accepted values are finite
Python floats, exact integers between `-2**53` and `2**53`, or `None`; booleans,
non-finite values, and larger integers are rejected. Global quantile edges use
inclusive ceil-based ranks, and equal values belong to the same group. Group an
observed value using the first upper edge greater than or equal to it; missing
values belong in a separate group. Do not recalculate edges from a visible page.

Pages return at most 4,096 ordinal/value pairs and use composite-index continuation,
including for descending order and missing values. Cursors belong to a specific
index revision and sort direction. Quantiles support 1–64 bins and are cached per
reader after their first computation. First-time quantiles traverse index positions;
their cost is distinct from the constant-size page lookup.

Publication consumes the values once and uses the same commit, file sync, and
non-overwriting hard-link protocol as the JSON catalog. Readers validate the schema,
pin a read transaction, and check returned row hashes. `verify()` performs a full
logical integrity scan; the receipt also includes the entire file's SHA-256. Header
revision checks alone do not validate unread data. Hashes detect accidental damage,
not malicious rewriting: authenticate receipts separately. Publish new files instead
of changing an existing index, and use one reader per thread/process.

The default SQLite cache is 8 MiB; interpreter overhead, decoded values and the OS
cache are additional memory. This is a local scalar-index component, not a distributed
catalog or a complete sample browser. Build indexes only for columns that need these
queries, since each index has its own storage and publication cost.

```sh
python -m arrayroute.benchmark_numeric_catalog --out /tmp/numeric-million --records 1000000 --repeats 5
```

This benchmark includes missing values, repeated values, deep cursors, full integrity
verification and global quantiles. It retains fresh-process measurements, checks each
result against an independent reference order, and does not claim numerical-array,
network or browser throughput.

## Shared, revision-specific tile caching

Install `arrayroute[disk-cache,tensorstore]` to reuse decoded numeric tiles across
workers and process restarts. This is optional: a native in-memory cache is generally
cheaper when the working set fits and workers stay alive. Persistent cache population
adds serialization and disk writes; use it for repeated remote reads, not a one-pass scan.

```python
from arrayroute.disk_cache import ArrayIdentity, TileCache
from arrayroute.remote import HttpStore, RemoteReadSession

cache = TileCache(
    "/local-ssd/my-user/tiles",
    capacity_bytes=8 << 30,
    max_entry_bytes=32 << 20,
    fill_timeout_seconds=60,
)
session = RemoteReadSession(
    HttpStore("https://my-data.example/immutable-snapshot"),
    tile_cache=cache,
    cache_identity=ArrayIdentity(
        source="simulation-dataset",
        revision="immutable-manifest-revision",
        representation="raw-velocity-metres-per-second-v1",
    ),
)
try:
    velocity = session.read("velocity", (0, slice(0, 64), slice(0, 64), slice(0, 64)))
finally:
    session.clear()
    cache.close()
```

The revision must come from the actual immutable dataset, never a mutable path or
a constant reused after edits. Representation changes must cover decoding, units and
preprocessing semantics. Endpoint/array path, shape, chunk layout, dtype and normalized
selection also enter the cache identity. Any `ArrayReader` backend can be wrapped with
`cache.backend(backend, identity)`; this does not require TensorStore or cloud storage.

Every hit is size-checked and SHA-256 verified against the key and retained content.
The cache reads directly into a writable NumPy output, with no pickle deserialization.
Caller mutations do not affect cached values. Damaged entries are removed and refetched;
origin failures and database I/O errors propagate. Hashes detect accidental corruption,
not malicious changes to both data and checksum. Cache identity is a caller assertion,
not source authentication or a snapshot creator.

DiskCache supplies the cross-process index and LRU eviction. Use a dedicated **local
filesystem** directory shared only by authorized workers of the same user/tenant;
do not put this SQLite-backed cache on NFS. It is not a distributed cache or an access
control boundary. FileLock hard OS locks coordinate concurrent misses: a waiting worker
rechecks the cache after the active fill, reusing its result while the entry remains
retained. Eviction can require another fill. Ordinary cache hits bypass the fill lock.
The 256 persistent lock stripes bound bookkeeping; unrelated keys on the same stripe
can wait for each other. A crashed worker releases its OS lock. Waiting longer than
`fill_timeout_seconds` raises `filelock.Timeout` rather than issuing another source read.
Only local filesystems with OS hard-lock support are supported; soft-lock fallback is
disabled. All workers must agree on cache size, admission policy and schema. Schema 3
records admission behavior; choose a new cache directory when upgrading from schemas
1 or 2. Serialized cache/session objects reopen in spawned workers.

Capacity is DiskCache's approximate database-plus-payload volume after eviction.
WAL, filesystem bookkeeping, concurrent writes and temporary files are additional;
use a filesystem quota when a strict disk bound is needed. Entries larger than
`max_entry_bytes` bypass caching. Cache hits allocate one output array. Cold fills hash
and stream contiguous source arrays without serialized payload copies; a non-contiguous
source requires one contiguous copy. Native/backend scratch space is additional. Backend
`stats` distinguish hits, misses, bypasses, corruptions, hit bytes and source bytes;
these are tile counts/bytes, not compressed network-transfer measurements. Opening an
array can still fetch its metadata even when all requested tiles are cached.
`coalesced_hits` counts hits found after entering the fill lock, and
`fill_wait_seconds` measures successful fill-lock acquisition time. Keep one backend
per worker; do not wrap a cached backend in another cache.

### Retaining tiles across epochs larger than the cache

`TileCache(..., admission="retain")` keeps successful fills instead of evicting them
to admit later tiles. With a fixed repeated scan larger than the cache, this avoids
rewriting the cache on every epoch. The default `admission="lru"` remains useful when
the active subset changes. Compare against uncached reads too: local persistence can
cost more than another read from a fast source.

Retain admission checks the estimated database-plus-payload volume under a cross-process
database transaction, after the upstream read finishes. It reserves 64 KiB of metadata
headroom and rejects a new entry if database growth still exceeds the allowance.
It never removes another retained entry to make room. A known-full cache declines
admission before serializing or hashing the rejected tile. Healthy hits do not update
LRU metadata. Quotas still exclude WAL, temporary files and filesystem overhead.

The first successful fills win; concurrent scheduling can change which subset is
retained. Returned values and ordering are unaffected. Changing revisions creates
different keys but does not reclaim old retained data: use a new directory for a new
working set and retire the old cache when its users have finished.

Backend statistics include `writes`, `write_bytes` and `admission_rejections`.
Write bytes count serialized tile headers and payloads, including a new entry removed
by the post-write volume check. They exclude database/WAL and filesystem writes and
are not a device-write counter. Full-cache reads continue to count as source misses.

## Reuse workers across passes

`OrderedPool` keeps each worker callable on its owning thread between passes. This
lets a preprocessing or training adapter retain immutable array handles and bounded
native caches without constructing them again for every epoch:

```python
from arrayroute.execution import OrderedPool

with OrderedPool(make_processor, workers=4, max_pending=4) as pool:
    for selections in epoch_selections:
        with pool.map(selections) as results:
            for result in results:
                consume(result)
```

`make_processor()` is called once on each active thread and returns a callable that
processes one selection. The factory must create independent reader state, and each
request must carry any epoch-specific selection or transform parameters. A pool permits
only one map context at a time, even before consumption begins. Exiting that context
cancels queued work and waits for already-running calls before another pass can begin.
Task errors surface at their input position without retrying or dropping samples.

The pool retains at most `max_pending` tasks/results, including the result currently
yielded to the caller. Native caches, decoder scratch and caller-retained results are
additional; this is not a process-memory cap. The input iterable remains caller-owned.
Use `pool.map(..., cancelled=...)` for cooperative cancellation between calls and close
the pool (preferably with `with`) to join all worker threads. A blocking native call
cannot be forcibly cancelled by this API; hard deadlines require process isolation.

The existing `ordered_map` convenience function uses a fresh pool for one pass and
releases it on exit. Neither API depends on a machine-learning framework or cloud.

For uneven work sizes, also constrain the planned bytes held by submitted tasks,
completed results and the currently yielded result:

```python
with OrderedPool(make_processor, workers=4, max_pending=8,
                 max_pending_bytes=128 << 20,
                 estimate_bytes=lambda request: request.peak_buffer_bytes) as pool:
    with pool.map(requests) as results:
        for result in results:
            consume(result)
            del result
```

The estimator runs once per input before submission. It must return a nonnegative
integer and account for every buffer covered by the claimed allowance. A task larger
than the allowance fails before execution; reduce its selection or batch size, or
increase the allowance. Tasks wait in input order instead of bypassing a large task.
Admission remains charged while a result is yielded and releases when iteration resumes.
Estimation and input-iteration errors abort the pass immediately; worker failures still
surface at their input position.

One input awaiting admission may be retained as lookahead. The input iterable,
worker caches, unaccounted scratch, allocator overhead and consumer-retained values
are outside this planned-byte allowance. Use lightweight selection descriptors as
inputs when bounding actual data-buffer lifetime. This is not a process RSS limit.

`python -m arrayroute.benchmark_admission --output admission.json` compares count-only
and byte admission in fresh processes with uneven synthetic buffers, verified output
hashes, traced allocations and peak RSS. It measures queue memory, not scientific
I/O or training throughput.

## Measure reader concurrency on your own data

`arrayroute.probe.probe_reads` runs bounded pilots through any `ArrayReader` factory:

```python
from arrayroute.probe import probe_reads
from arrayroute.session import LocalReadSession

def make_reader(budget):
    session = LocalReadSession("tensorstore", buffer_bytes=budget, tensorstore_threads=1)
    return session.open("simulation.zarr", "velocity")

result = probe_reads(
    make_reader,
    [(t, slice(0, 64), slice(0, 64), slice(0, 64), slice(None)) for t in range(8)],
    worker_counts=[1, 2, 4], total_buffer_bytes=256 << 20,
    max_output_bytes=1024 << 20, time_budget_seconds=60,
)
```

The factory must create independent, read-only worker state over unchanged values and
honor its assigned buffer size. A sequential reference pass precedes shuffled repetitions;
all results must match its hashes. Read caps include every pass and are checked before
payload I/O. Timed trials include opening readers, reads, SHA-256 and worker teardown.
Repeated regions may be warm in OS/remote caches: the result ranks only this pilot, not
raw storage bandwidth or training speed. No machine settings change automatically.

The deadline is cooperative between reads. Active native calls are joined; use process
isolation for a hard deadline. Partial pilots never recommend a worker count. Native
caches, codec scratch and up to 64 KiB hash scratch per active worker are additional to
the read-buffer allowance. Source bytes are never written. The reference checks read
consistency; it does not establish scientific ground truth or a storage snapshot.

## Bound range batches for a slow consumer

`arrayroute.ranges` plans known byte ranges and streams completion-order views from an
async callback. It depends on no storage provider. Each view carries its original input
index; input offsets are half-open. Inputs must remain unchanged during a read.

```python
from arrayroute.ranges import RangePolicy, read_ranges

policy = RangePolicy(
    max_request_bytes=1 << 20, max_inflight_bytes=4 << 20,
    max_gap_bytes=4096, max_extra_bytes=65536, concurrency=4,
)
# fetch(start, stop) asynchronously returns exactly those bytes from one object.
async for batch in read_ranges(fetch, [(0, 65536), (69632, 131072)], policy=policy):
    for input_index, view in batch.parts:
        consume(input_index, view)  # Must not retain the view for the stated live-byte bound.
    del view, batch
```

Scheduled tasks, completed results and the current yielded batch share the byte budget.
The consumer must release views before advancing. Transport copies, retained decoder
inputs, metadata and output arrays remain additional; this is not a process RAM cap.
On failure or explicit `aclose()`, pending tasks are cancelled and awaited. Use
`contextlib.aclosing` if a consumer may stop early. The callback must handle its own
transport cleanup; blocking native I/O cannot be forcibly interrupted.

The greedy planner bounds merged request size, bridged gaps, total extra fetched bytes
(including repeated overlap), and range count. It rejects oversized ranges before I/O;
split those explicitly. It may reject overlapping inputs where a different partition
could fit. It does not claim globally optimal packing.

For Zarr-Python 3.3+ partial-shard reads, wrap a store with
`arrayroute.zarr_ranges.BoundedRangeStore(store, policy=policy)`, or pass `range_policy`
to `LocalReadSession("zarr-python", range_policy=policy)`. Upstream per-array limits
can further tighten the policy. No process-global Zarr setting changes. Ordinary `get`
calls, including metadata and full shards, are unchanged. Each range batch has a
separate allowance, so concurrent shards multiply it. Synchronous range calls return
all results together and reject a plan whose total fetched bytes exceeds the allowance.
Unknown-length range batches are rejected. Older Zarr versions fail explicitly when
this optional adapter is constructed; the provider-neutral scheduler still works.

Zarr already coalesces reads by default. This adapter adds backpressure and a total
extra-byte budget; it is not universally faster. In a local 128-request control,
both implementations returned identical 128 MiB; live scheduler payload decreased
from 128 MiB to 4 MiB, with roughly 3–8% more time for the delayed consumer in two passes. A real
8 MiB sharded-array control had similar read time with matched coalescing. Forbidding
extra bytes reduced fetched bytes but increased requests and wall time. Measure your
own access patterns before selecting a stricter policy.

## Share a decode/output budget across readers

`ReadBudget` provides process-local admission control for independent reader threads.
Use one pool for analysis, visualization or preprocessing tasks that run concurrently:

```python
from contextlib import closing
from arrayroute.budget import ReadBudget
from arrayroute.session import LocalReadSession

budget = ReadBudget(256 << 20)

def inspect_field(path, field, selection):
    # Each worker owns its session; only the admission budget is shared.
    session = LocalReadSession("zarr-python", buffer_bytes=64 << 20, read_budget=budget)
    with closing(session.open(path, field).stream(selection)) as source:
        for tile, values in source:
            consume(tile.destination, values)
            del values

# Submit inspect_field calls to your own thread pool.
print(budget.snapshot())
```

Streaming reservations cover the planned output and decoded chunks until the consumer
advances or closes the generator. Materialized `read()` reserves before allocating the
output and releases on return; arrays subsequently held by the caller are additional.
`prefetch()` reserves both slots for its entire context, because a worker can advance
while its consumer still holds the earlier tile. Larger reservations wait FIFO and
cannot be repeatedly bypassed by smaller reads. Cancellation while waiting is supported;
already-running native I/O still follows the backend's cancellation behavior.

This is a conservative **planned-buffer** budget. It does not include compressed inputs,
transport copies, native caches, codec scratch, metadata or caller-retained arrays, and
does not enforce process RSS. The planner counts all chunks touched unless a backend
declares a verified decode window for that read path. Keep enough room for a decoded chunk
and measure the
memory/throughput curve for your data before choosing a limit.

Construct the pool after spawning each process; it cannot be pickled or reused after
fork. Reservations cannot be nested in one thread. Close the current stream before
starting another on that thread, or materialize a result and treat it as caller-owned
memory. Source data must remain unchanged during the operation. As with ordinary
sessions, mutable readers and statistics must not be shared between worker threads.
For TensorStore, `TensorStoreReadGroup.session(read_budget=budget)` composes admission
with shared native cache/thread controls. The cache soft limit is still separate from
the planned output/decoded allowance.

## Dense previews without changing stored data

`LocalReadSession(read_strategy="dense")` opts into a bounded native bounding-box
read followed by an exact in-memory gather. It applies to local Zarr-Python,
TensorStore and Zarrista sessions, including `TensorStoreReadGroup.session`.
The default remains `"native"`: measure both on your own regions before choosing.

```python
from arrayroute.session import LocalReadSession

session = LocalReadSession("zarrista", buffer_bytes=512 << 20, read_strategy="dense")
reader = session.open("velocity.zarr", "")
selection = (0, slice(None, None, 8), slice(None, None, 8),
             slice(None, None, 8), slice(None))
request = reader.plan(selection)
print(request.strategy, request.intermediate_bytes_upper_bound,
      request.buffer_bytes_upper_bound)
preview = reader.read(selection)
```

A dense gather is selected only if its bounding box touches exactly the same
chunks as the strided request and its output, bounding-array intermediate and
native peak decode estimate fit the effective reader buffer. Sparse requests
that skip chunks keep native selection. Contiguous and empty requests also keep
native selection. This is a memory and chunk-work rule, not a timing prediction;
extra copying can still make it slower on a particular backend or layout.
The gather copies into a compact C-order output so a small preview cannot retain
the entire bounding array. Values, including NaN payload bits, are unchanged.

A shared `ReadBudget` caps the effective per-reader buffer at the pool's capacity.
The planner re-evaluates each stream tile, each half-buffer prefetch slot, and the
remaining workspace after a materialized output allocation. Intermediates are
included in shared reservations; native caches, transport and codec scratch remain
outside the allowance. `reader.buffer_bytes` reports that effective cap.

For custom adapters, `DenseGatherBackend(backend)` supplies this policy. Its
`ArraySpec.dense_gather` capability requires `read_plan(ReadPlan)` execution.
`ArrayReader` passes the chosen plan to that method; ordinary unplanned `read`
calls retain the wrapped backend's original behavior. Wrappers must forward the
plan on cache misses (`CachedBackend` does). Pure `plan(spec, selection)` has no
buffer authority and reports native cost; pass `buffer_bytes=` to select a dense
plan. `plan_concurrency` reports each candidate's strategy (or `per-tile` when
stream subdivision is needed). Snapshot and remote session constructors do not
currently expose this opt-in.

## Parallel gathering with bounded intermediates

For independently decodable Zarr shards, a local Zarrista session can opt into
parallel region reads without allocating a dense bounding array:

```python
session = LocalReadSession(
    "zarrista", read_strategy="parallel", gather_workers=8, buffer_bytes=8 << 20,
)
reader = session.open("velocity.zarr", "")
request = reader.plan(selection)
print(request.strategy, request.gather_workers, request.intermediate_bytes_upper_bound)
preview = reader.read(selection)
```

The planner considers up to `gather_workers` workers (1–256), bounded further by the
number of selected shards and the available buffer. It charges the final output,
worker-owned selected regions, and one decoded chunk per active worker. It lowers
the worker count before choosing ordinary native selection when fewer than two
workers fit. Contiguous requests and requests within one region use native selection.
Streaming, materialization and prefetch re-plan with their actual remaining buffers.
This is an explicit policy; small requests can get slower from scheduling overhead.

Selected shards are generated lazily. Each worker gathers only touched subchunks
inside its shard; results are copied into disjoint locations in the final compact
array. The bounded executor retains at most one region per scheduled slot, including
the region being consumed. Every call joins its pool, including on failure, so no
background worker outlives its read reservation. It cannot interrupt an already
running native call. Native codec threads, shard indexes, caches and scratch remain
outside the explicit array-buffer allowance. Outer read workers multiply inner
worker pools; benchmark the whole concurrency configuration rather than assuming
more threads improve throughput.

`ParallelGatherBackend(base, factory, GatherWindow(region_shape, workers))` is the
generic extension point. The factory must supply independent mutable reader state
with the same metadata as `base`. A read-only native array handle may be shared.
Worker results must not retain arrays larger than their selected region or one
decoded chunk; workers must not reserve from the parent reader's shared `ReadBudget`.
The base must guarantee a one-chunk decode window for strided reads; regions must
align to whole chunks on every axis. This capability is not inferred for arbitrary
engines. `LocalReadSession` currently exposes it only for Zarrista partial-shard
reads; other engines and outer-codec shards are rejected for this strategy.

`ReadStats.read_strategies` counts completed tile executions by strategy, and
`gather_workers_peak` reports the largest admitted inner worker limit. This is
not a measurement of active threads; a cache hit may launch none.
`probe_reads` includes these in every trial, so the requested maximum can be
compared with actual budget-reduced counts. Native reads report one gather worker;
that does not count the engine's own internal threads.

### Bulk shard reads

For previews that touch every inner chunk of small shards, try
`read_strategy="bulk"` with the same `gather_workers` setting. It decodes an entire
shard per task through Zarrista's native `retrieve_chunk` operation and gathers the
selected values into a compact array. This can avoid thousands of small native
calls and byte-range requests. It is opt-in: large shards can be slower and require
substantially more memory than partial decoding. Benchmark it against `parallel`
and TensorStore on the intended selections and buffer limits.

The planner admits bulk reads only when they add no skipped in-domain chunks. It
charges the final output, compact pending results, and two full decoded shard
buffers per worker, including edge padding. It lowers the worker count as needed;
if no bulk worker fits or the selection skips chunks, it plans partial parallel
gathering or a native read using the existing buffer rules. Contiguous reads remain
native. `bulk-gather` identifies actual bulk execution in plans and statistics.
The allowance excludes compressed inputs, indexes, codec scratch and other native
state; observed process memory can exceed it. Per-shard codec concurrency is one,
while the outer executor runs the admitted shard tasks concurrently.

Generic adapters can opt in with `GatherWindow(..., bulk=True)` and
`ParallelGatherBackend(..., bulk_factory=...)`. Each factory returns a worker-local
callable accepting one normalized region and returning its compact selected values.
It must preserve dtype/shape, release all decoder state between calls, obey the
two-region decoded-buffer bound and never reserve from the parent reader's budget.
The ordinary factory still supplies the single-chunk partial reader. All workers
are joined before returning or raising; bulk reads retain the same cache and
streaming contracts as partial reads.

### Compare policies on your own regions

`compare_reads` compares reader engines and strategies as well as outer worker counts.
Each policy is a worker-owned reader factory with an assigned share of one total
buffer allowance. It is usable independently of Frontier, and custom backends may
participate without changing the comparison code.

```python
from arrayroute.compare import ReadPolicy, compare_reads
from arrayroute.settings import ReadSettings

settings = [ReadSettings(), ReadSettings("zarrista", "bulk", gather_workers=8),
            ReadSettings("tensorstore", tensorstore_threads=8)]

def factory(configuration):
    return lambda budget: configuration.session(budget).open("simulation.zarr", "velocity")

report = compare_reads(
    factory(settings[0]),
    [ReadPolicy(s.name, factory(s)) for s in settings],
    [(t, slice(None, None, 8), slice(None, None, 8), slice(None, None, 8), slice(None))
     for t in (0, 2, 4, 6)],
    worker_counts=[1, 2], total_buffer_bytes=64 << 20,
    max_output_bytes=1 << 30, time_budget_seconds=60, repeats=3,
)
if report["state"] == "completed":
    print(report["recommendation"])
```

Keep the source fixed during the pilot. One sequential reference establishes SHA-256
values, then all policy/worker/repetition combinations are shuffled together. Every
result must match the reference byte for byte. Shape and dtype must match across
policies; chunk layouts may differ. A missing optional dependency or an explicit
`UnsupportedReadPolicyError` is recorded as skipped. Other failures abort the pilot.

The output cap includes reference and repeated selected outputs. It does not limit
compressed bytes fetched or chunks decoded. Memory admission covers read buffers,
not process RSS, native caches or codec scratch. TensorStore thread settings apply
per native pool per outer reader; other engines retain process-level thread settings.

The recommendation minimizes observed median setup/read/hash/teardown wall time.
Reports retain CPU time, ranges across repetitions, actual execution strategies and
admitted gather concurrency. They are local measurements, not cold-storage or training
throughput predictions. Test the selected configuration on held-out regions before
using it for a large job. The cooperative deadline joins in-flight reads; incomplete
pilots retain diagnostics and never recommend a configuration.

### Choose a cache policy from your own workload

`TraceRecorder` observes exact array selections without saving field values. Wrap
the **outermost** backend so existing cache hits are included. Choose a short,
representative serial pilot with repeated accesses and a later validation segment:

```python
from arrayroute import ArrayReader
from arrayroute.cache_policy import TraceRecorder, compare_cache_policies
from arrayroute.session import LocalReadSession

source = LocalReadSession("tensorstore").open("simulation.zarr", "velocity")
capture = TraceRecorder(max_events=10_000)
reader = ArrayReader(capture.backend(
    source.backend, identity="dataset-revision-42:velocity:physical-m/s-v1",
))
for epoch in range(4):
    for frame in range(16):
        velocity = reader.read((frame, slice(None), slice(None), slice(None), slice(None)))
        # Use velocity in the same preparation/training/inspection loop being evaluated.

trace = capture.snapshot()
trace.save("velocity-access.json")
report = compare_cache_policies(
    trace, capacities_bytes=(120 << 20, 240 << 20),
    max_entry_bytes=32 << 20, training_reads=32, entry_overhead_bytes=40,
)
for row in report["rows"]:
    print(row["capacity_bytes"], row["recommendation"], row["recommendation_status"])
```

The source identity must include its immutable revision and representation, including
unit conversion and preprocessing. Do not reuse one identity for changing content.
Source metadata and selections are hashed; values, URLs and credentials are not
written into the trace. Hashes and access patterns remain sensitive metadata.

Compare saved traces without rereading the dataset:

```sh
python -m arrayroute.cache_policy velocity-access.json \
  --capacity-bytes 125829120 --capacity-bytes 251658240 \
  --max-entry-bytes 33554432 --training-reads 32 --entry-overhead-bytes 40 \
  --output cache-policy.json
```

For each capacity, the comparison selects `none`, `lru` or `retain` using only the
training prefix. It minimizes decoded source bytes, then payload write bytes.
Cache state continues into validation. A training winner that regresses against
LRU on the held-out suffix receives no recommendation. Ties prefer no cache, then
LRU. Reports retain every policy's training and validation counters, not only the
winner. A first-pass-only training prefix usually cannot establish cache benefit.

Capacities are **usable payload plus declared per-entry overhead**, not the disk
quota passed to `TileCache`. Reserve database, WAL and filesystem space separately.
The default entry overhead is zero; 40 bytes models the current tile header only.
Replay does not model compressed transfer sizes, source latency, eviction batching,
filesystem contention or decoder caches. Verify the candidate with actual reads
before applying it. The CLI does not change a cache or install settings.

Recording is optional, process-local and bounded (100,000 events by default,
1,000,000 maximum). Failed or truncated captures cannot be analyzed as complete
pilots. Concurrent callers are captured in invocation order but receive no policy
recommendation: serial replay cannot predict publication or single-flight behavior.
Take a snapshot after outstanding reads finish. Saved JSON uses private permissions,
bounded input size and atomic no-overwrite publication; it contains no executable
factories. The recorder and analysis require only ArrayRoute's base dependencies.

## Ordered frame batches

For minibatches containing overlapping time windows, `read_rows` preserves the
requested order and duplicates while avoiding repeated logical row reads:

```python
from arrayroute import read_rows

# Array axes: time, variable, pressure, latitude, longitude.
frames = read_rows(reader, [12, 13, 14, 13, 14, 15],
                   tail=(slice(None), slice(None), slice(None), slice(None)))
```

Adjacent requested rows can share a read, but unrequested gaps are not added to the
logical selection. Backend chunks may still decode unselected data. Set
`coalesce=False` to keep deduplication without merging adjacent rows. Benchmark both
against the actual layout: coalescing is not always faster. This operation changes
I/O order, not the sampler's output order. It has no persistent data cache.

The reader allowance covers the complete returned batch plus planned read buffers.
A batch that leaves insufficient read space fails; split it into smaller batches.
Native caches, encoded bytes, codec scratch and Python index metadata remain separate
costs. Sources must be immutable throughout the call. `ReadStats` counts physical
reads rather than duplicate returned rows. Shared `ReadBudget` admission covers the
whole operation, and cancellation prevents subsequent reads but cannot interrupt
an in-flight native call.


`LocalReadSession.read_rows(store, array_path, indices, ...)` reuses the session’s
selected engine, open handles and buffer policy for the same operation.

For a batch containing several windows, screen the overlap before deciding to
read and normalize them together:

```python
from arrayroute.batching import plan_row_batch

plan = plan_row_batch([[12, 13, 14], [13, 14, 15]], row_count=100,
                      min_reuse_fraction=0.25)
assert plan.combine
assert plan.unique_rows == (12, 13, 14, 15)
```

The plan uses only requested indices. It normalizes negative indices, retains
first-occurrence order and stores metadata proportional to the batch, not the
array size. `combine` requires multiple selections, repeated rows and a reuse
fraction at least as large as the threshold. A zero threshold still requires
reuse; one keeps finite batches separate. The caller executes the chosen path
and preserves its output order and ownership guarantees.

This is a deterministic overlap heuristic, not a calibrated latency model.
The default 25% threshold is an initial setting to benchmark on the actual layout
and hardware. It does not account for chunk reuse, decoding, copying or cache
effects, and it need not choose the fastest path. Persist the threshold alongside
the data revision and execution settings; do not change it from live timings when
reproducing a run.

# ArrayRoute

Selection planning, bounded reads and streaming statistics for scientific arrays.
Apache-2.0, with NumPy as the only core runtime dependency. Use it independently of
Frontier, on local machines or cloud workers.

- Plan exact integer and strided-slice selections without enumerating an array's chunk grid.
- Stream tiles within a decoded-data/output budget, or materialize a selected region.
- Read through optional Zarr-Python, TensorStore and Zarrista backends.
- Merge masked channel statistics and index long time-window datasets without a per-window list.
- Shuffle epoch positions with a reproducible constant-memory permutation, including worker sharding.
- Reproduce storage experiments with deterministic fixtures, output hashes and raw timings.
- Export large selections into ordinary NumPy files with bounded buffers and no partial publication.
- Compare CPU and optional GPU lossless codecs, including transfer costs and byte verification.
- Check reconstruction tolerances and merge error statistics across bounded, disjoint blocks.
- Read immutable remote Zarr arrays over HTTP or S3-compatible storage with explicit resource limits.

## Install from source

```sh
git clone https://github.com/DimensionLab/arrayroute.git
cd arrayroute
python -m pip install './packages/arrayroute[benchmark,test]'
python -m pytest packages/arrayroute/tests
```

```python
import zarr
from arrayroute import ArrayReader
from arrayroute.backends import ZarrBackend

reader = ArrayReader(ZarrBackend(zarr.open_array("velocity.zarr", mode="r")))
selection = (10, slice(100, 180), slice(0, 100), slice(0, 100), slice(None))
cost = reader.plan(selection)
velocity = reader.read(selection)
```

The API is experimental. Read the [package documentation](packages/arrayroute/README.md)
for supported selections, optional engines, buffering limits and benchmark commands.
Backend caches, codec scratch space and caller-retained buffers are outside the read
budget. Decoded-byte estimates are not measured transport bytes. Performance claims
require a workload, storage system, cache policy and matched concurrency settings.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md). Reproducible correctness cases and measurements
on different storage systems are welcome. Include the fixture recipe, backend versions,
thread settings and raw results when reporting performance.

Licensed under [Apache-2.0](LICENSE).

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

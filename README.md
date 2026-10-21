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

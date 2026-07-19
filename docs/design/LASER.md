# Laser on-disk Quantized Graph (alayalite.laser)

> The Python `Index`/`RawIndex` and `vamana.build_index` entry points in this
> historical implementation guide were removed in AlayaLite 1.1.0. The LASER
> format reader remains available through canonical `Collection` migration;
> `alayalite.laser.selected_simd()` remains a diagnostic only.

Laser is AlayaLite's on-disk Quantized Graph (QG) index for large-scale
ANN search. It is a port of the `symqglib` reference implementation that
backs the paper *"Efficient Index Layout and Search Strategy for Large-scale
High-dimensional Vector Similarity Search"*. The in-tree code lives under
`include/index/graph/laser/` and `python/src/alayalite/laser/`.

Treat Laser as a vertical port: the SIMD-friendly disk layout, FastScan
4-bit quantization, PCA-reduced coordinates, and asynchronous disk search are
coupled. Changing one part can invalidate the paper-alignment assumptions.

## Platform

Laser is supported on:

- Linux x86_64, using the `libaio` backend by default.
- macOS (arm64 and x86_64), using the portable thread-pool backend by default.

Linux aarch64 and Windows are not supported: their wheels ship without LASER,
and a canonical qg seal fails explicitly instead of silently falling back to
Flat. Linux aarch64 enablement is deferred until after the current paper work.
Build support is gated by `ALAYA_ENABLE_LASER` (ON by default on Linux x86_64
and macOS; OFF elsewhere).

```bash
# Debian / Ubuntu
sudo apt-get install libaio-dev

# Fedora / RHEL
sudo dnf install libaio-devel

# macOS (Homebrew)
brew install libomp

```

If `libaio` is missing while Laser is enabled on Linux x86_64, CMake fails
with a message that names the dependency and suggests either
`-DALAYA_ENABLE_LASER=OFF` or `-DALAYA_LASER_USE_THREADPOOL=ON`.

## Input

The public build path consumes base vectors in DiskANN `.fbin` format:
`<int32 N><int32 dim>` followed by `N * dim` row-major `float32` values.
`alayalite.laser.Index.fit(...)` writes all intermediate and final
artifacts under the selected output prefix:

- `<prefix>_pca_base.fbin`
- `<prefix>_pca.bin` when PCA is needed
- `<prefix>_medoids` and `<prefix>_medoids_indices`
- `<prefix>_vamana_graph.index`
- `<prefix>_R<R>_MD<main_dim>.index`

## Build

```bash
cd AlayaLite
uv sync --group laser
```

The `laser` dependency group installs the Python-side runtime needed for
building/searching Laser indexes (`scikit-learn`, `faiss-cpu`, `psutil`,
and supporting packages).

Wheel consumers (people installing the published `alayalite` wheel rather
than working in this repo) should use the PEP 621 `[laser]` extra instead,
since dependency-groups (PEP 735) are uv-only:

```bash
pip install 'alayalite[laser]'
# or, with uv outside a project:
uv pip install 'alayalite[laser]'
```

The `[laser]` extra only covers the runtime imports of `alayalite.laser`
(`scikit-learn`, `faiss-cpu`, `tqdm`). Examples / CLI / plotting tooling
(matplotlib, psutil, tomli backport) stay in the dev-only `laser` group.

On macOS, install OpenMP before configuring:

```bash
brew install libomp
```

CMake searches the standard Homebrew prefixes (`/opt/homebrew/opt/libomp` on
Apple Silicon and `/usr/local/opt/libomp` on Intel Macs). If OpenMP is not
found while `ALAYA_ENABLE_LASER=ON`, configuration fails with a message naming
`brew install libomp`.

Backend selection is automatic for normal builds:

```bash
# Linux x86_64 default
cmake -B build/Release -DALAYA_ENABLE_LASER=ON
# prints: LASER I/O backend: libaio (Linux x86_64)

# macOS default
cmake -B build/Release -DALAYA_ENABLE_LASER=ON
# prints: LASER I/O backend: thread pool (macOS)

# Linux fallback without libaio
cmake -B build/Release -DALAYA_ENABLE_LASER=ON -DALAYA_LASER_USE_THREADPOOL=ON
# prints: LASER I/O backend: thread pool (portable fallback)
```

The thread-pool backend uses buffered `pread` and worker threads. Override the
worker count for local experiments with `ALAYA_LASER_IO_THREADS`; production
Linux x86_64 builds should keep the default libaio backend unless the portable
fallback is intentionally being tested.

## SIMD Dispatch

LASER's handwritten SIMD kernels select their ISA at runtime. A single x86_64
wheel carries AVX-512F+BW and AVX2+FMA implementations for FastScan,
approximate-distance conversion, rotation, scalar range scans, and single-vector
L2 norm. On CPUs with both AVX-512F and AVX-512BW, LASER selects the AVX-512
path; otherwise it uses AVX2+FMA. Non-x86 builds use the generic scalar LASER
path; this is intended for macOS development portability, not throughput parity.

The baseline `-mavx2 -mfma` flags are still used for Eigen and other
non-handwritten code. Function-level target attributes do not change Eigen's
template-selected SIMD path, so PCA matrix-vector work remains tied to the wheel
baseline even on AVX-512 hosts. The expected impact is limited to that Eigen
portion of the search path.

## Performance Notes

- **Huge pages**: The Linux backend hints the kernel with
  `madvise(MADV_HUGEPAGE)` on the LASER read scratch buffers. macOS is a
  no-op by design. Expect a modest hit on search throughput when the working
  set exceeds the small-page TLB (single-digit percent on 100M-vector indexes;
  not measurable on 100K-vector smoke runs).

- **Sector size**: Both supported backends use a 4096-byte alignment
  contract (raised from the previous 512 on Linux). 4K matches the LASER
  on-disk page layout and is superset-acceptable for both 512n and 4Kn
  drives. Consumer code does not need to special-case any backend.

- **`num_threads` only controls query parallelism, not I/O fan-out**
  (thread-pool backend — macOS default, portable Linux fallback). The
  `Index.set_params(num_threads=N)` knob bounds the OpenMP loop that
  dispatches queries (`#pragma omp parallel for` over the query batch).
  Each query's per-page disk reads then fan out to a backend-owned I/O
  worker pool sized `min(MAX_IO_DEPTH=128, 2 * hardware_concurrency)`,
  which is **independent** from `num_threads`. So on an 8-core mac,
  `set_params(num_threads=1)` still uses up to 16 I/O worker threads
  for one query's beam search — the measured QPS is higher than a
  "truly single-thread" baseline. Linux libaio does not have this
  asymmetry: a single query thread there gets a single io_context with
  kernel-async inflight requests; no user-space I/O workers.

  Override the I/O pool via the `ALAYA_LASER_IO_THREADS` env var (see
  the build-flags section above). For a strict single-thread benchmark,
  set `ALAYA_LASER_IO_THREADS=1`; to match libaio's per-thread
  MAX_IO_DEPTH ceiling, set it to `128`.

  *Planned follow-up*: the thread-pool backend will be reworked so that
  `set_params(num_threads=N)` also caps the I/O worker pool to `N`
  (and the env var becomes an explicit override). This will make the
  macOS / portable backend's `num_threads` semantics match libaio, so
  cross-backend QPS comparisons at the same `num_threads` are
  apples-to-apples. Tracking under a separate change in
  `openspec/changes/`.

## Python API

Use the unified wrapper for normal code:

```python
from alayalite.laser import BuildParams, Index

idx = Index.fit(
    "/path/to/base.fbin",
    output_dir="/path/to/build/laser",
    name="dsqg_gist",
    build_params=BuildParams(
        metric="l2",
        main_dim=256,
        R=64,
        L=200,
        alpha=1.2,
        ef_indexing=200,
        ep_num=300,
    ),
    seed=42,
    num_threads=48,
    dram_budget_gb=32.0,
)

idx.set_params(ef_search=200, num_threads=1, beam_width=16)
ids = idx.batch_search(queries, 10)
```

To reopen an existing build:

```python
idx = Index.from_prefix("/path/to/build/laser/dsqg_gist", dram_budget_gb=1.0)
```

The raw pybind class is still exported as `alayalite.laser.RawIndex` for
research code that intentionally bypasses the wrapper.

## CLI

`examples/laser/main.py` is a compact TOML wrapper around `Index.fit`
plus an EF sweep:

```bash
# Build, then run the configured EF sweep.
uv run examples/laser/main.py -c examples/laser/configs/gist.toml all

# Build only.
uv run examples/laser/main.py -c examples/laser/configs/gist.toml build

# Search only against an existing build.
uv run examples/laser/main.py -c examples/laser/configs/gist.toml search \
    --threads 1 --efs 100 200 300
```

The CLI exposes only `build`, `search`, and `all`. The older
`vamana/pca/medoid/index` step names are retired.

### TOML Shape

```toml
seed = 42

[dataset]
name = "gist"
metric = "l2"
degree = 64
main_dimension = 256

[paths]
base = "/path/to/base.fbin"
query = "/path/to/query.fbin"
gt = "/path/to/gt.ibin"
output = "/path/to/build/laser"

[build]
build_threads = 48
ef_indexing = 200

[build_vamana]
L = 200
alpha = 1.2
dram_budget_gb = 32.0

[search]
topk = 10
threads = 1
beam_width = 16
dram_budget = 1.0
ep_num = 300
warmup = 10
runs = 30
efs = [100, 200, 300]
```

`[dataset].degree` is the single source of truth for Vamana `R` and the
Laser degree bound. `examples/laser/main.py` rejects stale fields that
belonged to the old step-by-step reproduction pipeline, including
`[paths].vamana`, per-step seed fields, `dump_rotator`, and
`[build_vamana].seed/num_threads`.

## Reproducibility

`seed` is the master seed used by the unified wrapper for randomized
build sub-steps. Set `build_threads = 1` and run under single-threaded
BLAS/OpenMP settings when you need the most stable local rebuilds.

The old cross-repo Tier A byte-equality harness no longer drives
`examples/laser/main.py` through individual build stages. It is retained
as an existing-artifact comparator at
`scripts/laser_alignment/tier_a_byte_equal.py`.

## Vamana Low-Level API

The standalone Vamana builder remains available for non-Laser workflows:

```python
from alayalite import vamana

vamana.build_index(
    data_path="/path/to/base.fbin",
    output_path="/path/to/graph.index",
    R=64,
    L=200,
    alpha=1.2,
    seed=42,
    num_threads=48,
    dram_budget_gb=32.0,
)
```

Laser's public wrapper builds its own Vamana artifact, so this low-level
entry point is not part of the `examples/laser` CLI contract.

## Known Issues

### Fixed: Low-Dimensional Page Layout

`fix-laser-low-dim-page-layout` fixes the upstream LASER page-layout
mismatch in `include/index/graph/laser/qg/qg_builder.hpp`: the build path
now packs up to `node_per_page_` consecutive node payloads into each
`page_size_` page, matching the `qg.hpp` read formulas
`get_page_offset(id)` and `offset_to_node(id)`.

For `node_per_page_ == 1` configurations, the layout reduces to the
original one-node-per-page output. For `node_per_page_ > 1`
configurations, including SIFT-1M-style `main_dim=64` builds, construction
is no longer refused for the historical write/read mismatch. This fix is
tracked by PR #88.

### Relationship To The In-Memory Graph

Canonical `qg` is now a same-id implementation swap: new Collection artifacts
carry the `qg` algorithm/factory identity and the `qg_laser_segment` reader
feature, while their physical files and service engine are LASER. The old
`qg_segment` reader stays available for compatibility.

The in-memory QG tree under `include/index/graph/qg/` remains builder-only. It
provides the temporary metric-aware topology for inner-product/cosine LASER
builds; its fixed degree limits those paths to `R <= 32`. Memory QG and LASER
still have distinct, non-interchangeable RaBitQ serializations. See
[`rabitq-formats.md`](rabitq-formats.md) for the format contract.

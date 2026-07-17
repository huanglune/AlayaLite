# Changelog

All notable changes to AlayaLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cosine metric support for the in-memory QG engine. QG normalizes rows
  before RaBitQ quantization at build time and wraps the query boundary with
  the same `L2NormalizedQuerySegment` adapter HNSW's cosine path used,
  reusing RaBitQSpace's existing cosine-as-inner-product kernel routing.
  `qg_target_support`/`build_qg_collection_target`/`open_qg_collection_target`
  all accept cosine now (float32 + rabitq quantization, same row-count floor
  as l2/inner_product).
- Durable in-place updates for the LASER on-disk quantized graph (op-WAL, the
  G1 gate). `QGUpdater` gains an opt-in after-image write-ahead log
  (`UpdateParams::enable_wal`, off by default) that rides in the shared
  Physical WAL v1 envelope as the `SEGMENT_OP` record family. A no-steal page
  cache logs a whole-page after-image before installing it, `publish()`
  group-commits a batch before it becomes visible, and `checkpoint()` commits an
  A/B superblock flip; reopen runs a dedicated recovery path with a durable
  segment lineage id and a fail-closed poisoned-writer contract. The two-layer
  crash matrix (SIGKILL kill points + a power-loss persistence model) is green.
  A new `MutableLaserSegment` handle exposes this as a single-writer (flock)
  mutable segment. The G1 minimal scope excludes PID reuse/reclaim, consolidate,
  and gardening under the WAL; those transaction formats are a later wave.
- A shared bottom-layer `wal/frame.hpp` module owns the Physical WAL v1 framing
  (WAL7 envelope, CRC, scan, `WalFile`, byte `Decoder`) for both the collection
  logical WAL and the segment op-WAL — one frame format, no divergence.

### Changed

- **Breaking:** the default sealed target algorithm flips from `hnsw` to
  `qg`. `CollectionOptions::target_algorithm`'s C++ struct default is now
  `core::algorithm::qg` (every existing call site already set it explicitly,
  so this only changes bare-default construction). Python's
  `IndexParams.fill_none_values()` now dispatches on dtype when `index_type`
  is left unset: float32 with no competing quantization request defaults to
  `index_type="qg", quantization_type="rabitq"`; float32 with an explicit
  non-rabitq `quantization_type` (`sq8`/`sq4`/`none`) and any non-float32
  dtype both honestly default to `index_type="flat"` (exact search) instead
  of an approximate engine, since qg is rabitq-only and flat never
  quantizes. Users who relied on the old dtype-independent `hnsw` default
  for int8/uint8 vectors or for float32 with `sq8`/`sq4` quantization move
  from approximate HNSW search to exact flat search.
- Requesting `target_algorithm=hnsw` (or the legacy `index_type="hnsw"`
  spelling) now fails with `not_supported` ("target algorithm is
  unsupported") instead of the old `hnsw`+`rabitq` cross-check's
  `invalid_argument` ("requires explicit index_type=qg"): hnsw's algorithm
  id is rejected by the capability gate before any cross-check runs, the
  same as the already-retired nsg/fusion/vamana/diskann ids. At the Python
  layer, `index_type="hnsw"` is now rejected before ever reaching the native
  layer at all (`common.py`'s valid `index_type` set dropped `"hnsw"`).

### Removed

- The HNSW in-memory graph engine (`hnsw_segment`) and its build kernel,
  including the hand-rolled `hnswlib.hpp` it was built on. `algorithm::hnsw`
  (id `2`) remains reserved and is rejected by the capability gate, matching
  nsg/fusion/vamana/diskann; `flat` and `qg` are unaffected. QG is now the
  only in-memory graph engine (see the cosine-support entry above for how it
  closes HNSW's one remaining capability gap).
- The `tools/codegen/dispatch.yaml`-driven canonical identity test matrix
  and its generator (`tools/codegen/gen.py`, the `alaya_codegen` CMake
  target, and the generated `python/tests/client/_dispatch_matrix_params.py`).
  The matrix's "index" dimension was the literal constant `HNSW` for every
  row (including the one row that actually built qg), so it was entirely
  hnsw-keyed; retiring hnsw retired the whole matrix, not just some rows.
- The NSG and Fusion memory graph engines (`nsg_segment`/`fusion_segment`),
  their build kernels, and the NN-Descent kernel they built on. The
  `algorithm::nsg`/`algorithm::fusion` registry ids remain reserved and are
  rejected by the capability gate; `flat`, `hnsw`, and `qg` are unaffected.
- The standalone Vamana segment engines, disk (`disk_vamana_segment`) and
  memory (`vamana_mem_segment`), and `DiskIndexType::Vamana`. Vamana's
  build primitives are unaffected: they remain the sole producer of the
  LASER on-disk QG format. `algorithm::vamana` remains reserved and is
  rejected by the capability gate, matching nsg/fusion.
- The DiskANN segment engine (`diskann_segment`/`diskann_mutable_segment`)
  and its beam-search/disk-layout implementation. `algorithm::diskann`
  remains reserved and is rejected by the capability gate. The research
  line lives on in the `feat/diskann-delete-repair` branch.

## [1.1.0] - 2026-07-15

AlayaLite 1.1.0 is a full architecture and engineering rewrite on top of the
1.0 line. It replaces the legacy per-index Python/C++ types with a single
canonical `Collection` facade over immutable, capability-typed segments, and
removes the entire legacy surface in one release. **This is a breaking
release: 1.1.0 does not preserve the pre-1.1 Python API and does not open
pre-1.1 on-disk artifacts.**

### ⚠ BREAKING CHANGES

- The legacy Python/native API was removed (not deprecated). `Index`,
  `DiskCollection`, `alayalite.laser.Index`/`RawIndex`,
  `alayalite.vamana.build_index`, the six `Client` index methods, and the
  `Collection.get_cpp_index()`/`get_index()` escape hatches are gone; importing
  or calling them raises `AlayaLiteLegacyApiWarning` pointing at `Collection`.
  Enter through `alayalite.Collection` and `Client` collection methods.
- Pre-1.1 on-disk artifacts are no longer readable. The pre-rewrite import path
  (the legacy PyIndex importer, the DiskCollection-v1 reader, and the read-only
  RocksDB scalar-checkpoint decoder) and the legacy recovery corpus were
  removed; the canonical `Collection` opens only its own manifest-v2 layout.
  Convert any pre-1.1 data on a 1.0.x release before upgrading.
- RocksDB is no longer a dependency; scalar/document/metadata state now lives in
  the Collection checkpoint.
- `quantization_type="rabitq"` now requires an explicit `index_type="qg"`. The
  legacy spellings that silently mapped `{hnsw,nsg,fusion}+rabitq` to QG were
  removed with the legacy `Index`.
- Python memory indexes declared `nsg` or `fusion` now build the requested
  algorithm; earlier releases silently built HNSW.

### Added

- A canonical `alaya::Collection` facade over an internal `SegmentedCollection`
  that owns the only logical WAL, mutation coordinator, checkpoint/version map,
  and manifest-v2 control plane, routing across immutable contract-v3 segments
  (HNSW/NSG/Fusion/memory-QG/Vamana-memory in RAM; DiskFlat/DiskVamana/LASER/
  DiskANN on disk).
- Collection-owned filter execution (prefilter/traversal/postfilter by
  selectivity), RAII search leases with strict budget accounting, and a
  successor-first `seal`/`compact`/`gc` state machine with epoch-delayed
  reclamation.
- A default-off, Collection-internal mutable DiskANN Segment bundle with dark
  WAL staging, COMMIT-before-publish ordering, strict tombstone/version
  filtering, idempotent applied-op replay, and manifest-v2 checkpoints. No
  DiskANN mutation API is exposed through the SDK or Python surface.
- An independently gated readonly `DiskAnnSegment` over the native DiskANN file
  family, with typed sync/native-async search, cooperative cancel/deadline at
  drained beam-wave boundaries, and exactly-once lane delivery.
- Immutable C++ `VamanaMemSegment` with typed build/open/search/batch and
  byte-compatible Vamana graph + `.fbin` persistence. NN-Descent is now a
  detail-only KNNG build kernel, not an implied searchable index.
- `ArtifactManifestV2`: a SHA-256 artifact inventory with a five-step
  READY-bound publication transaction.

### Changed

- NSG and Fusion use immutable contract-v3 segments with byte-compatible
  artifact readers, SQ4/SQ8 search spaces, and scalar-enabled variants. Memory
  QG runs through an immutable `QgSegment` reporting the honest `qg_segment/qg`
  runtime identity.

### Removed

- The legacy Python/native API surface and its native registrations
  (`PyIndexInterface`, `Client`, `DiskCollection`, `_CollectionReadView`, and
  the raw LASER/Vamana builder modules).
- The source bridges `core/compat.hpp`, `index/compat.hpp`, and
  `index/disk/disk_collection.hpp`.
- The collection-level `legacy_importer.hpp`, `disk_collection_v1.hpp`,
  `scalar_data.hpp`, the RocksDB storage stack, and the `include/recovery` and
  `include/executor` modules.
- The old Python dispatch factory and its 33 native template instantiations;
  code generation now owns only the canonical identity test matrix.

## [1.0.3] - 2026-07-07

### Added
- `search_pipelined()` query-level coroutine pipelining: pool threads drive
  concurrent query coroutines over a shared io_uring reactor, each suspending
  on beam-wave reads. Throughput follows Little's law instead of threads/latency.
  Requires `update_io=uring`. (#103)
- `pq_distance_batch()` batched PQ distance kernel with gather-then-transposed
  accumulate and software prefetch; eliminates per-neighbor dependent random
  code fetch at 100M+ scale. (#103)
- Coroutine async update I/O via cooperatively-polled io_uring reactor,
  replacing blocking threads during page reads/writes. (#103)
- Unified page pool: searches peek and fill the shard page cache so hot pages
  written by updates are visible without cold reads. (#103)
- `--eval_pipeline` flag and per-query IO/timing breakdown in the update
  benchmark. (#103)

### Changed
- `VisitedBitset::clear()` uses dirty-word tracking (O(words set)) instead of
  full-array memset, removing the per-query DRAM bandwidth wall at large slot
  counts. (#103)
- `DiskPageCache::write()` recycles the LRU victim's map node and buffer
  in-place at steady-state capacity instead of erase+alloc. (#103)
- `make_search_snapshot()` skips tombstone bitmap copy when count is zero. (#103)
- `make format` now routes through pre-commit pinned formatters, preventing
  version-drift formatting mismatches. (#103)

### Fixed
- Portable `alaya::prefetch_l3()` replaces bare `__builtin_prefetch` in
  `pq_distance_batch` for MSVC compatibility. (#103)
- `io_engine.hpp` adds `<io.h>` and `ssize_t` typedef for Windows builds. (#103)
- `uring_reactor_test` gated behind `CMAKE_SYSTEM_NAME STREQUAL "Linux"`. (#103)

## [1.0.2] - 2026-06-30

### Changed
- perf(diskann): reduce search scratch allocations (#101)

### Fixed
- fix(python): preserve API state on failed operations (#102)

## [1.0.1] - 2026-06-20

### Added
- Self-contained DiskANN disk index (`alaya::diskann::DiskANNIndex`): in-memory
  Vamana build, sector-aligned disk layout, optional PQ, BFS node cache, and
  cached beam search (PQ + No-PQ greedy) with async-pipelined I/O and opt-in
  deterministic mode for byte-exact batch/sequential reproducibility.
- IP-DiskANN in-place update with tombstone, slot reuse, and proactive graph
  repair: `insert()` / `remove()` / `flush()` on the No-PQ disk index,
  following the IP-DiskANN design (search + c=3 replacement edges at delete
  time). New components: `TombstoneBitmap`, `SlotAllocator`, `DiskPageIO`,
  `DiskUpdateContext`. Meta v2 with backward-compatible v1 read.

### Changed
- Refactored DiskANN update internals: extracted shared reconnect backbone
  (`cached_l2`, `prune_and_write`) from `update_node_impl` and
  `update_node_ipdiskann`, eliminating ~60 lines of duplication. Removed
  redundant `removed_vertices_` set from `DiskUpdateContext`. Replaced
  `unordered_set` with sorted comparison in `same_neighbor_set`. Reuse
  `AlignedRead` in PQ rerank loop. Use `partial_sort` for IP-DiskANN top-3
  candidate selection.

## [1.0.0] - 2026-05-22

First stable release of AlayaLite. Highlights since the 0.1.x alpha line:

- LASER on-disk Quantized Graph index reaches production status across the
  full wheel matrix (Linux x86_64 / aarch64, macOS x86_64 / arm64, Windows
  x86_64), with three I/O backends (Linux libaio, Windows IOCP, portable
  thread pool).
- `DiskCollection` gains a single-writer lock, atomic ctor / open
  contract, and a `SegmentFactory` dispatch layer; the LASER engine is
  reachable end-to-end through `DiskCollection` on Linux+libaio builds.
- Codegen-driven dispatch (`tools/codegen/dispatch.yaml`) replaces the
  hand-written macro chain; `_alayalitepy` binary drops from ~43 MiB to
  ~26 MiB. **Breaking**: the supported vector dtype matrix is narrowed
  to `{np.float32, np.int8, np.uint8}` at the engine boundary.

### Added
- LASER on-disk index is now buildable and runnable on Windows x64 via the
  new IOCP (I/O Completion Ports) I/O backend
  (`include/index/graph/laser/utils/iocp_file_reader.hpp`). The release
  wheel matrix now ships native LASER on all 5 platform lanes
  (`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macos x86_64`,
  `macos arm64`, `windows-2022 AMD64`); Linux ARM remains the only lane
  with LASER off-by-default (tracked in a separate change). The Windows
  lane uses MSVC 2022 with the toolchain-default `/openmp` runtime
  (`vcomp140.dll`); LASER stays inside OpenMP 2.0 surface, no
  vcpkg/Conan OpenMP package is required. `_alayalitepy.pyd` runtime
  DLLs are bundled into the wheel by `delvewheel`.
- LASER on-disk index is now buildable and runnable on macOS (M-series and
  Intel) using a portable thread-pool I/O backend. Linux x86_64 continues to
  use the libaio backend with no behavior change.

### Changed
- Disk/storage/space layer no longer pulls POSIX headers
  (`<unistd.h>`, `<sys/mman.h>`, `<sys/file.h>`, ...) unconditionally —
  the LAYER 2 portability sweep routed every previously-POSIX file
  operation through `include/utils/platform_fs.hpp` helpers
  (`write_all_fsync`, `read_regular_file_bounded`, `read_file_prefix`,
  `atomic_replace_no_overwrite`, `sync_file_or_throw`,
  `sync_directory_or_throw`, `truncate_file`, `get_pid`,
  `is_readable_regular_file`). The collection lock acquisition in
  `disk_collection.hpp` is now per-platform: POSIX keeps the existing
  `flock` + inode-swap defense, Windows uses
  `LockFileEx(LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY)` plus
  shared-mode flags that exclude DELETE (the kernel-level analogue of
  the POSIX KL3 defense).
- `MMapFile` (`include/storage/mmap_file.hpp`) replaced its
  `throw "unsupported"` Windows stub with a real
  `CreateFileMappingW + MapViewOfFile` implementation.
- LASER sector-size constant raised from 512 to 4096 across all backends
  (matches the LASER on-disk page layout; 4Kn-drive compatible without
  reducing 512n correctness).
- LASER memory.hpp prefetch hint detection no longer relies on the
  `__SSE2__` macro (which MSVC does not define); uses
  `ALAYA_ARCH_X86`-gated `_mm_prefetch` with a `__builtin_prefetch`
  fallback that is itself gated on GCC/Clang availability.

### Fixed
- LASER SIMD rotation and scalar range kernels now use unaligned load/store
  instructions in the runtime-dispatched paths, avoiding potential aligned
  AVX-512 stores into Eigen buffers that are only 32-byte aligned under the
  wheel's AVX2 baseline.
- Disk LASER index now supports `node_per_page_ > 1` (low-dim datasets like
  SIFT-1M); previously refused at construction. Fixes the upstream
  `qg_builder.hpp` write/read page-layout mismatch.
- Disk LASER build no longer zero-clobbers per-node neighbor IDs. The
  `update_qg_out_of_memory` path used to write neighbor IDs into the per-node
  payload **before** the `read()` that pulled the centroid vector into the
  same `page_buf`; for any `node_len_` that fit inside a single sector
  (SIFT-1M `main_dim=64`), the read overwrote the neighbor region with the
  zero-padded tail of the temporary vector file, leaving every node with
  `neighbors[*] = 0` and collapsing recall to random level. Lifts SIFT-1M
  `recall@10` from 0.0009 to 0.9984 at `ef_search=800`. `main_dim=256`
  configurations (gist1m, bigcode, synth_20k_768d) were never affected, since
  `neighbor_offset_ > full_page_size` there.

### Changed
- LASER handwritten SIMD kernels now use runtime dispatch: wheels keep the
  AVX2+FMA x86 baseline while selecting AVX-512F+BW kernels automatically on
  capable hosts.
- `pybind-dispatch-codegen`: replaced the hand-written
  `python/include/dispatch.hpp` macro dispatch chain with a codegen-driven
  pipeline. Single source of truth lives in `tools/codegen/dispatch.yaml`,
  expanded by `tools/codegen/gen.py` into
  `python/include/dispatch_generated.hpp` (out-of-line `IndexFactory::create`
  body) and `python/tests/client/_dispatch_matrix_params.py` (parametrized
  test inputs). `IndexFactory::create` lives in its own translation unit
  (`python/src/index_factory.cpp`) so the 33 template specializations land in
  one `.o`; `_alayalitepy.so` drops from ~43M to ~26M (-40%). New
  `make codegen` target regenerates both files. The two generated files are
  excluded from pre-commit format/lint hooks (clang-format / ruff / pylint /
  cpplint) so `gen.py` has no build-time dependency on clang-format.
  **Breaking**: the supported vector dtype matrix is narrowed to
  `{np.float32, np.int8, np.uint8}` — `np.float64`, `np.int32`, `np.uint32`
  are no longer accepted at the engine boundary (`alayalite.common.valid_dtype`
  raises `ValueError`). The previous wider matrix was declared in macros but
  rarely exercised end-to-end; downstream callers must cast to `np.float32`
  (the recommended default) before `fit` / `insert` / `search`.
- `disk-collection-single-writer-lock`: `DiskCollection` now holds a
  process-level single-writer lock at `<collection>/.lock`; concurrent opens
  of the same collection raise with a stable dual-substring error containing
  the lock path and `collection is already open by another process`. NFS /
  Windows remain unsupported; no reader-writer lock and no WAL are introduced.
- `disk-collection-ctor-create-atomicity`: closes the three residual
  concurrency windows acknowledged by `disk-collection-single-writer-lock`.
  The `DiskCollection` constructor now uses an `O_CREAT|O_EXCL` acquire
  helper (`acquire_collection_lock_for_create`) that serializes concurrent
  ctors at the kernel layer and rolls back its just-created `path/`
  directory on acquire failure. `DiskCollection::open(path)` rejects a
  half-published target (`path/` exists but neither `.lock` nor
  `collection_manifest.txt` are present) with the dual-substring error
  containing the `.lock` path and `target path is a collection-in-progress,
  not yet published`. Both acquire entry points run a post-flock
  `(st_dev, st_ino)` revalidation against `fstat(fd)` so an in-flight
  `unlink(.lock); touch .lock` swap throws `lock file inode mismatch
  after acquire`. EEXIST on the ctor entry surfaces `target path already
  exists or is being created concurrently`. Out-of-band `.lock`
  manipulation that completes before a new acquire begins remains an
  explicit non-goal (no stat-based mechanism can observe it).
- `disk-segment-searcher-dispatch`: refactored `DiskCollection` to dispatch
  segment construction through a new `disk-segment-factory` layer
  (`include/index/disk/segment_factory.hpp`). Five sites that previously
  hard-coded `DiskFlatBuilder` / `DiskFlatSegmentSearcher` now route through
  the factory's `engine_supported_v1` / `create_segment_from_pending` /
  `load_segment_from_manifest` entry points. Cross-segment label
  uniqueness is now engine-agnostic (mmap'd `manifest.ids_file` reads
  instead of `dynamic_cast<DiskFlatSegmentSearcher *>`). Flat behaviour,
  byte format, and Python API are unchanged. Laser remains rejected at the
  v1 capability gate with the same dual-substring error contract (engine
  name + "not implemented in v1").

### Added
- Add unified DiskCollection benchmark harness
  (`python -m alayalite.bench.disk_collection`) covering disk_flat /
  disk_vamana / disk_laser engines with runtime-gated laser support and
  provenance-bearing JSON+MD output.
- expose `disk_laser` via `alayalite.DiskCollection` Python binding (import +
  search; v1 does not support add/flush/batch_search). Lifts the binding-side
  hard veto so `index_type="disk_laser"` is gated by the C++
  `engine_supported_v1(Laser)` predicate (Linux + `ALAYA_ENABLE_LASER=ON` +
  libaio). Adds `DiskCollection.import_laser_segment(src_dir, labels, *,
  copy=True)` driving the C++ importer with binding-side validation
  (src_dir directory check, labels dtype/contig/shape, GIL release). Threads
  `beam_width` (keyword-only, default 4) through `search()`, applied
  uniformly across engines. Adds an engine-uniform NaN / Inf check on
  `query` that uses the bit-pattern `is_finite_f32` helper (per
  `project_ofast_finiteness_check`). Rejects `add()` / `flush()` on
  `disk_laser` collections at the binding boundary with the dual-substring
  contract pointing at `import_laser_segment`. Non-goals: no in-C++ build
  pipeline, no add/flush/batch_search, no PCA/medoid C++ port, no LASER
  format change, no metadata filter, no WAL/compaction/delete/upsert. v1's
  `copy=False` raises `NotImplementedError` (the C++ entry point does not
  accept a per-call params override; the keyword is preserved on the API
  surface for a future change). On unsupported configurations
  (Linux+OFF / macOS / Windows / SIMD-unsupported Linux) the wheel keeps
  building and `disk_laser` raises `ValueError` with the dual-substring
  message; positive-path Python tests are gated on a runtime probe so the
  test matrix stays green across the wheel matrix.
- DiskCollection now supports the LASER engine end-to-end at the C++ level for
  load+search+import (L2 only; Linux + libaio +
  `ALAYA_ENABLE_LASER=ON` only): `index_type=disk_laser` reachable through
  `DiskCollection`, segment importer + searcher under `include/index/disk/`,
  and a SegmentFactory registration. Native LASER files and sidecars are
  co-located inside `seg_<id>/` under their native filenames and recorded in
  `manifest.x_extras`. v1 has no in-C++ build pipeline; segments enter via
  `DiskCollection::import_laser_segment` from precomputed artifacts produced by
  the upstream Python module + `QGBuilder::build`. Python `disk_laser` exposure
  remains deferred to a follow-up.
- DiskCollection now supports the Vamana engine end-to-end (L2 only):
  `index_type="disk_vamana"` at the C++ level, segment builder + searcher
  under `include/index/disk/`, and a SegmentFactory registration. Python
  `disk_vamana` exposure remains deferred to a follow-up.
- Disk-resident segmented collection (`disk-collection` + `disk-flat-builder`
  + `disk-flat-searcher` + `disk-types` + `mmap-file` + `segment-manifest`
  capabilities). New `alayalite.DiskCollection` Python surface (constructor
  + static `open(path)` factory + `add` / `flush` / `search` / `dim` /
  `size`), wrapping `alaya::disk::DiskCollection` in C++. v1 supports the
  Flat (brute-force) segment type with L2, IP, and COS metrics; Vamana and
  Laser segment types are reserved enum values rejected at the v1
  capability gate. On-disk layout: `<collection>/segments/seg_NNNNNNNN/`
  containing `manifest.txt`, `ids.u64.bin`, and `vectors.f32.bin`, plus a
  top-level `collection_manifest.txt`. POSIX-only (Linux + macOS); Windows
  builds compile but throw at runtime. See
  `openspec/changes/archive/2026-04-30-add-disk-collection-flat/` and
  `examples/disk_collection_basic.py`.
- Disk-based vector index (DiskANN) for billion-scale datasets
- Real-time update support for dynamic vector insertion and deletion
- Scalar-vector fusion search for hybrid queries
- Laser on-disk Quantized Graph index (`laser-disk-index` capability).
  New `alayalite.laser.Index` Python surface, CLI pipeline at
  `examples/laser/`, and the `alaya::laser::QuantizedGraph` C++ class
  under `include/index/graph/laser/`. Consumes DiskANN-format Vamana
  `.index` + `.fbin` inputs; produces a FastScan + RabitQ quantized
  on-disk layout served via `libaio` beam search. See `docs/LASER.md`.
- Build-time `ALAYA_ENABLE_LASER` CMake option (default ON on Linux).
  Adds a new system build dependency on `libaio-dev` for Linux builds
  when the option is ON.
- `scripts/laser_alignment/gen_synth_100k_512d.py` — synthetic dataset
  generator used as a secondary alignment judge for the Laser port.
- Sharded Vamana partition-merge alignment with patched upstream
  DiskANN (`diskann-sharded-alignment-gate` capability). Tier A
  asserts byte-equality on `_medoids.bin` (the partition-stage
  invariant) and structural parity on the other artifact classes
  between AlayaLite's `build_vamana_index` CLI and the patched
  DiskANN `build_merged_vamana_standalone` CLI at matched seeds on
  `synth_100k_512d`. Test driver: `tests/vamana/test_sharded_byte_equality.py`.
  Exposed `BuildVamanaParams::sampling_rate` (sentinel auto =
  `min(1.0, 256000/N)`) so the partition growth loop is numerically
  stable on small datasets. Tier B retained as nightly statistical
  envelope; harness now accepts `--expected_num_parts_envelope <lo> <hi>`.
  See `openspec/changes/archive/2026-04-25-align-diskann-sharded-with-upstream/`.

## [0.1.1-alpha1] - 2026-01-28

### Added
- Full cibuildwheel support for multi-platform builds:
  - Linux: x86_64, aarch64
  - macOS: x86_64 (Intel), arm64 (Apple Silicon)
  - Windows: x86_64
- Unified Conan dependency management script (`conan_install.py`)
- pylint integration for Python code quality

### Changed
- Simplified CMake build system
- Updated type annotations for Python 3.8 compatibility
- Configured static linking for all Conan dependencies

## [0.1.0-alpha3] - 2026-01-15

### Added
- RaBitQ (Random Bit Quantization) implementation
- Standalone app with reset functionality
- C++ code coverage support (Codecov)
- Custom index parameters in Collection class
- SetMetricRequest endpoint for collection metric configuration

### Changed
- Refactored ANN-benchmark adaptation
- Modularized CMake configuration
- Improved RAG example error handling

### Fixed
- RAG example embeddings check failure
- Multiple bug fixes and code refactoring
- AlayaLite wheel URL in Dockerfile

## [0.1.0-alpha2] - 2026-01-01

### Added
- Python code coverage (Codecov)
- Typo's pre-commit check
- Initial standalone app implementation

### Changed
- Updated documentation with absolute URLs for online access
- Added icons to documentation

## [0.1.0-alpha1] - 2025-12-15

### Added
- Initial release of AlayaLite
- HNSW (Hierarchical Navigable Small World) index
- NSG (Navigating Spreading-out Graph) index
- Fusion Graph index
- SQ8 (8-bit scalar quantization) support
- L2, Inner Product, and Cosine similarity metrics
- Python SDK with Client, Index, and Collection APIs
- RAG components (embedders, chunkers)
- Basic CI/CD pipeline

[Unreleased]: https://github.com/AlayaDB-AI/AlayaLite/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/AlayaDB-AI/AlayaLite/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/AlayaDB-AI/AlayaLite/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/AlayaDB-AI/AlayaLite/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/AlayaDB-AI/AlayaLite/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.1a1...v1.0.0
[0.1.1-alpha1]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.0a3...v0.1.1a1
[0.1.0-alpha3]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.0a2...v0.1.0a3
[0.1.0-alpha2]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.0a1...v0.1.0a2
[0.1.0-alpha1]: https://github.com/AlayaDB-AI/AlayaLite/releases/tag/v0.1.0a1

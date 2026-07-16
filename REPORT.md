<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Topology-preserving seal production report

## Outcome

This change turns the validated topology-preserving seal path into public,
header-only production APIs. A finalized Vamana or memory-QG topology can now be
captured once and materialized into either a complete read-only DiskANN directory
or the existing LASER row format without rebuilding the graph.

No Collection rotation wiring, PQ optimization, delete-repair work, or default
build-policy change is included.

## Public API

`FrozenGraphSnapshot` lives at
`include/index/graph/frozen_graph_snapshot.hpp` in namespace `alaya`:

```cpp
using Adjacency = std::vector<std::vector<std::uint32_t>>;

FrozenGraphSnapshot(Adjacency adjacency,
                    std::uint32_t entry_point,
                    std::uint32_t max_degree,
                    std::uint64_t frozen_pts = 0);

FrozenGraphSnapshot(vamana::VamanaBuilder &&builder,
                    std::uint32_t max_degree,
                    std::uint64_t frozen_pts = 0);

static FrozenGraphSnapshot from_vamana(vamana::VamanaBuilder &&builder,
                                       std::uint32_t max_degree,
                                       std::uint64_t frozen_pts = 0);
static FrozenGraphSnapshot load(const std::filesystem::path &path);

void validate() const;
void save(const std::filesystem::path &path) const;
```

The snapshot is move-only and exposes only const adjacency access. `N` is derived
from the adjacency size, so it cannot diverge from the owned graph. Validation
checks a non-empty graph, entry-point range, degree bounds, neighbor ranges, and
self-loops. `save()` reuses `vamana::save_graph`; `load()` consumes the same
24-byte native-layout header and infers `N` from the node records. Non-zero
`frozen_pts` is preserved by snapshot save/load.

`VamanaBuilder::release_graph() &&` transfers the outer adjacency vector and all
per-node edge buffers. The builder is left with an empty graph, avoiding a second
copy of a large edge set.

DiskANN adds the topology-independent artifact parameters and entry point:

```cpp
struct diskann::DiskANNMaterializeParams {
  std::uint32_t record_capacity = 0;
  std::uint32_t pq_n_chunks = 0;
  double cache_ratio = 0.05;
  std::uint32_t num_threads = 0;
  std::uint32_t pq_train_iters = 15;
  std::uint64_t seed = 1234;
  bool verbose = false;
};

static void diskann::DiskANNIndex::build_from_graph(
    const std::string &index_dir,
    const FrozenGraphSnapshot &snapshot,
    const float *vectors,
    const std::uint64_t *labels,
    std::uint64_t dim,
    const diskann::DiskANNMaterializeParams &params = {});
```

The function derives `N`, medoid, and graph degree from the snapshot. It writes
`diskann.index`, `ids.bin`, both cache files, optional PQ files, and `meta.bin`.
The existing `DiskANNIndex::build` still constructs Vamana exactly as before and
then calls the same private artifact materializer. Failed builds remove their
partial target directory.

Memory QG and LASER add:

```cpp
FrozenGraphSnapshot QgSegment<SpaceType>::export_graph_snapshot() const;

void laser::QGBuilder::build_from_graph(
    const FrozenGraphSnapshot &snapshot,
    const char *filename);
```

## Placement decisions

The snapshot sits at `include/index/graph/` because it is a format-neutral graph
handoff shared by Vamana, memory QG, DiskANN, and LASER. It depends on the Vamana
single-file format only for persistence interoperability.

DiskANN materialization remains inside `DiskANNIndex`. This allows the default
build and the new entry point to share the existing private metadata and label
serializers as well as the exact layout/PQ/cache code. A separate
`DiskANNMaterializeParams` avoids accepting and silently ignoring topology knobs
such as R, L, and alpha.

QG export is on `QgSegment`, not `QgBuilderKernel`. The kernel's
`new_neighbors_` is transient candidate state, while the Segment-owned
`QgGraph`/RaBitQ edge slots are the finalized topology used by search and
serialization. Segment placement also makes export available after reopening an
artifact. Export scans the 32 fixed slots, removes self-loops and duplicates, and
validates the resulting snapshot. The conversion necessarily allocates one owned
adjacency because the source topology is interleaved with the QG codec.

The LASER wrapper intentionally does not alter the packer. It validates snapshot
shape/degree, writes a unique temporary Vamana file, calls the existing
`QGBuilder::build(vamana_path, prefix)`, and removes the temporary file on success
or exception.

## Differences from the PoC

- The PoC's digest, timing, recall, and benchmark-only CLI code is not part of
  the API. Correctness checks moved into deterministic CTest coverage.
- The formal DiskANN API accepts caller-provided external labels rather than
  forcing identity labels.
- Snapshot ownership transfer, structural validation, malformed-file checks,
  cleanup guards, and target-format compatibility checks are explicit.
- The PoC used a hidden staging directory followed by rename. Production reuses
  the existing `DiskANNIndex::build` directory/cleanup contract so default build
  behavior stays unchanged.
- PQ training and encoding are deliberately unchanged and shared with the
  default build path.

## Tests

Release configuration used the requested command plus `-DBUILD_TESTING=ON`
(testing defaults to off in this worktree). Full compilation completed with
`cmake --build build/Release -j 64`.

New or extended CTest coverage:

- `frozen_graph_snapshot_test`: byte-identical save/load/save round trip,
  metadata preservation, positive and negative validation cases, move-only type
  contract, and Vamana adjacency ownership transfer.
- `test_diskann_build_from_graph`: fixed-seed 10,000 x 64 float32 input, R=32,
  L=64, one build/PQ thread, eight PQ chunks, and three PQ iterations. The
  default build and snapshot materialization produced the same seven filenames;
  every file was byte-identical, including PQ pivots and compressed codes.
- `index_test_qg_segment`: the exported 128-node graph validates, retains the QG
  entry point, and contains exactly 32 unique non-self edges per node (4,096
  edges total).
- `laser_test_page_layout_round_trip`: now enters the unchanged LASER packer via
  `FrozenGraphSnapshot` and still passes all page-layout/payload assertions.

Results:

```text
ctest --test-dir build/Release -L diskann -j 16 --output-on-failure
  15/15 passed

ctest --test-dir build/Release \
  -R '^(frozen_graph_snapshot_test|test_diskann_build_from_graph|index_test_qg_segment|laser_test_page_layout_round_trip)$' \
  --output-on-failure
  4/4 passed

ctest --test-dir build/Release -j 16 --output-on-failure \
  --timeout 600 -LE 'performance|tsan|stress'
  86/87 passed
```

The one broad-suite failure is the pre-existing `laser_segment_test`, which
expects `collection_manifest.txt` after calling the documented no-op v1 writer
gate. The identical failure and missing-file diagnostic reproduces in the clean
`main@5d5bf28` Release build at
`/home/huangliang/workspace/alaya-dev/AlayaLite-rabitq-equiv/build/Release`.
No file touched by this task participates in that manifest path.

No performance benchmark, taskset, or NUMA binding was run.

## Known limitations

- Snapshot persistence supports non-zero `frozen_pts`, but both DiskANN and
  LASER materialization reject it because their current read-only target layouts
  do not preserve frozen-point semantics.
- LASER currently incurs a temporary Vamana-file write; direct in-memory packer
  ingestion is a possible follow-up.
- `build_from_graph` still requires all raw vectors and labels and runs the
  existing PQ train/encode implementation when PQ is enabled.
- QG export is an O(N x 32) conversion and owns its adjacency. It does not copy
  the graph twice.
- The Vamana snapshot file remains native-endian and has no explicit `N`, matching
  the existing DiskANN-compatible format.
- Collection rotate-to-successor integration is intentionally out of scope.

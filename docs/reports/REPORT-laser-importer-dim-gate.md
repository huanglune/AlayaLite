<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# LASER importer dimension-gate final report

## Verdict

The stop-loss condition did not trigger. LASER has no raw-dimension
power-of-two dependency: its own single-round `FHTRotator` pads to the next
power of two, and the RaBitQ builder, row layout, loader, and search path all
consistently use that padded width for quantized data while retaining the raw
width for vectors and exact L2 terms.

The v1 public LASER dimension contract is now `33 <= dim <= 2048`. A dimension
inside that range need not be a power of two; 768 is represented with
`padded_dim=1024`. Dimensions 32 and 2049 still fail admission. This range is
derived from LASER's own FHT table domain, not from memory-QG's mathematically
different `FhtKacRotator`.

## Forensic chain

### 1. Rotator and the actual bound

- LASER selects FHT implementations for transform orders 6 through 11 and
  rejects other orders (`include/index/graph/laser/utils/rotator.hpp:78-107`).
  Its constructor computes `padded_dim = 1 << ceil_log2(dim)` and selects by
  that same ceiling (`include/index/graph/laser/utils/rotator.hpp:128-140`;
  `include/index/graph/laser/utils/tools.hpp:21-29`). The hard condition is
  therefore `ceil_log2(dim) in [6,11]`, whose raw-dimension interval is 33
  through 2048. In particular, 33 through 64 select order 6; 32 selects the
  unsupported order 5. Likewise, 768 selects order 10 and pads to 1024. This
  ceiling-derived lower boundary is a LASER-specific result, not memory-QG's
  floor-log rule.
- Rotation consumes exactly `dim` source elements, zero-fills
  `[dim,padded_dim)`, and runs FHT over the padded destination
  (`include/index/graph/laser/utils/rotator.hpp:150-156`). Padding is thus
  defined data, not an out-of-bounds tolerance.
- LASER owns this concrete single-round rotator. Memory QG defaults to the
  separate four-round/Kac-walk `FhtKacRotator`
  (`include/space/quant/rabitq/rotator.hpp:158-207,235-283`). The format
  contract now calls out the different rotation mathematics explicitly
  (`docs/design/rabitq-formats.md:8-16,24-30`); no memory-QG floor/log rule was
  copied into the LASER decision.

### 2. RaBitQ codebook and query preparation

- `QuantizedGraph` independently derives the same next-power-of-two
  `padded_dim`, constructs both scanner and LASER rotator with it, and sizes the
  node from raw/residual dimensions plus `degree * padded_dim`
  (`include/index/graph/laser/qg/qg.hpp:679-699`). Its executable invariant says
  explicitly that FHT/RaBitQ/FastScan use `padded_dim`, raw and exact terms use
  `dimension_`, and a non-power-of-two main dimension is supported
  (`include/index/graph/laser/qg/qg.hpp:714-719`).
- RaBitQ receives the rotated matrix width, requires that padded width to be a
  multiple of 64, allocates `dim/64` words per code, and passes the same width
  to the packer (`include/index/graph/laser/quantization/rabitq.hpp:63-96`).
  LASER's packer iterates and allocates by `padded_dim`
  (`include/index/graph/laser/quantization/fastscan_impl.hpp:46-76`). Nothing in
  this path reuses raw 768 as a packed-code stride.
- Query preparation allocates/rotates/quantizes a padded vector and builds the
  padded LUT (`include/index/graph/laser/qg/qg_query.hpp:48-82`). The scanner
  retains that padded width for LUT packing, accumulation, and code-block
  stepping (`include/index/graph/laser/qg/qg_scanner.hpp:53-83`).

### 3. Row format and page layout

- The row byte length is
  `(32*main_dim + 32*residual_dim + 128*R + R*padded_dim)/8`
  (`include/index/graph/laser/qg/qg.hpp:697-699`). Offsets agree with that
  formula: raw main/residual floats first, then `padded_dim/64 * 2 * R` float
  slots of packed code, three factor arrays, and neighbor IDs
  (`include/index/graph/laser/qg/qg.hpp:1347-1356`).
- The builder validates the raw fbin width against `main_dim+residual_dim`,
  sector-aligns raw-vector reads, rotates into padded matrices, and calls
  `rabitq_codes` on those matrices
  (`include/index/graph/laser/qg/qg_builder.hpp:250-266` and
  `include/index/graph/laser/qg/qg.hpp:1408-1471`). Page assembly writes each
  complete `node_len_` row at the same slot addressing used by the reader
  (`include/index/graph/laser/qg/qg_builder.hpp:354-393`).
- Page geometry is derived solely from `node_len`, rounded to 4096-byte sectors,
  with one format-v2 trailer reserved per row
  (`include/index/graph/laser/qg/qg.hpp:97-105,143-156`). The loader validates
  dimension, node length, nodes per page, page size, and file size against the
  reconstructed geometry (`include/index/graph/laser/qg/qg.hpp:1498-1555`).
  There is no page-address calculation that assumes the raw dimension is a
  power of two.

For 768d:

- `padded_dim=1024`.
- Persistent padding overhead is `R*(1024-768)/8 = 32R` bytes per row, before
  sector rounding. This is 1024 bytes at the Collection default `R=32`, or
  2048 bytes at `R=64`.
- At `R=32`, the tested Collection row is
  `4*768 + 16*32 + 32*1024/8 = 7680` bytes and occupies one 8192-byte page.
  At `R=64`, the row is 12288 bytes; because the v2 trailer needs non-zero
  slack, page geometry raises it to 16384 bytes. Raw vectors remain 768 floats
  in both cases. These semantics and the formula are documented at the importer
  gate (`include/index/disk/laser_segment_importer.hpp:228-247`) and in the
  format contract (`docs/design/rabitq-formats.md:20-28`).

### 4. Searcher closure

- `LaserSegmentSearcher` requires `x_main_dim` to equal the manifest dimension
  in this v1 importer path, constructs `QuantizedGraph(count,R,main_dim,dim)`,
  and then loads the native index
  (`include/index/disk/laser_segment_searcher.hpp:217-278`). Thus build and
  search reconstruct the same 768/1024 geometry.
- Search preprocessing builds `QGQuery` with `padded_dim_`
  (`include/index/graph/laser/qg/qg.hpp:1015-1029`). During a row scan, exact L2
  uses raw `dimension_`, whereas approximate scanning reads the packed-code and
  factor offsets computed from padded width
  (`include/index/graph/laser/qg/qg.hpp:1299-1326`).
- The disk searcher passes the caller's raw query to `QuantizedGraph::search`
  and translates returned PIDs through the label sidecar
  (`include/index/disk/laser_segment_searcher.hpp:299-345`). The public query
  width therefore remains 768; callers never supply the 1024-element padding.

## Gate changes

- `LaserSegmentImporter` replaced the old `power-of-two && dim>=128` checks
  with one shared v1 range predicate `[33,2048]`. The error now reports the
  supported range, and its code comment documents padding and storage cost
  (`include/index/disk/laser_segment_importer.hpp:33-47,227-247`). The existing
  `main_dim == 0 || main_dim == dim` restriction remains unchanged
  (`include/index/disk/laser_segment_importer.hpp:248-253`).
- Sealed Collection target support now uses the same predicate, so 768 reaches
  the real Vamana -> QGBuilder -> importer path rather than silently becoming
  Flat (`include/index/collection/detail/collection_target_builder.hpp:244-275,
  592-650`). The dimension-specific fallback diagnostic was updated in
  `include/index/collection/collection.hpp:780-785`.
- The active-LASER Collection admission shared the same stale copied gate and
  the same `QuantizedGraph`, so it was synchronized as well; its diagnostic now
  describes the range and padding (`include/index/collection/collection.hpp:858-895`).
- No native index bytes, manifest version, row formula, page addressing, or
  search kernel were changed.

## 768d Collection result

The new test builds 400 deterministic unit vectors at 768d with `R=32`,
`ef_construction=400`, and four build threads, seals through the public
Collection path, asserts both the receipt and persisted manifest identify a
real LASER segment, searches 20 perturbed queries at top-10, and compares IDs
with an exact L2 oracle
(`tests/collection/collection_laser_recall_floor_test.cpp:215-226,246-364,408-427`).

Result on Release/Linux x86_64, AVX-512 dispatch:

```text
measured_laser_collection_dim768_recall_at_10=1.0000
five consecutive build+search repetitions: 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
asserted floor: 0.85
```

The floor is intentionally below the small synthetic fixture's measured 1.0.
Its scale reference is the existing bare LASER gte768 full-main result,
recall@10 0.98637 at ef=200
(`docs/LASER_UPDATE_EXPLORATION.md:1325-1333,1344-1350`). That is a 1.007M-row
embedding benchmark rather than a dataset-equivalent CI fixture; 0.85 retains
the repository's existing Collection LASER noise margin while still detecting
collapse or an accidental non-LASER implementation.

## Positive, boundary, and negative coverage

- Importer and concrete FHT constructors accept 33, 63, 64, 96, 768, and 2048;
  both reject 32 and 2049, and importer failures check the new diagnostic
  (`tests/disk/test_laser_segment_importer.cpp:272-306`).
- Sealed Collection 768d builds and searches as LASER with exact-oracle recall;
  32d and 2049d each clear the independent 32-row floor, then demonstrably
  resolve to a persisted Flat segment with the range-specific fallback reason
  (`tests/collection/collection_laser_recall_floor_test.cpp:408-458`).
- Active Collection 768d creates, closes, and reopens as LASER; 32d and 2049d
  reject before filesystem persistence
  (`tests/collection/collection_active_laser_test.cpp:135-160`).

## Verification

Release was configured with testing explicitly enabled:

```text
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DPython_EXECUTABLE=<worktree>/.venv/bin/python3
cmake --build build/Release -j16
```

The configuration summary reported `Build Type: Release`, `BUILD_TESTING: ON`,
`ENABLE_LASER: ON`, and `MUTABLE_LASER: ON`; the final build exited 0. Before
the targeted regex run, `ctest -N -R` listed exactly three matching tests, so
the run was not a zero-match false green:

```text
test_laser_segment_importer
collection_laser_recall_floor_test
collection_active_laser_test
Total Tests: 3

targeted ctest: 3/3 passed (exit 0)
full ctest -j16: 94/94 passed, 0 failed, 30.58s (exit 0)
```

The first pre-commit pass made only clang-format rewrites; the final pass over
all changed files, including this report, passed every hook.

## Residual risks and non-goals

- Importer v1 still forbids a PCA/residual split (`main_dim` must be zero or the
  raw dimension). Consequently this Collection path cannot import a raw
  dimension above 2048 with a smaller supported main dimension, even though
  the lower-level QG format can represent residual dimensions. That is a
  separate capability change.
- The 768 CI fixture is deliberately small and deterministic. It guards wiring,
  engine identity, padding correctness, and catastrophic recall regression; it
  does not replace the 1.007M-row gte benchmark or establish a performance
  claim.
- Full regression was executed on Linux x86_64 with AVX-512 selected. Existing
  portable/AVX2 unit coverage passed, but this change did not run the new 768d
  Collection scenario on macOS, Windows, or non-x86 hardware.
- Padding has a real space cost and can cross a 4 KiB page boundary, as
  quantified above. This is an explicit format property, not a regression from
  the gate change; previously those already-supported artifacts were merely
  unreachable through the importer/Collection admission layer.

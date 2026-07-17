<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# ANN sealed-segment build cutover (`target_algorithm` → real ANN)

> **Superseded / historical (2026-07-17).** This record predates two later
> waves: HNSW (one of this document's two target engines) was retired
> entirely, and `CollectionOptions.target_algorithm`'s default flipped from
> `hnsw` to `qg`. Statements below about the default being `hnsw` and about
> HNSW as a live target no longer hold. See `CHANGELOG.md`'s `[Unreleased]`
> entry for the current state.

Status: design approved, implementation in progress. Target release: **1.1.0**.

## Problem

The canonical `alaya::Collection` facade is a complete control plane but its data
plane is exact-Flat only. `CollectionOptions.target_algorithm`
(`collection.hpp:69`, default `hnsw`) is validated and persisted as identity
strings but never drives a build: `seal_locked` (`collection.hpp:1271`) and
`compact_locked` (`:1572`) unconditionally call
`build_collection_flat_target(...)` (`collection_flat_target.hpp:200`), which
always builds an exact `DiskFlatSegment`. A collection built from inserted
vectors therefore always yields O(n) brute-force segments regardless of the
requested `index_type`; ANN only exists for externally imported DiskANN/LASER
segments. This document specifies wiring `target_algorithm` into seal/compact so
the advertised memory engines (hnsw/nsg/fusion/qg) actually build ANN.

Constraints: **no frozen `include/core/*.hpp` change** (additive-only); **no
RocksDB / no pre-1.1 backward-compat** (1.1.0 is a hard break, see
`legacy-cleanup.md`); a requested target that cannot be honored must be an
explicit error or an **honestly-labelled** Flat fallback — never a Flat segment
masquerading under an HNSW/QG identity.

This design was produced by two independent passes (a structured-reader workflow
and a max-effort codex review) that converged on the same seam; the phased scope
below reflects their reconciliation.

## The seam

Two new detail-layer components, plus a registry as the single identity
authority. Nothing in `include/core` changes.

### Build dispatcher

Add `include/index/collection/detail/collection_target_builder.hpp`; keep
`collection_flat_target.hpp` as the `flat` implementation it delegates to.

```cpp
struct CollectionTargetBuildParams {   // maps from CollectionOptions
  CollectionQuantization quantization{CollectionQuantization::none};
  std::uint32_t max_neighbors{32};     // HNSW/NSG/Fusion M, Vamana R
  std::uint32_t ef_construction{400};  // graph L / QG ef_build
  std::uint32_t thread_count{1};
  float alpha{1.2F};
  std::uint64_t seed{1234};
};

struct CollectionTargetBuildResult {
  core::AnySegment segment{};
  core::AlgorithmId requested_algorithm{};
  core::AlgorithmId built_algorithm{};   // == flat on fallback
  std::string implementation_key{};      // == SegmentEntryV2.required_features
  std::string factory_key{};             // == SegmentEntryV2.factory_key
  std::uint64_t artifact_bytes{};
  bool flat_fallback{};
  std::string fallback_reason{};
};

auto build_collection_target(core::AlgorithmId requested_algorithm,
                             const CollectionSchema &schema,
                             std::span<const RegisteredRow> rows,
                             const CollectionTargetBuildParams &params,
                             const CollectionTargetPublication &publication,
                             core::BuildContext &context)
    -> core::Result<CollectionTargetBuildResult>;
```

`CollectionTargetPublication` carries the same fields the two existing
`Disk*PublicationOptions` blocks already require (collection_root, segment_id,
generation, manifest_generation, wal_cut, row_versions, id_map_checkpoint,
collection_features, abort_policy, fail_point, base_manifest) — a bare `dir` is
insufficient. `DiskFlatPublicationOptions` and `DiskVamanaPublicationOptions` are
field-for-field identical (`disk_flat_segment.hpp:47-63` vs
`disk_vamana_segment.hpp:47-64`), so the block transplants 1:1.

### Registry (identity authority)

```cpp
struct CollectionTargetRegistration {
  core::AlgorithmId algorithm_id{};
  std::string_view implementation_key{};   // required_features string
  std::string_view factory_key{};          // SegmentEntryV2.factory_key
  TargetSupport (*supports)(const CollectionSchema &, core::RowCount,
                            const CollectionTargetBuildParams &);
  BuildTargetFn build{};
  OpenTargetFn open{};
};
```

`target_implementation_key()` / `target_engine_factory_key()`
(`collection.hpp:580-611`) query this registry instead of hand-maintained
switches, so the persisted `SegmentEntryV2.{algorithm_id,factory_key}` always
equals the record that built and reopens the segment. This is what makes those
identity accessors real. **Reopen must dispatch on each entry's persisted
`factory_key`, not the current collection option** (a snapshot can mix
fallback-Flat, older targets, and imported heterogeneous segments).

### Keyed reopen factory

Add `include/index/collection/detail/collection_segment_factory.hpp`:

```cpp
class CollectionSegmentFactory {
 public:
  static auto open_entry(const std::filesystem::path &collection_root,
                         const SegmentEntryV2 &entry,
                         const CollectionSchema &schema,
                         core::OpenContext &context) -> core::Result<core::AnySegment>;
};
```

It dispatches on `entry.factory_key`, cross-checks `entry.algorithm_id`, maps the
entry's own logical artifact names to absolute paths, calls the engine `open`,
and erases via `into_any`. Replaces the unconditional
`open_collection_flat_entry` call at `open_segmented` (`collection.hpp:738`),
which today rejects any `algorithm_id != flat`
(`collection_flat_target.hpp:137`). New engine `required_features` strings must be
added to `ManifestReaderOptions.available_features` (`manifest_dual_reader.hpp:35`).

### Injection points (exactly three)

1. **Build** — `seal_locked:1271` and `compact_locked:1572`: call the dispatcher
   with `options_.target_algorithm`. Everything after (patch manifest, install,
   checkpoint, receipt) is already algorithm-agnostic
   (`install_segment_replacement_locked` checks only search-cap + dim/metric/
   scalar; `patch_published_target_manifest` locates by `seg_%08d` name only).
2. **Reopen** — `open_segmented:738`: `CollectionSegmentFactory::open_entry`.
3. **Identity/validation** — `validate_options:666` accept the real engine ids;
   registry supplies the key strings.

### Engine-neutral supporting changes

- Rename `flat_segment_name(uint64)` → `collection_segment_name(uint64)`; use in
  seal/compact/recovery/GC. Artifact dir stays `root/segments/seg_########`.
- **Dense-live-first row IDs**: replace the `preserve_source_row_ids` bool in
  `collect_replacement_rows` (`collection.hpp:819`) with a two-pass plan — live
  source versions get `0..live_count-1` in vector-build order, tombstones after.
  Memory graphs return dense native node IDs and accept no label array, so this
  is required for them; it also satisfies `install_segment_replacement_locked`'s
  "every current source version has a replacement" invariant
  (`segmented_collection.hpp:990-1013`). Sealed physical row IDs are internal
  `SegmentRowId`s with no external promise (confirm no golden pins them).
- **Compact source selection** (`collection.hpp:1473`): select every
  Collection-owned sealed source whose live versions still carry vectors (the
  checkpoint `RecordPayload` is the raw-vector source,
  `collection_checkpoint.hpp:42-74`), not only `algorithm_id==flat`. Replace
  `verify_flat_exports` with an engine-neutral eligibility check; imported
  rank-only/disk segments without payload vectors stay unselected. ANN sources
  thus need no `export_rows` (DiskVamana is not `Exportable`).

## Engine inventory

| Engine | build → segment | publishes SegmentEntryV2? | medium | first-class sealed target? |
|---|---|---|---|---|
| DiskFlat | `DiskFlatSegment::build` | yes (self) | disk | yes — the `flat` case / fallback |
| DiskVamana | `DiskVamanaSegmentFactory::build` | yes (self) | disk | **Phase-1 canary** (L2/f32 only) |
| HNSW | `HnswSegment::build` (throws; needs Space) | **no** — `save()` only fills `ArtifactManifest` | memory | Phase 2 — needs txn/factory wrapper |
| NSG | `MemoryGraphSegmentBase::build` (≥65 rows) | no | memory | Phase 3 |
| Fusion | `MemoryGraphSegmentBase::build` (≥65 rows) | no | memory | Phase 3 |
| QG | `QgSegment::build` (needs RaBitQSpace, f32, >32 rows) | no | memory | Phase 4 |
| DiskANN | `DiskANNIndex::build` (low-level dir writer) | no (open-only Segment) | disk | Phase 5 (not a 1.1.0 gate) |

Memory engines need a Collection-owned wrapper: run `ArtifactControlPlaneTransaction`
(`begin → writer(specs) → engine.save(writer,…) → prepare(entry) → publish`),
synthesise `SegmentEntryV2` with the real `algorithm_id`/`factory_key`,
`CapabilitiesSnapshotV2::from_runtime(any.capabilities())`, and a reopen path.
DiskVamana already does all of this internally — which is exactly why it is the
cheapest canary but **not** the release deliverable.

## Parameters

`CollectionOptions` today carries only `build_threads` + `ef_construction` (the
latter never reaches a builder). Add additive fields `max_neighbors`,
`build_alpha`, `build_seed`; map to each engine's options struct via
`target_params_from_options()`. Facade schema: adopt a clean **schema v2** (no
format-1 reader — the hard break permits it). Python: forward
`IndexParams.max_nbrs` (currently dropped) and stop hard-coding
`ef_construction=400` in `collection.py:215`; add the pybind `create()` args.
`nlist`/`nbits` do not map to any current engine — reject as inapplicable.

## Routing / score contracts

The only merge gate is `normalize_scores` (`segmented_collection.hpp:1744`):
all candidates for a query must share `(score_kind, comparable_metric)`; the
active exact-Flat generation defines the domain `{distance, schema.metric}` on
the `exact_distance_typed` scale (squared-L2 / -dot / -cos). `ResultFlag::approximate`
never gates the merge.

- **Contract A** (full-precision engines — HNSW/NSG/Fusion/DiskVamana): emit
  `score_kind=distance`, `comparable_metric=schema.metric`, finite score on the
  exact scale. Merges directly. Proven bit-exact in
  `heterogeneous_segment_integration_test`.
- **Contract B** (lossy — QG/RaBitQ, LASER): either emit a calibrated distance
  from the retained vector, or set `score_kind=rank_only` and register a
  `SegmentRegistration.exact_rerank` closure (`types.hpp:170`) over the
  Collection-owned vectors — else merge hard-fails or silently ranks incorrectly.

Two routing fixes before enabling ANN seal: (1) filter the shared
`SearchOptions.extensions` span per target descriptor (engines reject other-algorithm
extensions); (2) synthesise per-segment search effort ≥ `candidate_limit` (engines
default effort 100 and reject `effort<top_k`). Cosine needs a Collection-owned
query-normalization adapter (memory spaces treat cosine as -dot); int8/uint8
cosine falls back to Flat.

## Fallback policy

Fallback is a **pre-build planning result**, not exception recovery. Fall back to
Flat only for: explicit `flat`; a generation below an engine minimum (NSG/Fusion
<65, QG ≤32, Vamana <2); a schema outside an engine's proven matrix (non-L2/non-f32
for disk engines, int8 cosine for memory graphs); a deliberately-disabled writer
bit. **Never** fall back for unknown/invalid params, cancellation, budget denial,
I/O/corruption, or descriptor mismatch — those are errors. Persist the real Flat
identity plus `requested_algorithm`/`fallback_reason` extensions and receipt/stats.

## Phased plan

| Phase | Scope | Gate |
|---|---|---|
| **0** | registry + dispatcher + keyed reopen factory + engine-neutral naming + dense-live-first + `flat` routed through dispatcher + fallback provenance | Flat behaviour unchanged; all existing tests green |
| **1** | DiskVamana canary (L2/f32): seal, auto-seal, reopen, compact, kill-recovery, GC | seam proven end-to-end; stays internal (not aliased to hnsw) |
| **2** | HNSW: memory-engine txn/factory wrapper, raw/int8/uint8 + SQ4/SQ8, cosine adapter, per-segment extension/effort | **default type builds real ANN — 1.1.0 data-plane floor** |
| **3** | NSG + Fusion (≥65-row fallback, ef clamp policy) | |
| **4** | QG / RaBitQ (f32, explicit qg+rabitq, >32 rows, Contract A calibrated or B rerank; non-deterministic golden policy) | |
| 5 | DiskANN sealed builder (Collection wrapper over `DiskANNIndex::build`) | **not** a 1.1.0 gate |

**1.1.0 release scope (approved):** Phases 0–4 — all advertised memory types
(hnsw/nsg/fusion/qg) build real ANN. No advertised target may ship permanently on
a default-on Flat fallback.

## Tests & golden

Per-engine recall floors gate the writer bit: HNSW ≥0.95, NSG/Fusion/Vamana ≥0.90,
QG ≥0.80 @recall@10, on a ≥10k×64 deterministic set vs an exact oracle; facade
result must equal direct search on the persisted engine artifact. Generalise the
Flat-only lifecycle tests (`collection_facade_test.cpp:256+`) into per-target
families with fixtures large enough to clear the min-row fallback (else they pass
while the data plane is Flat). Extend `heterogeneous_segment_integration_test`
(active Flat + reopened HNSW; dense-live-first holes; top-k>100; rank-only LASER
with/without rerank). Add `collection_sealed_<engine>` composite golden families
(single-thread fixed seed; QG uses the differential/fixed-artifact strategy).

**Independent prerequisite (D3):** `generate_artifact_baseline.py:68` hard-requires
the deleted `artifact_legacy_v1_generator`; golden regen is broken on a clean
tree. Fix separately — replace with current-format generators / drop dead v1
families; do **not** restore the deleted legacy generator.

## Open decisions

- Public `index_type` for `vamana`/`diskann`: keep internal unless intentionally
  supported (recommended internal for 1.1.0).
- QG merge: Contract A (calibrated distance from retained vector) vs Contract B
  (rank_only + rerank) — decide in Phase 4 by measured recall.
- Small auto-seal generations: immediate honest Flat fallback (recommended) vs
  delayed seal.
- Multi-generation memory-resident footprint: add resident-size admission +
  compaction trigger.

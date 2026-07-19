<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# RaBitQ build and format contracts

AlayaLite has two RaBitQ graph pipelines with different rotation mathematics
and storage contracts:

- `memory_qg::Builder<RaBitQSpace<...>>` is a transient, topology-only build
  path. It returns a `FrozenGraphSnapshot` and has no serving or serialization
  surface.
- `disk_laser_qg::{Builder, Graph}` is LASER's persisted and searchable
  quantized graph. The compatibility names live in
  `include/index/graph/qg_naming.hpp`.

The former memory-QG v1 snapshot is retired. There is no `QgSegment`, artifact
reader/writer, golden family, or Collection legacy builder. A manifest that
requires `qg_segment` is recognized only so the opener can return an explicit
`not_supported` error containing `legacy qg_segment` and `re-seal`.

## Contract comparison

| Property | Memory-QG builder scratch | LASER v1 format |
| --- | --- | --- |
| Lifecycle | Build-local; discarded after snapshot export | Persisted, opened, and served |
| Graph surface | `alaya::memory_qg::Builder<RaBitQSpace<...>>` | `alaya::disk_laser_qg::{Builder, Graph}` |
| Output | `FrozenGraphSnapshot` adjacency, entry point, and degree bound | LASER `.index` plus rotator/cache sidecars |
| Rotation | `RaBitQSpace` uses its configured `RotatorType` while constructing topology. The default is `FhtKacRotator`. No rotator bytes are exported. | The graph owns the single-round `laser::FHTRotator`; its sign-scaled vector is stored in the `_rotator` companion artifact. |
| Padding | Builder scratch follows the selected RaBitQSpace rotator and its 64-wide fast-scan rules. | The public path admits `33 <= main_dim <= 2048` and uses `padded_dim = 2^ceil(log2(main_dim))` for FHT, codes, and FastScan while retaining `main_dim` for raw/exact terms. |
| Binary sign convention | A positive residual produces bit 1; `fastscan::pack_codes` transposes groups of 32 scratch codes. These bytes never cross the builder boundary. | A positive residual produces bit 1. Bits pass through LASER's 64-bit word byte/nibble reversal before its 32-code transpose. |
| Factors | Two scratch structure-of-arrays blocks: `f_add`, then `f_rescale`. | Three persisted structure-of-arrays blocks: `triple_x`, then `factor_dq`, then `factor_vq`. |

The IP/cosine Collection bridge deliberately uses the memory-QG builder for
metric-aware topology and then passes the resulting `FrozenGraphSnapshot` to
`laser::QGBuilder::build_from_graph`. The memory builder's fixed degree limits
those persisted LASER paths to `R <= 32`. L2 continues to use native Vamana
topology.

## Retired memory-QG v1 bytes

The retired snapshot interleaved each node's raw vector, packed neighbor codes,
`f_add`, `f_rescale`, and neighbor IDs, with rotator state embedded in the same
file. Those bytes are not a LASER index and must never be imported as one.
`rabitq_format_separation_test.cpp` still mints such bytes directly as hostile
input and verifies that LASER rejects them; this does not restore a reader or
make the old layout a current artifact family.

## Factor mapping to the shared core

`space/quant/rabitq_core.hpp` expresses the common calculation as
`{base, signed_query_scale}`. For the L2 path:

- builder scratch `f_add = base`; LASER `triple_x = base` (plus any
  residual-dimension norm used while assembling the disk node);
- builder scratch consumes half-sign query values, so
  `f_rescale = 2 * signed_query_scale`;
- LASER consumes full-sign query values, so
  `factor_dq = signed_query_scale = f_rescale / 2`;
- `factor_vq = factor_dq * (2 * popcount(sign_bits) - padded_dim)` supplies
  LASER's scalar-query correction.

`tests/laser/rabitq_factor_equivalence_test.cpp` pins the mathematical mapping,
code layout, and complete-estimator equivalence. It also records the historical
zero-residual difference: the memory calculation emits zero factors while
LASER v1 retains its NaN-factor behavior. LASER v1 bytes must not be silently
rewritten.

## Consumer matrix

| Producer or consumer | Required contract |
| --- | --- |
| `memory_qg::Builder`, `QgBuilderKernel`, `QgBuildGraph`, `RaBitQSpace` | Build-local scratch; export only `FrozenGraphSnapshot` |
| `include/index/graph/laser/qg/`, LASER builder/searcher | LASER v1 artifacts |
| `LaserSegmentImporter`, `LaserSegment`, disk LASER factory/searcher | LASER v1 artifacts only |
| Collection entry requiring `qg_segment` | Explicit `not_supported`; re-seal required |

Python and manifest dispatch still use the historical string `"QG"` for LASER
in several APIs. That string is behavior compatibility, not a format
discriminator. Dispatch, WAL, segment, or factory code must resolve the engine
before selecting a decoder.

The open-only AnySegment contract, rank-only score domain, manifest-v2
publication, and runtime fallback rules are specified in
[LASER immutable disk segment](laser-segment.md).

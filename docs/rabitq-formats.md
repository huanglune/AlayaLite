<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# RaBitQ format contracts

AlayaLite has two mathematically equivalent RaBitQ graph pipelines with different storage
contracts. Use **memory_qg** for the in-memory graph built around `RaBitQSpace`, and
**disk_laser_qg** for LASER's disk-resident quantized graph. The compatibility aliases live in
`include/index/graph/qg_naming.hpp`. The memory surface is the public
`QgSegment`; its former public builder has moved to a detail-only kernel. LASER
keeps its historical disk builder and graph names.

`MemoryRaBitQFormat` and `LaserRaBitQFormat` below are conceptual format names. They are not C++
types or interchangeable wire formats.

## Contract comparison

| Property | `MemoryRaBitQFormat` | `LaserRaBitQFormat` |
| --- | --- | --- |
| Current version | v1, implicit (no format tag in the snapshot) | v1, implicit (no format tag in the index page) |
| Graph surface | `alaya::memory_qg::Segment<RaBitQSpace<...>>` | `alaya::disk_laser_qg::{Builder, Graph}` |
| Rotation | `RotatorType` is serialized in the snapshot. The default is `FhtKacRotator`: four sign-flip/FHT rounds (with Kac walks when padding requires them). `MatrixRotator` and the registered `FhtRotator` are selectable legacy alternatives. | The graph owns the concrete single-round `laser::FHTRotator`, corresponding to `RotatorType::FhtRotator`. Its sign-scaled vector is stored in the companion `_rotator` artifact. |
| Padding | The selected rotator determines padding; the default path rounds up for 64-wide fast scan and supports the FhtKac padding/truncation rules. | `padded_dim = 2^ceil(log2(main_dim))`, and the current graph constructor requires `main_dim == padded_dim` (a power of two). Node/page tails are zero-filled. |
| Binary sign convention | A residual component `> 0` produces bit 1. The initial per-vector byte is formed most-significant-bit first, then `fastscan::pack_codes` transposes groups of 32 codes into nibble/SIMD order and zero-pads missing lanes. | A residual component `> 0` produces bit 1. Bits are first packed through 64-bit words, then each word's byte order and each byte's nibbles are reversed before LASER's 32-code nibble transpose; missing lanes are zero-padded. The intermediate bit/byte order is therefore a separate contract even where the current final fast-scan blocks compare equal. |
| Factors | Two structure-of-arrays blocks: `f_add[degree]`, then `f_rescale[degree]`. | Three structure-of-arrays blocks: `triple_x[degree]`, then `factor_dq[degree]`, then `factor_vq[degree]`. `laser::Factor` documents this field order; page storage is not an array of `Factor`. |
| Serialization destination | `RaBitQSpace::save` snapshot. Each stored node is raw vector, packed neighbor codes, all `f_add`, all `f_rescale`, then neighbor IDs; rotator state is also embedded in the snapshot. | LASER `.index` pages after a 4096-byte metadata sector. Each node payload is raw main/residual vector, packed codes, all three factor arrays, then neighbor IDs. Rotation is a companion artifact. |

Neither consumer may reinterpret or import the other format. Mathematical agreement is not byte
compatibility.

## Factor mapping to the shared core

`space/quant/rabitq_core.hpp` expresses the common result as
`{base, signed_query_scale}`. For the L2 path:

- `f_add = base` and `triple_x = base` (LASER may add the residual-dimension norm to
  `triple_x` when assembling the disk node).
- The legacy memory estimator consumes half-sign query values, so
  `f_rescale = 2 * signed_query_scale`.
- LASER consumes full-sign query values, so `factor_dq = signed_query_scale = f_rescale / 2`.
- `factor_vq = factor_dq * (2 * popcount(sign_bits) - padded_dim)` supplies LASER's scalar-query
  correction.

`tests/laser/rabitq_factor_equivalence_test.cpp` is the executable specification for factor,
code, and complete-estimator equivalence. It also pins the intentional exact-zero difference:
when the residual norm `r` is zero, memory v1 emits zero factors while LASER v1 retains its
historical NaN factors. A future, explicitly versioned format upgrade should unify that policy;
v1 bytes must not be silently rewritten.

## Consumer matrix

| Producer or consumer | Required contract |
| --- | --- |
| `QgSegment`, `RaBitQSpace`, the detail QG build kernel, in-memory graph search/executor | `MemoryRaBitQFormat` |
| `include/index/graph/laser/qg/`, LASER builder/searcher | `LaserRaBitQFormat` |
| `LaserSegmentImporter`, disk LASER segment factory/searcher | `LaserRaBitQFormat` artifacts only |

Python and manifest dispatch still use the historical string `"QG"` for LASER in several APIs.
That string is behavior compatibility, not a format discriminator. Dispatch, WAL, segment, or
factory code must resolve the engine first and must not select a RaBitQ decoder from the bare
`"QG"` token. Renaming that public behavior is outside this contract change.

For the legacy memory `Index` mapping from declared HNSW/NSG/Fusion rows to the
actual QG engine, including honest descriptor/runtime keys and the Gate 9 plan,
see [Memory QG legacy dispatch contract](memory-qg-legacy-dispatch.md).

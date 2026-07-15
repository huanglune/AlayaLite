<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Memory QG legacy dispatch contract

The legacy Python `Index` entry treats `quantization_type="rabitq"` as the
decisive engine selection. The requested `index_type` is retained as the
declared compatibility identity, but all three legal rows build and open the
same in-memory QG engine:

| legacy declared `index_type` | quantization | actual Segment | `Descriptor.algorithm_id` | current runtime keys | feature-disabled keys | artifact family |
| --- | --- | --- | ---: | --- | --- | --- |
| `hnsw` | `rabitq` | `QgSegment<RaBitQSpace<...>>` | `qg` (`5`) | `qg_segment / qg` | `legacy_qg_model / qg` | memory QG v1 |
| `nsg` | `rabitq` | `QgSegment<RaBitQSpace<...>>` | `qg` (`5`) | `qg_segment / qg` | `legacy_qg_model / qg` | memory QG v1 |
| `fusion` | `rabitq` | `QgSegment<RaBitQSpace<...>>` | `qg` (`5`) | `qg_segment / qg` | `legacy_qg_model / qg` | memory QG v1 |

This mapping applies to both scalar-disabled and scalar-enabled
specializations. It is pinned compatibility behavior, not HNSW-over-RaBitQ,
NSG-over-RaBitQ, or Fusion-over-RaBitQ. The declared name must not be copied
into the runtime descriptor, implementation key, engine key, manifest, or
artifact fingerprint.

The migration does not change what the legacy entry builds, how it searches,
or the bytes written by the retained `RaBitQSpace` codec. It changes runtime
introspection so the implementation reports QG honestly. Disabling only
`EngineFeature::qg_segment` selects the recorded `legacy_qg_model / qg`
factory and preserves the former direct path.

The legacy QG builder and rotator contain historical random sources, so two
independent builds are not promised to be byte-identical even for the same
input. Codec stability is instead pinned by byte-identical save/open/save and
by opening one QG artifact through all three declared rows, then checking that
their saved bytes and search results are identical. The checked-in golden QG
v1 family uses fixed rotator bytes and fixed neighbors so its format hash is
reproducible without changing the production quantization implementation.

Gate 9's canonical `Collection` entry will require an explicit QG declaration.
The table above remains as a compatibility quirk of the legacy Python `Index`
wrapper; Gate 9 must not infer this mapping for the canonical entry.

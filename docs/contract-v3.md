<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Core segment contract v3 (frozen)

The v3 engine boundary is frozen as of Gate 2. Engines must use the types and
operation slots under `include/core`; they must not add private request,
response, resource, lifecycle, or error fields at a parallel boundary.
Evolution uses a versioned tail/reserved field, a keyed algorithm extension,
or a new operation-table version.

The compatibility promise is C++ source compatibility within the same
compiler and standard-library ecosystem. Release wheels are built by this
project. These C++ layouts are not a cross-compiler binary ABI, and the layout
`static_assert`s are same-toolchain regression canaries only. Every erased
struct starts with `struct_size` and `abi_version`; the operation table starts
with `table_size` and `table_version`.

## Frozen surface

| Design section | Code owner | Frozen shape |
|---|---|---|
| §3.1 | `core/versioned.hpp`, `core/status.hpp` | v3 prefixes, stable `StatusCode`, stage/detail/retry/partial, `Result<T>`, exception conversion |
| §2.3/§3.2 | `core/value_types.hpp` | owning/view `LogicalId`, fixed-width `SegmentRowId`, deprecated `ExternalId` alias, `TypedTensorView` for float32/int8/uint8 |
| §3.3 | `core/value_types.hpp` | common `SearchOptions` goals plus algorithm-ID-keyed versioned payloads |
| §3.4 | `core/value_types.hpp` | flat `SearchHit` sink plus offsets, valid counts, per-query status/completeness, score domain and flags; no valid sentinel |
| §3.5 | `core/capabilities.hpp`, `core/any_segment.hpp` | static `Descriptor`, slot/config-derived `RuntimeCapabilities`, dynamic `SegmentStats`, split concepts, versioned operation table |
| §3.6 | `core/any_segment.hpp` | `start_search(request, completion) -> Result<OperationHandle>`; cooperative idempotent cancel; sync wait wrapper |
| §3.7 | `core/resource_contexts.hpp` | Open/Build/Mutation/Seal/Search/Checkpoint lease, credit, deadline, cancel and lane slots; explicit budget denial |
| §3.8 | `core/any_segment.hpp`, context types | admission status, optional close/drain slots, operation state pinned through exactly-once completion |

Stable algorithm IDs live in `core/algorithm_registry.hpp`, which has no
domain dependencies. Legacy disk/index/metric conversions live one layer up in
`index/compat.hpp`; `core/compat.hpp` is only a source forwarding bridge for
old algorithm constant names.

## Input and output invariants

- A zero-row tensor is valid and may have null data. A non-empty, non-zero-dim
  tensor may not. Row byte width and total extent use checked arithmetic.
- `top_k == 0` completes successfully without invoking an engine slot. Every
  query receives zero valid count, equal offsets, `ok`, and `complete_k`.
- A result slice is valid only when covered by response offsets/counts and its
  per-query status/completeness. With partial results disabled, overall failure
  zeros all offsets/counts and marks every query failed; bytes already written
  to the sink must be ignored.
- Numeric NaN is never a valid hit. Scores merge only within the same
  `score_kind` and `comparable_metric`; rank-only output requires exact rerank
  or rejection at the future Collection layer.
- An explicitly insufficient lease returns `resource_exhausted` with stable
  `budget_denied` detail. Gate 10 strengthens accounting policy without
  changing these signatures.

The sync-engine adapter schedules work off the caller stack, dispatches the
completion through the request lane, catches engine exceptions at the erased
boundary, and holds the request lifetime pin until the completion callback.
Cancel and timeout never authorize early buffer release.

Gate 2 does not introduce Collection routing, LogicalId persistence, WAL,
manifest v2, or Python facade switching. Those remain Gates 4 and later.

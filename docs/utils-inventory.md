<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# include/utils/ inventory

Audited at refactor/legacy-cleanup after log/platform/platform_fs/metric_type
migrated to core/.

## Classification

| File | Ext consumers | Consumer layer | Disposition | Rationale |
|------|---:|---|---|---|
| binary_io.hpp | 1 | collection (legacy_importer) | keep | format reader dependency |
| coro_gate.hpp | 2 | graph (diskann_index) + test | keep | coroutine gate for DiskANN async |
| data_utils.hpp | 2 | space (raw_space) + test | keep | vector distance helpers |
| dataset_utils.hpp | 7 | tests only | keep (test infra) | fvecs loader used by 7 test files |
| evaluate.hpp | 8 | tests only | keep (test infra) | recall/precision calculators |
| index_encoding.hpp | 2 | storage (rocksdb_storage) + test | keep | ID encoding for RocksDB keys |
| io_utils.hpp | 1 | dataset_utils (internal) + benchmark | keep (test infra) | fvecs/bvecs file I/O primitives |
| locks.hpp | 1 | test (search_test) | keep | FileLock used by cached test fixtures |
| macros.hpp | 3 | mixed (diskann, LASER, space) | keep | ALAYA_LIKELY/UNLIKELY, alignment |
| math.hpp | 6 | mixed (simd, space, LASER, diskann) | keep | bit scan, alignment math |
| memory.hpp | 2 | storage (static_storage) + utils internal | keep | aligned alloc/free |
| metadata_filter.hpp | 0 | only metadata_filter_matcher (internal) | keep (legacy leaf) | old filter AST; Gate 10 FilterExecution replaces for canonical path |
| metadata_filter_matcher.hpp | 1 | test only | keep (legacy leaf) | old filter executor; pairs with metadata_filter |
| openmp.hpp | 4 | graph (vamana, HNSW, fusion) + test | keep | OpenMP thread config |
| prefetch.hpp | 8 | mixed (diskann, LASER, graph, space) | keep | cache prefetch intrinsics |
| query_utils.hpp | 3 | graph (search_runtime) + test | keep | result sorting/dedup |
| random.hpp | 3 | space + LASER + test | keep | seeded random generators |
| scalar_data.hpp | 5 | space (SQ4/SQ8) + collection (importer) + storage | keep | ScalarData/EmptyScalarData type |
| thread_config.hpp | 9 | mixed (graph, tests, pybind) | keep | configured_thread_limit() |
| thread_pool.hpp | 4 | graph (diskann, vamana) + test | keep | ThreadPool for build parallelism |
| timer.hpp | 14 | mixed (all layers + tests) | keep | Timer utility class |
| **types.hpp** | **0** | **none** | **deleted** | WorkerID/CpuID defined but zero uses |

### rabitq_utils/ (7 files, all kept)

| File | Ext consumers | Consumer layer | Rationale |
|------|---:|---|---|
| defines.hpp | 3 | LASER + space (rabitq) | RaBitQ constants and Eigen types |
| fastscan.hpp | 3 | LASER + space (rabitq) | SIMD FastScan kernels |
| lut.hpp | 2 | LASER + space (rabitq) | lookup table generation |
| rotator.hpp | 4 | LASER + space (rabitq) | FHT rotation |
| search_utils/buffer.hpp | 4 | graph (search_runtime) + space | candidate buffer |
| search_utils/hashset.hpp | 2 | LASER + space | open-addressing hash set |
| search_utils/visited_pool.hpp | 1 | graph (search_runtime) | thread-local visited bitset pool |

## Summary

- **Deleted**: 1 file (types.hpp, 8 lines)
- **Migrated to core/ (prior step)**: 4 files (log, platform, platform_fs, metric_type)
- **Remaining**: 28 files in utils/, all with active consumers
- **Test infrastructure**: dataset_utils, evaluate, io_utils (consumed by 7-8 test files each)
- **Legacy leaf**: metadata_filter + metadata_filter_matcher (zero production consumers, only test; replaced by Gate 10 FilterExecution for canonical path)
- **Future candidates**: metadata_filter pair could be deleted when old filter tests are rewritten to use Collection FilterExecution; scalar_data could be simplified after SQ Space refactor

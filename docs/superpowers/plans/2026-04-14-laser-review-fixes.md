# LASER Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix confirmed bugs, safety issues, and code quality problems in `quantized_graph.hpp` and `laser_builder.hpp` identified during code review.

**Architecture:** Targeted fixes to two header files, grouped by severity. Critical correctness/safety fixes first (parameter validation, AIO error handling, RAII guard), then code quality improvements (write validation, logging, notify optimization, refactoring, lint). Each task is independently committable.

**Tech Stack:** C++20, GTest, libaio, Eigen, OpenMP

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `include/index/laser/quantized_graph.hpp` | Modify | Parameter validation, AIO error check, RAII guard, notify fix, drain loop extraction, NOLINT cleanup, node layout docs |
| `include/index/laser/laser_builder.hpp` | Modify | Write stream validation, error logging |
| `tests/index/laser_index_test.cpp` | Modify | Add QuantizedGraph constructor validation tests, set_params validation tests |

---

### Task 1: Parameter validation in QuantizedGraph constructor

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:251-268`
- Modify: `tests/index/laser_index_test.cpp` (append new tests)

- [ ] **Step 1: Write failing tests for constructor validation**

Append to `tests/index/laser_index_test.cpp`:

```cpp
// ==========================================================================
// QuantizedGraph constructor validation
// ==========================================================================

TEST(QuantizedGraphTest, ConstructorRejectsMainDimGreaterThanDim) {
  EXPECT_THROW(symqg::QuantizedGraph(100, 64, 256, 128), std::invalid_argument);
}

TEST(QuantizedGraphTest, ConstructorRejectsZeroMaxDegree) {
  EXPECT_THROW(symqg::QuantizedGraph(100, 0, 128, 256), std::invalid_argument);
}

TEST(QuantizedGraphTest, ConstructorRejectsZeroNumPoints) {
  EXPECT_THROW(symqg::QuantizedGraph(0, 64, 128, 256), std::invalid_argument);
}

TEST(QuantizedGraphTest, ConstructorRejectsNonPowerOf2Dim) {
  EXPECT_THROW(symqg::QuantizedGraph(100, 64, 96, 256), std::runtime_error);
}

TEST(QuantizedGraphTest, ConstructorAcceptsValidParams) {
  EXPECT_NO_THROW(symqg::QuantizedGraph(100, 64, 128, 256));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.claude/scripts/run_and_log.sh test_qg_ctor_fail .claude/scripts/run_and_log.sh build_for_test make build && ./build/tests/index/laser_index_test --gtest_filter="QuantizedGraphTest.*"`
Expected: `ConstructorRejectsMainDimGreaterThanDim` and `ConstructorRejectsZeroMaxDegree` and `ConstructorRejectsZeroNumPoints` FAIL (no validation exists yet).

- [ ] **Step 3: Add validation guards to constructor**

In `include/index/laser/quantized_graph.hpp`, replace the constructor body preamble (before the initializer list body):

```cpp
inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t main_dim, size_t dim)
    : num_points_(num),
      degree_bound_(max_deg),
      dimension_(main_dim),
      residual_dimension_(dim - main_dim),
      padded_dim_(1 << alaya::math::ceil_log2(main_dim)),
      scanner_(padded_dim_, degree_bound_),
      rotator_(main_dim),
      node_len_((32 * main_dim + 32 * (dim - main_dim) + 128 * max_deg + max_deg * padded_dim_) /
                8) {
```

Add validation **before** the initializer list by using a static helper or add guards at the top of the constructor body. Since we cannot validate before member init in C++ without a factory, add guards at the top of the body:

```cpp
  // --- existing body starts here ---
  if (num == 0) {
    throw std::invalid_argument("QuantizedGraph: num_points must be > 0");
  }
  if (max_deg == 0) {
    throw std::invalid_argument("QuantizedGraph: max_degree must be > 0");
  }
  if (main_dim == 0 || main_dim > dim) {
    throw std::invalid_argument("QuantizedGraph: main_dim must be in (0, dim]");
  }

  node_per_page_ = std::max(static_cast<size_t>(1), kSectorLen / node_len_);
  page_size_ = (node_per_page_ * node_len_ + kSectorLen - 1) / kSectorLen * kSectorLen;

  if (main_dim != padded_dim_) {
    throw std::runtime_error("Laser: dimension must be a power of 2");
  }
  initialize();
}
```

Note: `dim - main_dim` in the initializer list is safe after we confirm `main_dim <= dim` in the body guard. The subtraction in the initializer happens first, but `size_t` underflow produces a huge value that will be caught by the guard throwing before any use. For a truly safe approach, consider a static factory method in a future refactor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.claude/scripts/run_and_log.sh test_qg_ctor make build && ./build/tests/index/laser_index_test --gtest_filter="QuantizedGraphTest.*"`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/index/laser/quantized_graph.hpp tests/index/laser_index_test.cpp
git commit -m "fix: add parameter validation to QuantizedGraph constructor"
```

---

### Task 2: Parameter validation in set_params

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:322-334`
- Modify: `tests/index/laser_index_test.cpp` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/index/laser_index_test.cpp`:

```cpp
TEST(QuantizedGraphTest, SetParamsRejectsZeroThreads) {
  symqg::QuantizedGraph qg(100, 64, 128, 256);
  // Cannot call set_params without loading index, so expect logic_error for missing index
  // But we test that validation fires BEFORE the index check
  EXPECT_THROW(qg.set_params(200, 0, 4), std::invalid_argument);
}

TEST(QuantizedGraphTest, SetParamsRejectsNegativeBeamWidth) {
  symqg::QuantizedGraph qg(100, 64, 128, 256);
  EXPECT_THROW(qg.set_params(200, 4, -1), std::invalid_argument);
}

TEST(QuantizedGraphTest, SetParamsRejectsZeroBeamWidth) {
  symqg::QuantizedGraph qg(100, 64, 128, 256);
  EXPECT_THROW(qg.set_params(200, 4, 0), std::invalid_argument);
}

TEST(QuantizedGraphTest, SetParamsRejectsZeroEfSearch) {
  symqg::QuantizedGraph qg(100, 64, 128, 256);
  EXPECT_THROW(qg.set_params(0, 4, 4), std::invalid_argument);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.claude/scripts/run_and_log.sh test_set_params_fail make build && ./build/tests/index/laser_index_test --gtest_filter="QuantizedGraphTest.SetParams*"`
Expected: FAIL (no validation yet).

- [ ] **Step 3: Add validation to set_params**

Replace `set_params` in `quantized_graph.hpp:322-334`:

```cpp
inline void QuantizedGraph::set_params(size_t ef_search, size_t num_threads, int beam_width) {
  if (ef_search == 0) {
    throw std::invalid_argument("set_params: ef_search must be > 0");
  }
  if (num_threads == 0) {
    throw std::invalid_argument("set_params: num_threads must be > 0");
  }
  if (beam_width <= 0) {
    throw std::invalid_argument("set_params: beam_width must be > 0");
  }

  nthreads_ = num_threads;
  max_beam_width_ = static_cast<size_t>(beam_width);
  ef_search_ = ef_search;

  destroy_thread_data();

  if (index_file_name_.empty()) {
    throw std::runtime_error("Laser: load index before calling set_params()");
  }

  init_thread_pool();
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.claude/scripts/run_and_log.sh test_set_params make build && ./build/tests/index/laser_index_test --gtest_filter="QuantizedGraphTest.*"`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add include/index/laser/quantized_graph.hpp tests/index/laser_index_test.cpp
git commit -m "fix: validate set_params arguments before use"
```

---

### Task 3: Check AIO completion status in collect_events

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:475-487`

- [ ] **Step 1: Add io_event.res error check**

In `disk_search_qg`, modify the `collect_events` lambda (line ~475):

```cpp
  auto collect_events = [&](io_event *evts, int ret) {
    for (int i = 0; i < ret; i++) {
      auto id = static_cast<PID>(reinterpret_cast<uintptr_t>(evts[i].data));
      // Check AIO completion status: res < 0 means I/O error,
      // res != page_size_ means short read (partial page)
      if (evts[i].res < 0 ||
          static_cast<size_t>(evts[i].res) < page_size_) {
        // I/O failed or short read: reclaim slot, skip this node
        char *buf = ongoing.find(id);
        if (buf != nullptr) {
          ongoing.erase(id);
          free_slots.push(buf);
        }
        continue;
      }
      char *buf = ongoing.find(id);
      if (buf != nullptr) {
        const char *node_ptr = buf + offset_to_node(id);
        alaya::mem_prefetch_l2(node_ptr, prefetch_lines);
        prepared.push_back({id, buf});
        ongoing.erase(id);
      }
    }
  };
```

- [ ] **Step 2: Build and run existing laser tests to verify no regression**

Run: `.claude/scripts/run_and_log.sh test_aio_check make build && ./build/tests/index/laser_index_test`
Expected: All existing tests PASS.

- [ ] **Step 3: Commit**

```bash
git add include/index/laser/quantized_graph.hpp
git commit -m "fix: check AIO completion status before processing nodes"
```

---

### Task 4: ThreadData RAII guard for exception safety

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:108-135` (add nested class), `:349-375` (refactor search/batch_search)

- [ ] **Step 1: Add ThreadDataGuard nested class**

In the `private` section of `QuantizedGraph` (after `size_t nthreads_ = 1;`, around line 145), add:

```cpp
  struct ThreadDataGuard {
    alaya::ConcurrentQueue<ThreadData> &pool_;
    ThreadData data_;

    ThreadDataGuard(alaya::ConcurrentQueue<ThreadData> &pool, ThreadData &&data)
        : pool_(pool), data_(std::move(data)) {}

    ~ThreadDataGuard() {
      pool_.push(std::move(data_));
      pool_.push_notify_one();
    }

    ALAYA_NON_COPYABLE_NON_MOVABLE(ThreadDataGuard);
  };
```

Note: This uses `push_notify_one` (Task 6 fix included). Add `#include "utils/macros.hpp"` if not already present.

- [ ] **Step 2: Refactor search() to use guard**

Replace `search()`:

```cpp
inline void QuantizedGraph::search(const float *__restrict__ query,
                                   uint32_t knn,
                                   uint32_t *__restrict__ results) {
  ThreadDataGuard guard(thread_data_, acquire_thread_data());
  disk_search_qg(query, knn, results, guard.data_);
}
```

- [ ] **Step 3: Refactor batch_search() to use guard**

Replace `batch_search()`:

```cpp
inline void QuantizedGraph::batch_search(const float *__restrict__ query,
                                         uint32_t knn,
                                         uint32_t *__restrict__ results,
                                         size_t num_queries) {
  int num_threads = static_cast<int>(nthreads_);
#pragma omp parallel num_threads(num_threads)
  {
    ThreadDataGuard guard(thread_data_, acquire_thread_data());

    size_t full_dim = full_dimension();
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < num_queries; ++i) {
      disk_search_qg(query + i * full_dim, knn, results + i * knn, guard.data_);
    }
  }
}
```

- [ ] **Step 4: Remove release_thread_data method**

`release_thread_data` is now unused. Remove its declaration (line ~209) and implementation (lines ~386-389).

- [ ] **Step 5: Build and run existing tests**

Run: `.claude/scripts/run_and_log.sh test_raii_guard make build && ./build/tests/index/laser_index_test`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/index/laser/quantized_graph.hpp
git commit -m "fix: use RAII guard for ThreadData to prevent pool leak on exception"
```

---

### Task 5: Validate binary write streams in laser_builder.hpp

**Files:**
- Modify: `include/index/laser/laser_builder.hpp:438-459` (write_pca_transformed_base), `:60-135` (VamanaFormatWriter)

- [ ] **Step 1: Add stream check after PCA transform write loop**

In `write_pca_transformed_base`, after the write loop (after line 459), add:

```cpp
    if (!output.good()) {
      throw std::runtime_error("Failed to write PCA-transformed base (disk full?): " +
                               pca_base_path().string());
    }
```

- [ ] **Step 2: Add stream check in VamanaFormatWriter::finalize**

In `VamanaFormatWriter::finalize()`, before `output_.close()` (line ~124), add:

```cpp
    if (!output_.good()) {
      throw std::runtime_error("Failed to write Vamana graph (disk full?): " + path_.string());
    }
```

- [ ] **Step 3: Build to verify compilation**

Run: `.claude/scripts/run_and_log.sh build_write_check make build`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add include/index/laser/laser_builder.hpp
git commit -m "fix: validate binary write streams to detect disk-full corruption"
```

---

### Task 6: Replace push_notify_all with push_notify_one

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:386-389`

Note: If Task 4 is completed, `release_thread_data` is already removed and `ThreadDataGuard` destructor already uses `push_notify_one`. In that case, skip this task entirely.

If Task 4 was NOT completed (guard not yet applied):

- [ ] **Step 1: Change notify call**

In `release_thread_data`:

```cpp
inline void QuantizedGraph::release_thread_data(ThreadData &&data) {
  thread_data_.push(std::move(data));
  thread_data_.push_notify_one();
}
```

- [ ] **Step 2: Build and run tests**

Run: `.claude/scripts/run_and_log.sh test_notify make build && ./build/tests/index/laser_index_test`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add include/index/laser/quantized_graph.hpp
git commit -m "perf: use push_notify_one to avoid herd wakeup on ThreadData release"
```

---

### Task 7: Add warning logs for silent failure paths

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:839-861` (load_medoids), `:905-960` (load_cache)

- [ ] **Step 1: Add warnings to load_medoids failure paths**

In `load_medoids` (line ~850), after the failed load check:

```cpp
  if (!load_medoid_ids_from_disk(medoids_indices_file) ||
      !load_medoid_vectors_from_disk(medoids_file)) {
    std::cerr << "[WARN] Failed to load medoid data from: " << medoids_file
              << "; falling back to entry point only\n";
    medoids_.clear();
    medoids_vector_.clear();
    return;
  }
  size_t full_dim = full_dimension();
  if (medoids_.size() * full_dim != medoids_vector_.size()) {
    std::cerr << "[WARN] Medoid data size mismatch; falling back to entry point only\n";
    medoids_.clear();
    medoids_vector_.clear();
  }
```

- [ ] **Step 2: Add warnings to load_cache failure paths**

In `load_cache` (line ~905-960), add warnings at key failure points. After `stored_node_len != node_len_` check:

```cpp
  if (stored_node_len != node_len_) {
    std::cerr << "[WARN] Cache node_len mismatch (stored=" << stored_node_len
              << " expected=" << node_len_ << "); skipping cache\n";
    cache_ids_.clear();
    return;
  }
```

After `cache_bytes == 0` check:

```cpp
  if (cache_bytes == 0) {
    std::cerr << "[WARN] Cache has 0 loadable nodes; skipping cache\n";
    cache_ids_.clear();
    return;
  }
```

After `load_cache_nodes_standard` failure:

```cpp
  if (!load_cache_nodes_standard(cache_vectors_input, cache_bytes)) {
    std::cerr << "[WARN] Failed to read cache nodes from: " << cache_nodes_file << "\n";
    cache_ids_.clear();
    return;
  }
```

- [ ] **Step 3: Build and run tests**

Run: `.claude/scripts/run_and_log.sh test_logging make build && ./build/tests/index/laser_index_test`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add include/index/laser/quantized_graph.hpp
git commit -m "fix: add warning logs for silent cache and medoid load failures"
```

---

### Task 8: Extract drain loop helper from disk_search_qg

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:391-599`

The main search loop (lines ~564-579) and drain loop (lines ~582-596) contain duplicated "process prepared node with prefetch + free slot" logic.

- [ ] **Step 1: Add process_prepared_nodes helper to QuantizedGraph private section**

In the private method declarations (around line 188), add:

```cpp
  void process_prepared_nodes(size_t &need_process_num,
                              const std::function<void(PID, float *)> &process_node,
                              const std::function<void()> &wait_for_nodes,
                              LaserSearchContext &ctx,
                              size_t prefetch_lines);
```

- [ ] **Step 2: Implement the helper**

Add implementation after the `scan_neighbors` implementation:

```cpp
inline void QuantizedGraph::process_prepared_nodes(
    size_t &need_process_num,
    const std::function<void(PID, float *)> &process_node,
    const std::function<void()> &wait_for_nodes,
    LaserSearchContext &ctx,
    size_t prefetch_lines) {
  auto &prepared = ctx.prepared_ring();
  auto &free_slots = ctx.free_slot_stack();

  while (need_process_num > 0) {
    if (!prepared.empty()) {
      auto node = prepared.pop_front();
      if (!prepared.empty()) {
        auto &next = prepared.front();
        alaya::mem_prefetch_l1(next.second + offset_to_node(next.first), prefetch_lines);
      }
      process_node(node.first,
                   reinterpret_cast<float *>(node.second + offset_to_node(node.first)));
      --need_process_num;
      free_slots.push(node.second);
    } else {
      wait_for_nodes();
    }
  }
}
```

- [ ] **Step 3: Replace duplicated blocks in disk_search_qg**

Replace the pipelined processing block (lines ~564-579):

```cpp
    // Pipelined processing
    auto remain_num = static_cast<size_t>(0.5 * n_ops);
    size_t need_process_num = n_ops + previous_remain_num - remain_num;
    previous_remain_num = remain_num;

    process_prepared_nodes(need_process_num, process_node, wait_for_nodes, ctx, prefetch_lines);
  }

  // Drain remaining
  process_prepared_nodes(previous_remain_num, process_node, wait_for_nodes, ctx, prefetch_lines);
```

- [ ] **Step 4: Build and run tests**

Run: `.claude/scripts/run_and_log.sh test_drain_extract make build && ./build/tests/index/laser_index_test`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add include/index/laser/quantized_graph.hpp
git commit -m "refactor: extract drain loop helper to remove duplication in disk_search_qg"
```

---

### Task 9: Remove blanket NOLINTBEGIN and add specific suppressions

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:12,963`

- [ ] **Step 1: Remove NOLINTBEGIN/END**

Remove line 12 (`// NOLINTBEGIN`) and line 963 (`// NOLINTEND`).

- [ ] **Step 2: Run clang-tidy to find actual warnings**

Run: `.claude/scripts/run_and_log.sh clang_tidy_qg clang-tidy include/index/laser/quantized_graph.hpp -- -std=c++20 -I include -I build/_deps`
Note: exact flags may vary; use `make lint` if it runs clang-tidy, or check `.clang-tidy` config.

- [ ] **Step 3: Add specific NOLINT suppressions for legitimate exceptions**

For each warning, either fix it or add a targeted `// NOLINT(specific-check)` with a brief reason. Common expected suppressions:
- `reinterpret_cast` in AIO/node parsing: `// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)` - binary format requires it
- OpenMP pragmas may trigger misc warnings
- `std::getenv` in `check_omp_affinity`: `// NOLINT(concurrency-mt-unsafe)` - called once at init

- [ ] **Step 4: Run lint to verify**

Run: `.claude/scripts/run_and_log.sh lint_check make lint`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/index/laser/quantized_graph.hpp
git commit -m "style: replace blanket NOLINTBEGIN with targeted suppressions"
```

---

### Task 10: Document node layout formula

**Files:**
- Modify: `include/index/laser/quantized_graph.hpp:259-260,281-289`

- [ ] **Step 1: Add node layout documentation**

Above the constructor implementation (line ~251), add:

```cpp
// QuantizedGraph node layout (all offsets in float-count units):
//
//   [0 .. dimension_)                    main PCA-rotated vector    (32 bits/dim)
//   [dimension_ .. code_offset_)         residual vector            (32 bits/dim)
//   [code_offset_ .. factor_offset_)     RaBitQ packed codes        (padded_dim/64 * 2 * degree uint8s)
//   [factor_offset_ .. neighbor_offset_) correction factors         (3 floats per neighbor: triple_x, factor_dq, factor_vq)
//   [neighbor_offset_ .. row_offset_)    neighbor PIDs              (1 uint32 per neighbor)
//
// node_len_ in bytes = (32*main_dim + 32*residual_dim + 128*degree + degree*padded_dim) / 8
//                    = main_vec_bytes + residual_bytes + factor_bytes(12*deg) + code_bytes(padded*deg/8) + neighbor_bytes(4*deg)
//
// Nodes are packed into pages for Direct I/O. node_per_page_ = floor(kSectorLen / node_len_).
// page_size_ is rounded up to kSectorLen (4096) alignment for O_DIRECT.
```

- [ ] **Step 2: Add inline comments for offset fields in initialize()**

```cpp
inline void QuantizedGraph::initialize() {
  assert(padded_dim_ % 64 == 0);
  assert(padded_dim_ >= dimension_);

  res_dim_offset_ = dimension_;                                           // residual vector start (float units)
  code_offset_ = dimension_ + residual_dimension_;                        // RaBitQ codes start
  factor_offset_ = code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // correction factors start
  neighbor_offset_ = factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);  // neighbor PIDs start
  row_offset_ = neighbor_offset_ + degree_bound_;                         // end of node (float units)
}
```

- [ ] **Step 3: Build to verify**

Run: `.claude/scripts/run_and_log.sh build_docs make build`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add include/index/laser/quantized_graph.hpp
git commit -m "docs: document QuantizedGraph node layout formula and offset fields"
```

---

## Out of Scope (Future Work)

These issues were identified in review but require larger design changes:

1. **Search vs reconfiguration race condition** (`set_params`/`load_disk_index` vs concurrent `search`): Requires adding a reader-writer lock around the search path. Significant API change - should be a separate design task.

2. **VamanaFormatWriter extraction**: Moving to its own header is pure refactoring with no behavior change. Do separately.

3. **`cache_nodes_` alignment**: Change `std::vector<char>` to `std::vector<char, AlignedAlloc<char, 64>>` for formal correctness. Low practical risk - standard allocators already return aligned memory on Linux. Do separately.

4. **Checkpoint hash collision**: Include input path and output prefix in the hash. Needs `BuildState` API changes.

5. **PCA full-dataset shuffle**: Replace with `std::sample` or partial Fisher-Yates. Performance-only fix for very large datasets.

/**
 * @file laser_index_test.cpp
 * @brief Tests for Laser index types, search context, and system safety.
 *
 * Covers:
 * - OngoingSlot size assertion (Task 11.5)
 * - OngoingTable insert/find/remove/reset (Task 11.5)
 * - TaggedVisitedSet get/set/reset/overflow (Task 11.4)
 * - FixedRingBuffer / FixedStack behavior
 * - LaserSearchParams
 * - AIO quota pre-flight check (Task 11.6)
 * - LaserIndex load failure cases (Task 11.1)
 */

#include <gtest/gtest.h>

#include <cstring>

#include "index/laser/laser_index.hpp"
#include "index/laser/laser_types.hpp"
#include "index/laser/quantized_graph.hpp"
#include "index/laser/thread_data.hpp"

// ==========================================================================
// Task 11.5: OngoingSlot and OngoingTable tests
// ==========================================================================

TEST(LaserTypes, OngoingSlotSizeIs16) {
    EXPECT_EQ(sizeof(symqg::OngoingSlot), 16);
}

TEST(OngoingTableTest, InsertFindErase) {
    symqg::OngoingTable table(64);
    char buf1 = 'a';
    char buf2 = 'b';

    table.insert(42, &buf1);
    table.insert(99, &buf2);

    EXPECT_EQ(table.find(42), &buf1);
    EXPECT_EQ(table.find(99), &buf2);
    EXPECT_EQ(table.find(0), nullptr);

    table.erase(42);
    EXPECT_EQ(table.find(42), nullptr);
    EXPECT_EQ(table.find(99), &buf2);
}

TEST(OngoingTableTest, ResetInvalidatesEntries) {
    symqg::OngoingTable table(64);
    char buf = 'x';
    table.insert(10, &buf);
    EXPECT_NE(table.find(10), nullptr);

    table.reset();
    EXPECT_EQ(table.find(10), nullptr);
}

TEST(OngoingTableTest, CapacityBoundary) {
    // Small table, fill to capacity
    symqg::OngoingTable table(8);
    char bufs[8];
    for (uint32_t i = 0; i < 8; ++i) {
        table.insert(i, &bufs[i]);
    }
    for (uint32_t i = 0; i < 8; ++i) {
        EXPECT_EQ(table.find(i), &bufs[i]);
    }
}

// ==========================================================================
// Task 11.4: TaggedVisitedSet tests
// ==========================================================================

TEST(TaggedVisitedSetTest, SetAndGet) {
    symqg::TaggedVisitedSet vs(1024);
    EXPECT_FALSE(vs.get(42));

    vs.set(42);
    EXPECT_TRUE(vs.get(42));
    EXPECT_FALSE(vs.get(43));
}

TEST(TaggedVisitedSetTest, ResetO1) {
    symqg::TaggedVisitedSet vs(1024);
    vs.set(10);
    vs.set(20);
    vs.set(30);
    EXPECT_TRUE(vs.get(10));

    // O(1) reset via generation increment
    vs.reset();
    EXPECT_FALSE(vs.get(10));
    EXPECT_FALSE(vs.get(20));
    EXPECT_FALSE(vs.get(30));
}

TEST(TaggedVisitedSetTest, GenerationOverflow) {
    symqg::TaggedVisitedSet vs(64);
    vs.set(1);
    EXPECT_TRUE(vs.get(1));

    // Simulate many resets (gen overflow triggers full memset at gen==0)
    for (uint64_t i = 0; i < 100; ++i) {
        vs.reset();
    }
    // After many resets, previously set entries should be invalid
    EXPECT_FALSE(vs.get(1));

    // Should still work correctly after overflow
    vs.set(5);
    EXPECT_TRUE(vs.get(5));
}

TEST(TaggedVisitedSetTest, SetIdempotent) {
    symqg::TaggedVisitedSet vs(256);
    vs.set(42);
    vs.set(42);  // double-set should be idempotent
    EXPECT_TRUE(vs.get(42));
}

// ==========================================================================
// FixedRingBuffer tests
// ==========================================================================

TEST(FixedRingBufferTest, PushPopFIFO) {
    symqg::FixedRingBuffer<int> rb(8);
    rb.push_back(1);
    rb.push_back(2);
    rb.push_back(3);

    EXPECT_EQ(rb.size(), 3U);
    EXPECT_EQ(rb.pop_front(), 1);
    EXPECT_EQ(rb.pop_front(), 2);
    EXPECT_EQ(rb.pop_front(), 3);
    EXPECT_TRUE(rb.empty());
}

TEST(FixedRingBufferTest, ResetClearsState) {
    symqg::FixedRingBuffer<int> rb(4);
    rb.push_back(10);
    rb.push_back(20);
    rb.reset();
    EXPECT_TRUE(rb.empty());
    EXPECT_EQ(rb.size(), 0U);
}

// ==========================================================================
// FixedStack tests
// ==========================================================================

TEST(FixedStackTest, PushPopLIFO) {
    symqg::FixedStack<int> st(8);
    st.push(1);
    st.push(2);
    st.push(3);

    EXPECT_EQ(st.size(), 3U);
    EXPECT_EQ(st.pop(), 3);
    EXPECT_EQ(st.pop(), 2);
    EXPECT_EQ(st.pop(), 1);
    EXPECT_TRUE(st.empty());
}

// ==========================================================================
// LaserSearchParams tests
// ==========================================================================

TEST(LaserSearchParamsTest, DefaultAioEvents) {
    symqg::LaserSearchParams params;
    params.beam_width = 16;
    EXPECT_EQ(params.effective_aio_events(), 32U);  // 2 * beam_width

    params.aio_events_per_thread_ = 64;
    EXPECT_EQ(params.effective_aio_events(), 64U);  // explicit override
}

TEST(ThreadDataTest, AllocateVisitedSetCanTrackAllNodes) {
    constexpr size_t kNumPoints = 1000;

    symqg::ThreadData data;
    data.allocate(64, 32, 8, symqg::kSectorLen, 100, 128, kNumPoints);

    auto &visited = data.search_ctx_.visited_set();
    for (uint32_t id = 0; id < kNumPoints; ++id) {
        visited.set(id);
    }
    for (uint32_t id = 0; id < kNumPoints; ++id) {
        EXPECT_TRUE(visited.get(id)) << "missing visited id " << id;
    }

    data.deallocate();
}

// ==========================================================================
// Task 11.1: LaserIndex load error handling
// ==========================================================================

TEST(LaserIndexTest, SearchBeforeLoadThrows) {
    alaya::LaserIndex idx;
    EXPECT_FALSE(idx.is_loaded());

    float query[128] = {};
    uint32_t results[10] = {};
    EXPECT_THROW(idx.search(query, 10, results), std::runtime_error);
}

TEST(LaserIndexTest, LoadInvalidPathThrows) {
    alaya::LaserIndex idx;
    symqg::LaserSearchParams params;
    params.ef_search = 100;
    params.num_threads = 1;
    params.beam_width = 8;

    EXPECT_THROW(
        idx.load("/nonexistent/path/prefix", 1000, 32, 128, 128, params),
        std::runtime_error
    );
}

// ==========================================================================
// Task 11.6: AIO quota check (non-crashing validation)
// ==========================================================================

TEST(AioCheckTest, CheckAioCapacityDoesNotCrash) {
    // Just verify the function doesn't crash — actual warnings go to stderr
    symqg::check_aio_capacity(4, 32);
    symqg::check_aio_capacity(64, 128);
}

TEST(OmpAffinityTest, CheckOmpAffinityDoesNotCrash) {
    symqg::check_omp_affinity();
}

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

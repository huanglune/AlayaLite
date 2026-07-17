// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Unit tests for RowAdmission (segment admission contract v1):
// docs/design/segment-admission-contract.md. Pure kernel-level tests over
// the POD view + factories -- no QuantizedGraph/disk index needed.

#include "index/graph/laser/qg/row_admission.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace alaya::laser {
namespace {

TEST(RowAdmissionTest, DefaultConstructedAdmitsNothing) {
  const RowAdmission admission;
  EXPECT_FALSE(admission.test(0));
  EXPECT_FALSE(admission.test(63));
  EXPECT_EQ(admission.popcount, 0U);
  EXPECT_EQ(admission.capacity, 0U);
}

TEST(RowAdmissionTest, OutOfRangePidIsAlwaysFalse) {
  // All-ones payload: every in-range bit is 1, but pid >= capacity must
  // still report false regardless of the underlying bit content (defensive
  // bounds check protects against rows admitted after this snapshot).
  std::vector<uint64_t> words(2, ~uint64_t{0});
  const uint64_t capacity = 100;  // spans 2 words, partial second word
  const RowAdmission admission =
      admission_from_bitmap_payload(words.data(), words.size() * sizeof(uint64_t), capacity);
  EXPECT_TRUE(admission.test(0));
  EXPECT_TRUE(admission.test(99));
  EXPECT_FALSE(admission.test(100));
  EXPECT_FALSE(admission.test(127));
  EXPECT_FALSE(admission.test(1'000'000));
}

TEST(RowAdmissionTest, BitmapPayloadIsZeroCopyAndPopcountsCorrectly) {
  std::vector<uint64_t> words = {0b1010ULL, 0ULL};  // bits 1 and 3 set; capacity spans 2 words
  const uint64_t capacity = 128;
  const RowAdmission admission =
      admission_from_bitmap_payload(words.data(), words.size() * sizeof(uint64_t), capacity);
  EXPECT_EQ(admission.words, words.data()) << "must wrap the payload directly, no copy";
  EXPECT_FALSE(admission.test(0));
  EXPECT_TRUE(admission.test(1));
  EXPECT_FALSE(admission.test(2));
  EXPECT_TRUE(admission.test(3));
  for (uint64_t pid = 4; pid < 128; ++pid) {
    EXPECT_FALSE(admission.test(pid)) << pid;
  }
  EXPECT_EQ(admission.popcount, 2U);
}

TEST(RowAdmissionTest, BitmapPayloadPopcountClampsToShortBuffer) {
  // A short payload (fewer words than capacity implies) must not be read
  // out of bounds by the popcount scan; only the supplied words count.
  std::vector<uint64_t> words = {~uint64_t{0}};  // 64 bits set, but capacity asks for more
  const RowAdmission admission =
      admission_from_bitmap_payload(words.data(), words.size() * sizeof(uint64_t), /*capacity=*/256);
  EXPECT_EQ(admission.popcount, 64U);
}

TEST(RowAdmissionTest, SortedRowsMaterializesAndDedupsPopcount) {
  const std::vector<uint64_t> rows = {5, 5, 10, 63, 64, 200 /* out of range, dropped */};
  std::vector<uint64_t> storage;
  const uint64_t capacity = 100;
  const RowAdmission admission =
      admission_from_sorted_rows(rows.data(), rows.size(), capacity, storage);
  EXPECT_TRUE(admission.test(5));
  EXPECT_TRUE(admission.test(10));
  EXPECT_TRUE(admission.test(63));
  EXPECT_TRUE(admission.test(64));
  EXPECT_FALSE(admission.test(200)) << "row >= capacity must not be representable";
  EXPECT_FALSE(admission.test(0));
  EXPECT_FALSE(admission.test(11));
  EXPECT_EQ(admission.popcount, 4U) << "duplicate row 5 must not be double-counted";
  EXPECT_EQ(admission.words, storage.data());
}

TEST(RowAdmissionTest, SortedRowsEmptyCapacityIsSafe) {
  const std::vector<uint64_t> rows = {0, 1, 2};
  std::vector<uint64_t> storage;
  const RowAdmission admission =
      admission_from_sorted_rows(rows.data(), rows.size(), /*capacity=*/0, storage);
  EXPECT_EQ(admission.popcount, 0U);
  EXPECT_FALSE(admission.test(0));
}

TEST(RowAdmissionTest, ExcludeSetComplementsWithinCapacity) {
  std::unordered_set<uint32_t> excluded = {2, 5, 9};
  std::vector<uint64_t> storage;
  const uint64_t capacity = 10;
  const RowAdmission admission = admission_from_exclude_set(excluded, capacity, storage);
  for (uint64_t pid = 0; pid < capacity; ++pid) {
    const bool should_admit = excluded.find(static_cast<uint32_t>(pid)) == excluded.end();
    EXPECT_EQ(admission.test(pid), should_admit) << pid;
  }
  EXPECT_FALSE(admission.test(capacity)) << "tail padding bits must not leak past capacity";
  EXPECT_EQ(admission.popcount, capacity - excluded.size());
}

TEST(RowAdmissionTest, ExcludeSetOutOfRangeEntriesAreIgnored) {
  std::unordered_set<uint32_t> excluded = {3, 9999};  // 9999 is outside capacity
  std::vector<uint64_t> storage;
  const uint64_t capacity = 8;
  const RowAdmission admission = admission_from_exclude_set(excluded, capacity, storage);
  EXPECT_EQ(admission.popcount, capacity - 1);
  EXPECT_FALSE(admission.test(3));
  for (uint64_t pid = 0; pid < capacity; ++pid) {
    if (pid != 3) {
      EXPECT_TRUE(admission.test(pid)) << pid;
    }
  }
}

TEST(RowAdmissionTest, ExcludeSetTailBitsClearedWhenCapacityNotWordAligned) {
  // Empty exclude set over a non-word-aligned capacity: every real bit must
  // be set, but popcount must equal `capacity`, not a full word multiple --
  // proves the trailing padding bits were cleared, not left as garbage 1s.
  const std::unordered_set<uint32_t> excluded;
  std::vector<uint64_t> storage;
  const uint64_t capacity = 70;  // 2 words, 6 live bits in the second word
  const RowAdmission admission = admission_from_exclude_set(excluded, capacity, storage);
  EXPECT_EQ(admission.popcount, capacity);
  EXPECT_EQ(storage.size(), 2U);
}

TEST(RowAdmissionTest, AndComposesIntersectionAndMinCapacity) {
  std::vector<uint64_t> a_storage;
  std::vector<uint64_t> b_storage;
  const std::vector<uint64_t> a_rows = {1, 2, 3, 4};
  const std::vector<uint64_t> b_rows = {2, 4, 6};
  const RowAdmission a = admission_from_sorted_rows(a_rows.data(), a_rows.size(), 50, a_storage);
  const RowAdmission b = admission_from_sorted_rows(b_rows.data(), b_rows.size(), 40, b_storage);

  std::vector<uint64_t> and_storage;
  const RowAdmission combined = admission_and(a, b, and_storage);
  EXPECT_EQ(combined.capacity, 40U) << "capacity must be the min of the two sources";
  EXPECT_TRUE(combined.test(2));
  EXPECT_TRUE(combined.test(4));
  EXPECT_FALSE(combined.test(1));
  EXPECT_FALSE(combined.test(3));
  EXPECT_FALSE(combined.test(6)) << "row 6 is in b but not a";
  EXPECT_EQ(combined.popcount, 2U);
}

TEST(RowAdmissionTest, AndWithDisjointSourcesIsEmpty) {
  std::vector<uint64_t> a_storage;
  std::vector<uint64_t> b_storage;
  const std::vector<uint64_t> a_rows = {1, 2, 3};
  const std::vector<uint64_t> b_rows = {10, 11, 12};
  const RowAdmission a = admission_from_sorted_rows(a_rows.data(), a_rows.size(), 20, a_storage);
  const RowAdmission b = admission_from_sorted_rows(b_rows.data(), b_rows.size(), 20, b_storage);

  std::vector<uint64_t> and_storage;
  const RowAdmission combined = admission_and(a, b, and_storage);
  EXPECT_EQ(combined.popcount, 0U);
  for (uint64_t pid = 0; pid < 20; ++pid) {
    EXPECT_FALSE(combined.test(pid)) << pid;
  }
}

TEST(RowAdmissionTest, WordsForCapacityRoundsUp) {
  EXPECT_EQ(admission_words_for_capacity(0), 0U);
  EXPECT_EQ(admission_words_for_capacity(1), 1U);
  EXPECT_EQ(admission_words_for_capacity(64), 1U);
  EXPECT_EQ(admission_words_for_capacity(65), 2U);
  EXPECT_EQ(admission_words_for_capacity(128), 2U);
}

}  // namespace
}  // namespace alaya::laser

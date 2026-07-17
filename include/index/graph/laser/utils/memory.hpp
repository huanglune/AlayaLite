// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>

#include "platform/detect.hpp"
#include "utils/math.hpp"

// O_DIRECT scratch allocation: fixed Alignment regardless of size (unlike
// alaya::AlignedAlloc<T>, utils/memory.hpp, which picks 64B/2MB by size and
// so cannot guarantee kSectorLen for buffers under its 16KB huge-page
// threshold). W1 moved every other caller of this file to AlignedAlloc<T>;
// see REPORT-allocator-merge.md. align_allocate/align_free delegate to the
// same shared kernel (alaya_aligned_alloc_impl/_free_impl, platform/detect.hpp;
// math::round_up_pow2, utils/math.hpp) that AlignedAlloc itself uses, so
// there is exactly one madvise/round-up implementation left in the tree.
namespace alaya::laser::memory {

/** @brief Allocates zero-filled Alignment-aligned memory, optionally with huge page hints. */
template <size_t Alignment>
inline void *align_allocate(size_t nbytes, bool huge_page = false) {
  size_t size = math::round_up_pow2(nbytes, Alignment);
  void *ptr = alaya_aligned_alloc_impl(size, Alignment);
  assert(ptr != nullptr);
  if (huge_page) {
#ifdef ALAYA_OS_LINUX
    madvise(ptr, nbytes, MADV_HUGEPAGE);
#endif
  }
  std::memset(ptr, 0, size);
  return ptr;
}

inline void align_free(void *ptr) noexcept { alaya_aligned_free_impl(ptr); }

}  // namespace alaya::laser::memory

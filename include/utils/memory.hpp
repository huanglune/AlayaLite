/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>
#include "utils/math.hpp"
#include "utils/platform.hpp"

namespace alaya {

/// Alignment constant: auto-select based on size
constexpr size_t kAlignAuto = 0;
/// Alignment constant: 64 bytes (cache line)
constexpr size_t kAlign64B = 64;
/// Alignment constant: 4KB (sector/page size)
constexpr size_t kAlign4K = 4 * 1024;
/// Alignment constant: 2MB (huge page)
constexpr size_t kAlign2M = 2 * 1024 * 1024;

/**
 * @brief Aligned memory allocator compatible with STL containers.
 *
 * @tparam T Element type
 * @tparam Alignment Alignment requirement. Use kAlignAuto (0) for automatic
 *         selection based on size: 64B for < 16KB, 2MB for >= 16KB.
 */
template <typename T, size_t Alignment = kAlignAuto, bool HugePage = false>
class AlignedAlloc {
 private:
  static constexpr size_t kAlignSmall = 64;
  static constexpr size_t kAlignLarge = 2 * 1024 * 1024;   // 2MB
  static constexpr size_t kHugePageThreshold = 16 * 1024;  // 16KB

  static constexpr auto get_alignment(size_t raw_size) -> size_t {
    if constexpr (HugePage) {
      return (Alignment != kAlignAuto) ? Alignment : kAlignSmall;
    } else if constexpr (Alignment != kAlignAuto) {
      return Alignment;
    } else {
      return (raw_size >= kHugePageThreshold) ? kAlignLarge : kAlignSmall;
    }
  }

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  template <class U>
  struct rebind {  // NOLINT
    using other = AlignedAlloc<U, Alignment, HugePage>;
  };

  constexpr AlignedAlloc() noexcept = default;
  constexpr AlignedAlloc(const AlignedAlloc &) noexcept = default;

  template <typename U, size_t OtherAlign, bool OtherHP>
  constexpr explicit AlignedAlloc(
      const AlignedAlloc<U, OtherAlign, OtherHP> & /*unused*/) noexcept {}

  /**
   * @brief Allocate aligned memory
   *
   * @param n Number of elements to allocate
   * @return Pointer to allocated memory
   * @throws std::bad_alloc on allocation failure
   * @throws std::bad_array_new_length if n is too large
   */
  [[nodiscard]] auto allocate(std::size_t n) -> T * {
    if (n == 0) {
      return nullptr;
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    size_t raw_size = n * sizeof(T);
    size_t align = get_alignment(raw_size);
    bool use_huge_page = HugePage || (align == kAlignLarge);

    const auto kNbytes = math::round_up_pow2(raw_size, align);
    void *ptr = alaya_aligned_alloc_impl(kNbytes, align);
    if (ALAYA_UNLIKELY(!ptr)) {
      throw std::bad_alloc();
    }

#ifdef ALAYA_OS_LINUX
    if (use_huge_page) {
      madvise(ptr, kNbytes, MADV_HUGEPAGE);
    }
#endif
    return reinterpret_cast<T *>(ptr);
  }

  /**
   * @brief Deallocate aligned memory
   * @param ptr Pointer to memory to deallocate
   * @param n Number of elements (not used)
   */
  void deallocate(T *ptr, [[maybe_unused]] std::size_t n) noexcept {
    if (ptr) {
      alaya_aligned_free_impl(ptr);
    }
  }
};

template <typename T, size_t A1, bool HP1, typename U, size_t A2, bool HP2>
auto operator==(const AlignedAlloc<T, A1, HP1> & /*unused*/,
                const AlignedAlloc<U, A2, HP2> & /*unused*/) noexcept -> bool {
  return A1 == A2 && HP1 == HP2;
}

template <typename T, size_t A1, bool HP1, typename U, size_t A2, bool HP2>
auto operator!=(const AlignedAlloc<T, A1, HP1> & /*unused*/,
                const AlignedAlloc<U, A2, HP2> & /*unused*/) noexcept -> bool {
  return !(A1 == A2 && HP1 == HP2);
}

// =====================================================================
// Default-init allocator (skips value-initialization for trivial types)
// =====================================================================

/**
 * @brief STL-compatible allocator that default-initializes instead of
 *        value-initializing. For trivial types this means no zeroing.
 */
template <typename T>
struct DefaultInitAlloc {
  using value_type = T;

  constexpr DefaultInitAlloc() noexcept = default;

  template <typename U>
  explicit constexpr DefaultInitAlloc(const DefaultInitAlloc<U> & /*unused*/) noexcept {}

  [[nodiscard]] constexpr auto allocate(std::size_t n) -> T * {
    return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
  }

  constexpr void deallocate(T *ptr, size_t count) noexcept { ::operator delete(ptr, count); }

  template <typename U>
  void construct(U *ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void *>(ptr)) U;
  }
};

// =====================================================================
// Aligned buffer types
// =====================================================================

/**
 * @brief Type alias template for aligned byte buffer.
 *
 * @tparam Alignment Alignment requirement
 */
template <size_t Alignment>
using AlignedBufferT = std::vector<uint8_t, AlignedAlloc<uint8_t, Alignment>>;

/// 4KB aligned buffer (default, for Direct IO)
using AlignedBuffer = AlignedBufferT<kAlign4K>;
/// 64-byte aligned buffer (cache line)
using AlignedBuffer64B = AlignedBufferT<kAlign64B>;
/// 2MB aligned buffer (huge page)
using AlignedBuffer2M = AlignedBufferT<kAlign2M>;

inline auto make_aligned_buffer(size_t size, size_t alignment = kAlign4K) -> AlignedBuffer {
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
  return AlignedBuffer(aligned_size);
}

}  // namespace alaya

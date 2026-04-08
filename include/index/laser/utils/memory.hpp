/**
 * @file memory.hpp
 * @brief Memory allocation utilities with alignment and huge page support.
 *
 * Provides:
 * - AlignedAllocator: STL-compatible allocator with configurable alignment
 * - align_allocate(): Raw aligned memory allocation
 * - prefetch utilities: L1/L2 cache prefetch hints
 *
 * Alignment is critical for SIMD operations (AVX2 requires 32-byte, AVX512 requires 64-byte).
 */

#pragma once

#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>

#include "index/laser/utils/tools.hpp"

namespace symqg::memory {
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

/**
 * @brief STL-compatible allocator with configurable alignment and huge page support.
 * @tparam T Element type
 * @tparam Alignment Byte alignment (default 64 for cache line)
 * @tparam HugePage Enable huge page allocation via madvise
 */
template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
 private:
  static_assert(Alignment >= alignof(T));

 public:
  using value_type = T;

  template <class U>
  struct rebind {  // NOLINT(readability-identifier-naming)
    using other = AlignedAllocator<U, Alignment>;
  };

  constexpr AlignedAllocator() noexcept = default;

  constexpr AlignedAllocator(const AlignedAllocator &) noexcept = default;

  template <typename U>
  constexpr explicit AlignedAllocator(AlignedAllocator<U, Alignment> const & /*unused*/) noexcept {}

  [[nodiscard]] auto allocate(std::size_t n) -> T * {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    auto nbytes = round_up_to_multiple(n * sizeof(T), Alignment);
    auto *ptr = ::operator new[](nbytes, std::align_val_t(Alignment));
    if (HugePage) {
      madvise(ptr, nbytes, MADV_HUGEPAGE);
    }
    return reinterpret_cast<T *>(ptr);
  }

  void deallocate(T *ptr, [[maybe_unused]] std::size_t bytes) {
    ::operator delete[](ptr, std::align_val_t(Alignment));
  }
};

template <typename T>
struct Allocator {
 public:
  using value_type = T;

  constexpr Allocator() noexcept = default;

  template <typename U>
  explicit constexpr Allocator(const Allocator<U> & /*unused*/) noexcept {}

  [[nodiscard]] constexpr auto allocate(std::size_t n) -> T * {
    return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
  }

  constexpr void deallocate(T *ptr, size_t count) noexcept { ::operator delete(ptr, count); }

  // Intercept zero-argument construction to do default initialization.
  template <typename U>
  void construct(U *ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void *>(ptr)) U;
  }
};

/** @brief Allocates aligned memory, optionally with huge page hints. */
template <size_t Alignment>
inline auto align_allocate(size_t nbytes, bool huge_page = false) -> void * {
  size_t size = round_up_to_multiple(nbytes, Alignment);
  void *ptr = std::aligned_alloc(Alignment, size);
  if (huge_page) {
    madvise(ptr, nbytes, MADV_HUGEPAGE);
  }
  std::memset(ptr, 0, size);
  assert(ptr != nullptr);
  return ptr;
}

static inline void prefetch_l1(const void *addr) {
#if defined(__SSE2__)
  _mm_prefetch(addr, _MM_HINT_T0);
#else
  __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void *addr) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)addr, _MM_HINT_T1);
#else
  __builtin_prefetch(addr, 0, 2);
#endif
}

inline auto mem_prefetch_l1(const char *ptr, size_t num_lines) -> void {
  size_t count = std::min(num_lines, static_cast<size_t>(20));
  const char *cursor = ptr;
  for (size_t i = 0; i < count; ++i) {
    prefetch_l1(cursor);
    cursor += 64;
  }
}

inline auto mem_prefetch_l2(const char *ptr, size_t num_lines) -> void {
  size_t count = std::min(num_lines, static_cast<size_t>(20));
  const char *cursor = ptr;
  for (size_t i = 0; i < count; ++i) {
    prefetch_l2(cursor);
    cursor += 64;
  }
}
}  // namespace symqg::memory

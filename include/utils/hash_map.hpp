/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hash_map.hpp
 * @brief High-performance hash map wrapper with automatic fallback.
 *
 * This header provides type aliases for hash map and hash set that use
 * ankerl::unordered_dense when available (via Conan), falling back to
 * std::unordered_map/set otherwise.
 *
 * @section Performance
 *
 * ankerl::unordered_dense provides significant performance improvements:
 * - 2-3x faster lookups than std::unordered_map
 * - Better cache locality (flat storage)
 * - Lower memory overhead
 *
 * @section Usage
 *
 * @code
 *   #include "utils/hash_map.hpp"
 *
 *   // Use like std::unordered_map
 *   alaya::fast::map<int, std::string> my_map;
 *   my_map[1] = "one";
 *   my_map.insert({2, "two"});
 *
 *   // Hash set
 *   alaya::fast::set<int> my_set;
 *   my_set.insert(42);
 *
 *   // Check which implementation is being used
 *   if (alaya::fast::is_dense()) {
 *     // Using ankerl::unordered_dense
 *   }
 * @endcode
 */

#pragma once

// Detect and include high-performance hash map if available
#ifndef ALAYA_USE_UNORDERED_DENSE
  #if __has_include(<ankerl/unordered_dense.h>)
    #include <ankerl/unordered_dense.h>
    #define ALAYA_USE_UNORDERED_DENSE 1
  #else
    #include <unordered_map>
    #include <unordered_set>
    #define ALAYA_USE_UNORDERED_DENSE 0
  #endif
#endif  // ALAYA_USE_UNORDERED_DENSE

namespace alaya::fast {

/**
 * @brief High-performance hash map.
 *
 * Uses ankerl::unordered_dense::map when available, otherwise std::unordered_map.
 *
 * @tparam Key Key type
 * @tparam Value Value type
 * @tparam Hash Hash function (default: std::hash<Key>)
 * @tparam KeyEqual Equality function (default: std::equal_to<Key>)
 */
template <typename Key,
          typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
#if ALAYA_USE_UNORDERED_DENSE
using map = ankerl::unordered_dense::map<Key, Value, Hash, KeyEqual>;
#else
using map = std::unordered_map<Key, Value, Hash, KeyEqual>;
#endif

/**
 * @brief High-performance hash set.
 *
 * Uses ankerl::unordered_dense::set when available, otherwise std::unordered_set.
 *
 * @tparam Key Key type
 * @tparam Hash Hash function (default: std::hash<Key>)
 * @tparam KeyEqual Equality function (default: std::equal_to<Key>)
 */
template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
#if ALAYA_USE_UNORDERED_DENSE
using set = ankerl::unordered_dense::set<Key, Hash, KeyEqual>;
#else
using set = std::unordered_set<Key, Hash, KeyEqual>;
#endif

/**
 * @brief Check if high-performance hash map is available.
 * @return true if using ankerl::unordered_dense, false if using std::unordered_map
 */
constexpr auto is_dense() -> bool {
#if ALAYA_USE_UNORDERED_DENSE
  return true;
#else
  return false;
#endif
}

}  // namespace alaya::fast

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

#pragma once

#include <concepts>
#include <cstddef>
#include <optional>

namespace alaya {

/**
 * @brief Concept for page replacement algorithms.
 *
 * A Replacer manages which frames are candidates for eviction.
 * Different implementations provide different eviction policies
 * (e.g., LRU, CLOCK, LFU).
 *
 * Required operations:
 * - pin(frame_id): Mark frame as not evictable (being used)
 * - unpin(frame_id): Mark frame as evictable (can be evicted)
 * - evict(): Select and remove a victim frame
 * - remove(frame_id): Remove a specific frame
 * - size(): Get number of evictable frames
 * - reset(): Clear all state
 */
template <typename T>
concept ReplacerStrategy = requires(T r, size_t frame_id) {
  { r.pin(frame_id) } -> std::same_as<void>;
  { r.unpin(frame_id) } -> std::same_as<void>;
  { r.evict() } -> std::same_as<std::optional<size_t>>;
  { r.remove(frame_id) } -> std::same_as<void>;
  { r.size() } -> std::same_as<size_t>;
  { r.reset() } -> std::same_as<void>;
};

}  // namespace alaya

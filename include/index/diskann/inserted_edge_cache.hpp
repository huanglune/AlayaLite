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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace alaya {

/**
 * @brief Sharded LRU cache for reverse-edge hints emitted during insert.
 *
 * Callers are responsible for synchronizing access per shard.
 */
template <typename IDType = uint32_t>
class InsertedEdgeCache {
 public:
  static constexpr size_t kDefaultNumShards = 1024;
  static constexpr size_t kMaxEntriesPerKey = 8;

  InsertedEdgeCache() : InsertedEdgeCache(kDefaultNumShards, 1) {}

  InsertedEdgeCache(size_t num_shards, size_t max_keys_per_shard)
      : num_shards_(std::max(num_shards, size_t{1})),
        max_keys_per_shard_(std::max(max_keys_per_shard, size_t{1})),
        shards_(num_shards_) {}

  static auto with_index_capacity(uint32_t index_capacity, size_t num_shards = kDefaultNumShards)
      -> InsertedEdgeCache {
    auto shard_count = std::max(num_shards, size_t{1});
    auto per_shard_capacity =
        std::max(size_t{1}, (static_cast<size_t>(index_capacity) + shard_count - 1) / shard_count);
    return InsertedEdgeCache(shard_count, per_shard_capacity);
  }

  void add(size_t shard, IDType target_id, IDType source_id) {
    auto &state = shard_state(shard);
    auto existing = state.map_.find(target_id);
    if (existing != state.map_.end()) {
      existing->second->sources_.push_back(source_id);
      state.lru_list_.splice(state.lru_list_.begin(), state.lru_list_, existing->second);
      return;
    }

    if (state.map_.size() >= max_keys_per_shard_) {
      auto &lru_entry = state.lru_list_.back();
      state.map_.erase(lru_entry.target_id_);
      state.lru_list_.pop_back();
    }

    state.lru_list_.emplace_front(Entry{target_id, {}});
    auto entry = state.lru_list_.begin();
    entry->sources_.push_back(source_id);
    state.map_.emplace(target_id, entry);
  }

  auto consume(size_t shard, IDType target_id) -> std::vector<IDType> {
    auto &state = shard_state(shard);
    auto existing = state.map_.find(target_id);
    if (existing == state.map_.end()) {
      return {};
    }

    auto result = existing->second->sources_.to_vector();
    state.lru_list_.erase(existing->second);
    state.map_.erase(existing);
    if (state.map_.empty()) {
      state.map_ = Map{};
    }
    return result;
  }

  [[nodiscard]] auto size() const -> size_t {
    size_t total = 0;
    for (const auto &shard : shards_) {
      total += shard.map_.size();
    }
    return total;
  }

  [[nodiscard]] auto capacity() const -> size_t { return num_shards_ * max_keys_per_shard_; }
  [[nodiscard]] auto num_shards() const -> size_t { return num_shards_; }
  [[nodiscard]] auto shard_capacity() const -> size_t { return max_keys_per_shard_; }

  [[nodiscard]] auto bucket_count(size_t shard) const -> size_t {
    return shard_state(shard).map_.bucket_count();
  }

 private:
  struct SourceEntries {
    std::array<IDType, kMaxEntriesPerKey> values_{};
    size_t count_{0};

    void push_back(IDType source_id) {
      if (count_ < kMaxEntriesPerKey) {
        values_[count_++] = source_id;
        return;
      }

      std::move(values_.begin() + 1, values_.end(), values_.begin());
      values_[kMaxEntriesPerKey - 1] = source_id;
    }

    [[nodiscard]] auto to_vector() const -> std::vector<IDType> {
      return std::vector<IDType>(values_.begin(), values_.begin() + count_);
    }
  };

  struct Entry {
    IDType target_id_;
    SourceEntries sources_;
  };

  using LruList = std::list<Entry>;
  using Map = std::unordered_map<IDType, typename LruList::iterator>;

  struct ShardState {
    LruList lru_list_;
    Map map_;
  };

  [[nodiscard]] auto shard_state(size_t shard) -> ShardState & { return shards_.at(shard); }
  [[nodiscard]] auto shard_state(size_t shard) const -> const ShardState & {
    return shards_.at(shard);
  }

  size_t num_shards_;
  size_t max_keys_per_shard_;
  std::vector<ShardState> shards_;
};

}  // namespace alaya

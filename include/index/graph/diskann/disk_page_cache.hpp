// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace alaya::diskann {

class DiskPageCache {
 public:
  explicit DiskPageCache(size_t capacity) : capacity_(capacity) {}

  [[nodiscard]] bool enabled() const { return capacity_ > 0; }

  bool read(uint64_t page_off, char *out, size_t page_size) {
    if (!enabled()) {
      return false;
    }
    auto it = pages_.find(page_off);
    if (it == pages_.end()) {
      return false;
    }
    std::memcpy(out, it->second.bytes.data(), page_size);
    touch(it);
    return true;
  }

  template <typename FlushPage>
  void write(uint64_t page_off, const char *page, size_t page_size, bool dirty, FlushPage &&flush) {
    if (!enabled()) {
      return;
    }
    auto it = pages_.find(page_off);
    if (it != pages_.end()) {
      it->second.bytes.assign(page, page + page_size);
      it->second.dirty = it->second.dirty || dirty;
      touch(it);
      return;
    }
    if (pages_.size() >= capacity_) {
      // At capacity (the steady state once the pool is warm): recycle the LRU
      // victim's map node, byte buffer and LRU node instead of erase+alloc —
      // a full pool would otherwise pay one 4KB heap alloc + free per fill,
      // which under concurrent shard traffic dominates the search fill path.
      evict_excess(std::forward<FlushPage>(flush));  // only if capacity shrank below size
      const uint64_t victim_off = lru_.back();
      auto vit = pages_.find(victim_off);
      if (vit == pages_.end()) {  // defensive: lru_/pages_ out of sync
        lru_.pop_back();
        lru_.push_front(page_off);
        pages_.emplace(page_off,
                       Entry{std::vector<char>(page, page + page_size), dirty, lru_.begin()});
        return;
      }
      if (vit->second.dirty) {
        flush(victim_off, vit->second.bytes.data());
      }
      auto node = pages_.extract(vit);
      node.key() = page_off;
      Entry &entry = node.mapped();
      entry.bytes.assign(page, page + page_size);  // same 4KB buffer, no realloc
      entry.dirty = dirty;
      lru_.back() = page_off;
      lru_.splice(lru_.begin(), lru_, std::prev(lru_.end()));
      entry.lru_it = lru_.begin();
      pages_.insert(std::move(node));
      return;
    }
    lru_.push_front(page_off);
    pages_.emplace(page_off, Entry{std::vector<char>(page, page + page_size), dirty, lru_.begin()});
  }

  template <typename FlushPage>
  void flush_dirty(FlushPage &&flush) {
    for (auto &page : pages_) {
      if (!page.second.dirty) {
        continue;
      }
      flush(page.first, page.second.bytes.data());
      page.second.dirty = false;
    }
  }

 private:
  using LruList = std::list<uint64_t>;

  struct Entry {
    std::vector<char> bytes;
    bool dirty = false;
    LruList::iterator lru_it;
  };

  using PageMap = std::unordered_map<uint64_t, Entry>;

  void touch(PageMap::iterator it) {
    lru_.splice(lru_.begin(), lru_, it->second.lru_it);
    it->second.lru_it = lru_.begin();
  }

  /// Drop entries until size() == capacity_ (no-op in the normal at-capacity
  /// case, which recycles the victim in write() instead of erasing it).
  template <typename FlushPage>
  void evict_excess(FlushPage &&flush) {
    while (pages_.size() > capacity_) {
      const uint64_t page_off = lru_.back();
      auto it = pages_.find(page_off);
      if (it != pages_.end() && it->second.dirty) {
        flush(page_off, it->second.bytes.data());
      }
      lru_.pop_back();
      pages_.erase(page_off);
    }
  }

  size_t capacity_ = 0;
  LruList lru_;
  PageMap pages_;
};

}  // namespace alaya::diskann

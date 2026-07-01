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
    evict_until_room(std::forward<FlushPage>(flush));
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

  template <typename FlushPage>
  void evict_until_room(FlushPage &&flush) {
    while (pages_.size() >= capacity_) {
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

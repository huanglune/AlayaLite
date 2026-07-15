// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <unistd.h>

#include "test_config.hpp"

namespace alaya::test {

inline auto data_dir() -> std::filesystem::path { return kDataDir; }
inline auto source_dir() -> std::filesystem::path { return kSourceDir; }
inline auto build_dir() -> std::filesystem::path { return kBuildDir; }

inline auto tmp_root() -> std::filesystem::path {
  return std::filesystem::temp_directory_path();
}

class ScopedTempDir {
 public:
  explicit ScopedTempDir(std::string_view prefix = "alaya-test") {
    static std::uint64_t serial{};
    path_ = tmp_root() / (std::string(prefix) + "-" +
                          std::to_string(::getpid()) + "-" +
                          std::to_string(++serial));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  ScopedTempDir(const ScopedTempDir &) = delete;
  auto operator=(const ScopedTempDir &) -> ScopedTempDir & = delete;
  ScopedTempDir(ScopedTempDir &&) = delete;
  auto operator=(ScopedTempDir &&) -> ScopedTempDir & = delete;

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace alaya::test

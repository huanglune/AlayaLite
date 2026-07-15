// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <unistd.h>

namespace alaya::test {

inline auto data_dir() -> std::filesystem::path {
  if (const auto *env = std::getenv("ALAYA_TEST_DATA_DIR"); env && *env) {
    return env;
  }
#ifdef ALAYA_TEST_DATA_DIR_DEFAULT
  return ALAYA_TEST_DATA_DIR_DEFAULT;
#else
  return std::filesystem::current_path().parent_path() / "data";
#endif
}

inline auto tmp_root() -> std::filesystem::path {
  if (const auto *env = std::getenv("ALAYA_TEST_TMP_DIR"); env && *env) {
    return env;
  }
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

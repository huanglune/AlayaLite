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
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <mutex>
#include <string>

/**
 * @brief Extract relative path from full path.
 *
 * Looks for common source directories (include/, src/, tests/) in the path
 * and returns the path starting from there. This ensures consistent log output
 * regardless of the build directory location.
 *
 * Examples:
 *   /tmp/build-xxx/alayalite/include/utils/log.hpp -> include/utils/log.hpp
 *   /home/user/project/src/main.cpp -> src/main.cpp
 */
inline auto extract_relative_path(const char *full_path) -> const char * {
  // Look for common source directory markers
  const char *pos = std::strstr(full_path, "/include/");
  if (pos != nullptr) {
    return pos + 1;
  }
  pos = std::strstr(full_path, "/src/");
  if (pos != nullptr) {
    return pos + 1;
  }
  pos = std::strstr(full_path, "/tests/");
  if (pos != nullptr) {
    return pos + 1;
  }
  // Fallback: return just the filename
  const char *last_slash = std::strrchr(full_path, '/');
  return last_slash != nullptr ? last_slash + 1 : full_path;
}

#define RELATIVE_FILE extract_relative_path(__FILE__)

/// @brief Lazy-initialize spdlog with a rotating file sink.
///
/// Called automatically by every LOG_* macro. Uses std::call_once so that
/// after the first invocation the cost is a single relaxed atomic load.
/// The log file path is {ALAYA_LOG_DIR}/alaya.log (default: ./alaya.log).
inline void init_alaya_logger() {
  static constexpr size_t kMaxFileSize = 10 * 1024 * 1024;  // 10 MB
  static constexpr size_t kMaxFiles = 3;

  static std::once_flag flag;
  std::call_once(flag, [] {
    std::string log_dir = ".";
    const char *env = std::getenv("ALAYA_LOG_DIR");
    if (env != nullptr && env[0] != '\0') {
      log_dir = env;
      std::filesystem::create_directories(log_dir);
    }
    std::string log_path = log_dir + "/alaya.log";

    auto sink =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_path, kMaxFileSize, kMaxFiles);
    auto logger = std::make_shared<spdlog::logger>("alaya", sink);
    logger->set_level(spdlog::level::trace);
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);
    spdlog::flush_every(std::chrono::seconds(3));
  });
}

/// @brief Check if ALAYA_QUIET=1 is set (cached once per process).
inline auto is_quiet() -> bool {
  static const bool kQuiet = [] {
    const char *env = std::getenv("ALAYA_QUIET");
    return env != nullptr && std::string(env) == "1";
  }();
  return kQuiet;
}

#define CONCATENATE_STRINGS(a, b) a b
#define LOG_TRACE(msg, ...)                                                     \
  {                                                                             \
    init_alaya_logger();                                                        \
    spdlog::trace(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                  RELATIVE_FILE,                                                \
                  __LINE__,                                                     \
                  ##__VA_ARGS__);                                               \
  }
#define LOG_DEBUG(msg, ...)                                                     \
  {                                                                             \
    init_alaya_logger();                                                        \
    spdlog::debug(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                  RELATIVE_FILE,                                                \
                  __LINE__,                                                     \
                  ##__VA_ARGS__);                                               \
  }
#define LOG_INFO(msg, ...)                                                     \
  {                                                                            \
    init_alaya_logger();                                                       \
    spdlog::info(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                 RELATIVE_FILE,                                                \
                 __LINE__,                                                     \
                 ##__VA_ARGS__);                                               \
  }
#define LOG_WARN(msg, ...)                                                     \
  {                                                                            \
    init_alaya_logger();                                                       \
    spdlog::warn(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                 RELATIVE_FILE,                                                \
                 __LINE__,                                                     \
                 ##__VA_ARGS__);                                               \
  }
#define LOG_ERROR(msg, ...)                                                     \
  {                                                                             \
    init_alaya_logger();                                                        \
    spdlog::error(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                  RELATIVE_FILE,                                                \
                  __LINE__,                                                     \
                  ##__VA_ARGS__);                                               \
  }
#define LOG_CRITICAL(msg, ...)                                                     \
  {                                                                                \
    init_alaya_logger();                                                           \
    spdlog::critical(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                     RELATIVE_FILE,                                                \
                     __LINE__,                                                     \
                     ##__VA_ARGS__);                                               \
  }

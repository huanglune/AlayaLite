// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#if !defined(_WIN32)
  #include <poll.h>
  #include <signal.h>
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

#include "alaya/collection.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 128;
constexpr std::uint64_t kRows = 8;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::atomic_uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-active-laser-recovery-" + std::string(name) + "-" +
             std::to_string(platform::get_pid()) + "-" + std::to_string(++serial));
    std::filesystem::remove_all(path_);
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

[[nodiscard]] auto options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions result;
  result.root = root;
  result.dim = kDim;
  result.metric = core::Metric::l2;
  result.scalar_type = core::ScalarType::float32;
  result.target_algorithm = core::algorithm::laser;
  result.active_engine = core::algorithm::laser;
  result.quantization = CollectionQuantization::rabitq;
  result.max_neighbors = 32;
  result.ef_construction = 128;
  result.build_threads = 1;
  return result;
}

[[nodiscard]] auto logical_id(std::uint64_t row) -> core::LogicalId {
  return core::LogicalId::from_utf8("b09-row-" + std::to_string(row));
}

void create_populated_collection(const std::filesystem::path &root) {
  auto created = Collection::create(options(root));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  for (std::uint64_t row = 0; row < kRows; ++row) {
    std::array<float, kDim> vector{};
    vector.fill(static_cast<float>(row) * 0.25F);
    CollectionItem item;
    item.logical_id = logical_id(row);
    item.vector = core::TypedTensorView::contiguous(vector.data(), 1, vector.size());
    ASSERT_TRUE(collection->add(item).ok());
  }
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
}

[[nodiscard]] auto source_directory(const std::filesystem::path &root)
    -> std::filesystem::path {
  return root / ".alaya_internal" / "active_laser" / "seg_00000002_g1";
}

[[nodiscard]] auto active_directory_count(const std::filesystem::path &root) -> size_t {
  const auto active_root = root / ".alaya_internal" / "active_laser";
  std::error_code error;
  size_t count = 0;
  if (!std::filesystem::is_directory(active_root, error)) {
    return count;
  }
  for (const auto &entry : std::filesystem::directory_iterator(active_root, error)) {
    if (entry.is_directory(error)) {
      ++count;
    }
  }
  return count;
}

void verify_rows(Collection &collection) {
  EXPECT_EQ(collection.size(), kRows);
  for (std::uint64_t row = 0; row < kRows; ++row) {
    auto record = collection.get_by_id(logical_id(row));
    EXPECT_TRUE(record.ok()) << "missing row " << row << ": " << record.status().diagnostic();
  }
}

#if !defined(_WIN32)
struct CrashResult {
  bool notified{};
  int status{};
};

void notify_parent_and_wait(int fd) {
  const char marker = 'R';
  ssize_t written;
  do {
    written = ::write(fd, &marker, sizeof(marker));
  } while (written < 0 && errno == EINTR);
  if (written != sizeof(marker)) {
    ::_exit(90);
  }
  for (;;) {
    ::pause();
  }
}

[[nodiscard]] auto kill_after_seal_failpoint(const std::filesystem::path &root,
                                             CollectionSealFailPoint failpoint) -> CrashResult {
  int ready[2]{};
  if (::pipe(ready) != 0) {
    return {};
  }
  const pid_t child = ::fork();
  if (child == 0) {
    ::close(ready[0]);
    auto opened = Collection::open(root);
    if (!opened.ok()) {
      ::_exit(80);
    }
    CollectionSealOptions seal_options;
    seal_options.fail_point = failpoint;
    seal_options.failpoint_hook = [failpoint, fd = ready[1]](CollectionSealFailPoint observed) {
      if (observed == failpoint) {
        notify_parent_and_wait(fd);
      }
    };
    (void)opened.value()->seal(std::move(seal_options));
    ::_exit(81);
  }
  ::close(ready[1]);
  if (child < 0) {
    ::close(ready[0]);
    return {};
  }

  pollfd event{ready[0], POLLIN, 0};
  const int polled = ::poll(&event, 1, 15000);
  char marker{};
  const bool notified = polled == 1 && (event.revents & POLLIN) != 0 &&
                        ::read(ready[0], &marker, sizeof(marker)) == sizeof(marker) && marker == 'R';
  ::close(ready[0]);
  (void)::kill(child, SIGKILL);
  int status{};
  (void)::waitpid(child, &status, 0);
  return {notified, status};
}

[[nodiscard]] auto kill_after_open(const std::filesystem::path &root) -> CrashResult {
  int ready[2]{};
  if (::pipe(ready) != 0) {
    return {};
  }
  const pid_t child = ::fork();
  if (child == 0) {
    ::close(ready[0]);
    auto opened = Collection::open(root);
    if (!opened.ok()) {
      ::_exit(82);
    }
    notify_parent_and_wait(ready[1]);
  }
  ::close(ready[1]);
  if (child < 0) {
    ::close(ready[0]);
    return {};
  }

  pollfd event{ready[0], POLLIN, 0};
  const int polled = ::poll(&event, 1, 15000);
  char marker{};
  const bool notified = polled == 1 && (event.revents & POLLIN) != 0 &&
                        ::read(ready[0], &marker, sizeof(marker)) == sizeof(marker) && marker == 'R';
  ::close(ready[0]);
  (void)::kill(child, SIGKILL);
  int status{};
  (void)::waitpid(child, &status, 0);
  return {notified, status};
}

void assert_true_sigkill(const CrashResult &crash) {
  ASSERT_TRUE(crash.notified) << "child did not reach the requested durable cut";
  ASSERT_TRUE(WIFSIGNALED(crash.status)) << "child status=" << crash.status;
  EXPECT_EQ(WTERMSIG(crash.status), SIGKILL);
}

void finish_recovery_and_verify(const std::filesystem::path &root) {
  const auto source = source_directory(root);
  ASSERT_TRUE(std::filesystem::is_directory(source));
  auto reopened = Collection::open(root);
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  auto collection = std::move(reopened).value();
  EXPECT_TRUE(std::filesystem::is_directory(source))
      << "the sweep must retain a source referenced by the pre-recovery control state";
  verify_rows(*collection);
  if (collection->stats().sealed_segments_count == 0) {
    auto sealed = collection->seal();
    ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  }
  EXPECT_EQ(collection->stats().sealed_segments_count, 1U);
  verify_rows(*collection);
  EXPECT_TRUE(std::filesystem::is_directory(source))
      << "in-process seal completion defers active-source reclamation to the next open";
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto second = Collection::open(root);
  ASSERT_TRUE(second.ok()) << second.status().diagnostic();
  EXPECT_FALSE(std::filesystem::exists(source));
  EXPECT_EQ(active_directory_count(root), 1U);
  verify_rows(*second.value());
  ASSERT_TRUE(second.value()->close().ok());
  second.value().reset();

  auto stable = Collection::open(root);
  ASSERT_TRUE(stable.ok()) << stable.status().diagnostic();
  EXPECT_FALSE(std::filesystem::exists(source));
  EXPECT_EQ(active_directory_count(root), 1U);
  verify_rows(*stable.value());
  ASSERT_TRUE(stable.value()->close().ok());
}

struct B09Case {
  const char *name;
  CollectionSealFailPoint failpoint;
};

class CollectionActiveLaserB09Recovery : public ::testing::TestWithParam<B09Case> {};

TEST_P(CollectionActiveLaserB09Recovery, ReferencedSourceSurvivesSweepUntilIdleOpen) {
  const auto param = GetParam();
  TemporaryDirectory root(param.name);
  create_populated_collection(root.path());
  ASSERT_TRUE(std::filesystem::is_directory(source_directory(root.path())));

  const auto crash = kill_after_seal_failpoint(root.path(), param.failpoint);
  assert_true_sigkill(crash);
  EXPECT_TRUE(std::filesystem::is_directory(source_directory(root.path())));
  finish_recovery_and_verify(root.path());
}

INSTANTIATE_TEST_SUITE_P(
    B09Matrix,
    CollectionActiveLaserB09Recovery,
    ::testing::Values(
        B09Case{"RI",
                CollectionSealFailPoint::after_active_control_publish_before_routing_install},
        B09Case{"SA", CollectionSealFailPoint::after_successor_switch},
        B09Case{"B", CollectionSealFailPoint::during_export_build},
        B09Case{"M", CollectionSealFailPoint::after_manifest_publish}),
    [](const ::testing::TestParamInfo<B09Case> &info) {
      return info.param.name;
    });

TEST(CollectionActiveLaserB09RecoveryStandalone, ConsecutiveCrashAfterReopenKeepsSourcePath) {
  TemporaryDirectory root("2k");
  create_populated_collection(root.path());
  auto first = kill_after_seal_failpoint(
      root.path(), CollectionSealFailPoint::after_active_control_publish_before_routing_install);
  assert_true_sigkill(first);
  ASSERT_TRUE(std::filesystem::is_directory(source_directory(root.path())));

  auto second = kill_after_open(root.path());
  assert_true_sigkill(second);
  EXPECT_TRUE(std::filesystem::is_directory(source_directory(root.path())))
      << "the second boot sweep must not unlink a still-referenced source";
  finish_recovery_and_verify(root.path());
}
#endif

}  // namespace
}  // namespace alaya

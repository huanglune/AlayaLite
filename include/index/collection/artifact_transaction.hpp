// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "core/resource_contexts.hpp"
#include "core/value_types.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "utils/platform.hpp"
#include "utils/platform_fs.hpp"
#include "utils/log.hpp"

namespace alaya::internal::collection {

enum class ArtifactAbortPolicy : std::uint8_t {
  eager_cleanup = 0,
  retain_for_restart_cleanup = 1,
};

enum class ArtifactTransactionFailPoint : std::uint8_t {
  none = 0,
  after_staging_write = 1,
  before_ready = 2,
  after_ready_before_publish = 3,
  after_payload_publish_before_manifest = 4,
};

struct ArtifactTransactionOptions {
  std::filesystem::path collection_root{};
  std::filesystem::path target_relative_directory{};
  std::string transaction_id{};
  bool manifest_v2_writer{};
  ArtifactAbortPolicy abort_policy{ArtifactAbortPolicy::eager_cleanup};
  ArtifactTransactionFailPoint fail_point{ArtifactTransactionFailPoint::none};
};

struct LogicalArtifactSpec {
  std::string logical_name{};
  std::filesystem::path relative_path{};
  bool required{true};
  ReaderCompatibilityV2 reader_compatibility{};
};

class ArtifactControlPlaneTransaction {
 public:
  ArtifactControlPlaneTransaction(const ArtifactControlPlaneTransaction &) = delete;
  auto operator=(const ArtifactControlPlaneTransaction &)
      -> ArtifactControlPlaneTransaction & = delete;
  ArtifactControlPlaneTransaction(ArtifactControlPlaneTransaction &&) = delete;
  auto operator=(ArtifactControlPlaneTransaction &&)
      -> ArtifactControlPlaneTransaction & = delete;

  ~ArtifactControlPlaneTransaction() {
    if (state_ != State::published && state_ != State::aborted &&
        options_.abort_policy == ArtifactAbortPolicy::eager_cleanup) {
      cleanup_owned_paths();
    }
  }

  [[nodiscard]] static auto begin(ArtifactTransactionOptions options,
                                  core::BuildContext &context) noexcept
      -> core::Result<std::unique_ptr<ArtifactControlPlaneTransaction>> {
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::build);
    if (!control.ok()) {
      return control;
    }
    auto reservation = context.growing_reservation.ensure(
        4096,
        core::OperationStage::build,
        "artifact transaction staging reservation is too small");
    if (!reservation.ok()) {
      return reservation;
    }
    if (context.io_credits.available_requests != core::kUnlimitedResource &&
        context.io_credits.available_requests < 4) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::build,
                                 core::StatusDetail::budget_denied,
                                 "artifact transaction requires at least four I/O request credits",
                                 core::Retryability::retryable_with_backoff);
    }
    try {
      if (options.collection_root.empty() || options.target_relative_directory.empty() ||
          !artifact_manifest_v2_detail::safe_relative_path(
              options.target_relative_directory.generic_string())) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::build,
                                   core::StatusDetail::malformed_struct,
                                   "artifact transaction root/target is invalid");
      }
      if (options.transaction_id.empty()) {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        options.transaction_id = "tx_" + std::to_string(platform::get_pid()) + "_" +
                                 std::to_string(ticks);
      }
      if (!safe_component(options.transaction_id)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::build,
                                   core::StatusDetail::malformed_struct,
                                   "artifact transaction id must be one safe path component");
      }
      auto transaction = std::unique_ptr<ArtifactControlPlaneTransaction>(
          new ArtifactControlPlaneTransaction(std::move(options),
                                              context.io_credits,
                                              context.deadline,
                                              context.cancellation));
      std::filesystem::create_directories(transaction->staging_root_.parent_path());
      if (!std::filesystem::create_directory(transaction->staging_root_)) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::build,
                                   core::StatusDetail::already_exists,
                                   "artifact transaction staging directory already exists");
      }
      platform::sync_directory_or_throw(transaction->staging_root_.parent_path());
      return std::move(transaction);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
  }

  [[nodiscard]] auto staging_payload_directory() const noexcept
      -> const std::filesystem::path & {
    return staging_payload_;
  }

  [[nodiscard]] auto final_payload_directory() const noexcept
      -> const std::filesystem::path & {
    return final_payload_;
  }

  // Engine save path: resolves all logical names before the first artifact is
  // written and returns the existing core ArtifactWriter view over staging.
  [[nodiscard]] auto writer(std::vector<LogicalArtifactSpec> specs) noexcept
      -> core::Result<core::ArtifactWriter> {
    if (state_ != State::created) {
      return invalid_state(core::OperationStage::save,
                           "artifact writer can only be created once");
    }
    try {
      auto status = install_specs(std::move(specs));
      if (!status.ok()) {
        return status;
      }
      std::filesystem::create_directory(staging_payload_);
      rebuild_locations();
      return core::ArtifactWriter(std::span<const core::ArtifactLocation>(locations_));
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  // Builder path: DiskFlatBuilder creates the payload directory atomically
  // inside this transaction, then the transaction adopts its logical files.
  [[nodiscard]] auto adopt(std::vector<LogicalArtifactSpec> specs) noexcept -> core::Status {
    if (state_ != State::created) {
      return invalid_state(core::OperationStage::build,
                           "artifact transaction can only adopt files once");
    }
    try {
      if (!std::filesystem::is_directory(staging_payload_)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::build,
                                   core::StatusDetail::malformed_struct,
                                   "artifact builder did not create its staging payload directory");
      }
      return install_specs(std::move(specs));
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
  }

  [[nodiscard]] auto prepare(SegmentEntryV2 segment) noexcept
      -> core::Result<SegmentEntryV2> {
    if (state_ != State::created || bindings_.empty()) {
      return invalid_state(core::OperationStage::save,
                           "artifact transaction has no adopted logical artifacts");
    }
    try {
      const auto control = validate_control(core::OperationStage::save);
      if (!control.ok()) {
        return control;
      }
      for (const auto &binding : bindings_) {
        if (binding.spec.required && !std::filesystem::is_regular_file(binding.absolute_path)) {
          return core::Status::error(core::StatusCode::io_error,
                                     core::OperationStage::save,
                                     core::StatusDetail::none,
                                     "required logical artifact was not written: " +
                                         binding.spec.logical_name);
        }
      }
      state_ = State::artifacts_written;
      if (options_.fail_point == ArtifactTransactionFailPoint::after_staging_write) {
        return injected_failure("after staging artifact write");
      }

      std::uint64_t io_bytes{};
      segment.artifacts.clear();
      for (const auto &binding : bindings_) {
        if (!std::filesystem::exists(binding.absolute_path)) {
          continue;
        }
        platform::sync_file_or_throw(binding.absolute_path);
        const auto bytes = std::filesystem::file_size(binding.absolute_path);
        if (bytes > std::numeric_limits<std::uint64_t>::max() - io_bytes) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::save,
                                     core::StatusDetail::arithmetic_overflow,
                                     "artifact transaction byte size overflows uint64");
        }
        io_bytes += static_cast<std::uint64_t>(bytes);
        OwnedArtifactV2 artifact;
        artifact.logical_name = binding.spec.logical_name;
        artifact.relative_path =
            (options_.target_relative_directory / binding.spec.relative_path).generic_string();
        artifact.required = binding.spec.required;
        artifact.size_bytes = static_cast<std::uint64_t>(bytes);
        artifact.checksum_algorithm = ChecksumAlgorithmV2::sha256;
        artifact.digest = sha256_file(binding.absolute_path);
        artifact.ready = true;
        artifact.reader_compatibility = binding.spec.reader_compatibility;
        segment.artifacts.push_back(std::move(artifact));
      }
      if (io_credits_.available_bytes != core::kUnlimitedResource &&
          io_bytes > io_credits_.available_bytes) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::save,
                                   core::StatusDetail::budget_denied,
                                   "artifact transaction I/O byte credits are too small",
                                   core::Retryability::retryable_with_backoff);
      }

      if (!options_.manifest_v2_writer) {
        prepared_segment_ = std::move(segment);
        state_ = State::prepared;
        return prepared_segment_;
      }

      const auto owned_body = serialize_owned_artifacts(segment);
      const auto owned_path = staging_payload_ / kOwnedArtifactManifestFilename;
      platform::write_all_fsync(owned_path, owned_body.data(), owned_body.size());
      OwnedArtifactV2 owned_manifest;
      owned_manifest.logical_name = "artifact_manifest_v2";
      owned_manifest.relative_path =
          (options_.target_relative_directory / kOwnedArtifactManifestFilename).generic_string();
      owned_manifest.required = true;
      owned_manifest.size_bytes = owned_body.size();
      owned_manifest.digest = sha256(owned_body);
      owned_manifest.ready = true;
      owned_manifest.reader_compatibility.required_features = {"manifest_v2"};
      segment.artifacts.push_back(owned_manifest);
      state_ = State::prepared;
      if (options_.fail_point == ArtifactTransactionFailPoint::before_ready) {
        return injected_failure("before READY publication");
      }

      const auto ready_body = "version=1\nsegment=" + segment.segment_id +
                              "\ngeneration=" + std::to_string(segment.generation) +
                              "\nowned_manifest_sha256=" + owned_manifest.digest.hex() + "\n";
      const auto ready_path = staging_payload_ / kReadyFilename;
      platform::write_all_fsync(ready_path, ready_body.data(), ready_body.size());
      segment.ready = true;
      segment.ready_marker =
          (options_.target_relative_directory / kReadyFilename).generic_string();
      segment.ready_digest = sha256(ready_body);
      platform::sync_directory_or_throw(staging_payload_);
      platform::sync_directory_or_throw(staging_root_);
      prepared_segment_ = std::move(segment);
      state_ = State::ready;
      if (options_.fail_point == ArtifactTransactionFailPoint::after_ready_before_publish) {
        return injected_failure("after READY and before atomic publish");
      }
      return prepared_segment_;
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::save,
                                 core::StatusDetail::none,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  // Step five. The payload directory is made durable first; routing changes
  // only when collection_manifest.txt is atomically replaced. A crash in
  // between leaves an owned, READY orphan that restart cleanup can identify.
  [[nodiscard]] auto publish(ArtifactManifestV2 manifest) noexcept -> core::Status {
    const auto expected_state =
        options_.manifest_v2_writer ? State::ready : State::prepared;
    if (state_ != expected_state) {
      return invalid_state(core::OperationStage::save,
                           "artifact transaction is not ready to publish");
    }
    try {
      std::filesystem::create_directories(final_payload_.parent_path());
      if (std::filesystem::exists(final_payload_)) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::save,
                                   core::StatusDetail::already_exists,
                                   "artifact publish target already exists");
      }
      platform::atomic_replace_no_overwrite(staging_payload_, final_payload_);
      payload_published_ = true;
      platform::sync_directory_or_throw(final_payload_.parent_path());
      if (options_.fail_point ==
          ArtifactTransactionFailPoint::after_payload_publish_before_manifest) {
        return injected_failure("after payload publish and before manifest publish");
      }

      if (options_.manifest_v2_writer) {
        auto found = std::find_if(manifest.segments.begin(),
                                  manifest.segments.end(),
                                  [&](const SegmentEntryV2 &entry) {
                                    return entry.segment_id == prepared_segment_.segment_id;
                                  });
        if (found == manifest.segments.end()) {
          manifest.segments.push_back(prepared_segment_);
        } else {
          *found = prepared_segment_;
        }
        const auto body = manifest.serialize();
        const auto pending_manifest = options_.collection_root /
                                      (".collection_manifest.v2." + options_.transaction_id);
        platform::write_all_fsync(pending_manifest, body.data(), body.size());
        platform::atomic_replace(pending_manifest,
                                 options_.collection_root / kCollectionManifestFilename);
        manifest_published_ = true;
        state_ = State::published;
        try {
          platform::sync_directory_or_throw(options_.collection_root);
        } catch (const std::exception &error) {
          // Atomic replacement already changed routing. Never report an
          // abortable transaction or delete its payload after this point.
          LOG_WARN("manifest v2 root-directory fsync failed after atomic publication: {}",
                   error.what());
        }
      }
      std::error_code ec;
      std::filesystem::remove(staging_root_, ec);
      state_ = State::published;
      return core::Status::success();
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::save,
                                 core::StatusDetail::none,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  [[nodiscard]] auto abort() noexcept -> core::Status {
    if (state_ == State::published || manifest_published_) {
      return invalid_state(core::OperationStage::save,
                           "published artifact transaction cannot be aborted");
    }
    cleanup_owned_paths();
    state_ = State::aborted;
    return core::Status::success();
  }

  // Called while the collection has exclusive control-plane ownership. Every
  // staging directory is an interrupted transaction. READY final directories
  // carrying ARTIFACTS.v2 are removed only when no v2 manifest references
  // them; legacy directories are never selected by this cleanup rule.
  [[nodiscard]] static auto cleanup_orphans(const std::filesystem::path &collection_root) noexcept
      -> core::Status {
    try {
      std::error_code ec;
      std::filesystem::remove_all(collection_root / kStagingDirectory, ec);
      if (ec) {
        return core::Status::error(core::StatusCode::io_error,
                                   core::OperationStage::open,
                                   core::StatusDetail::none,
                                   "cannot remove interrupted artifact staging directories: " +
                                       ec.message());
      }
      std::set<std::filesystem::path> live_directories;
      const auto manifest_path = collection_root / kCollectionManifestFilename;
      if (std::filesystem::is_regular_file(manifest_path)) {
        const auto prefix = platform::read_file_prefix(manifest_path, 9);
        if (prefix == "version=2") {
          const auto manifest = ArtifactManifestV2::load(manifest_path);
          for (const auto &segment : manifest.segments) {
            const auto marker = std::filesystem::path(segment.ready_marker);
            live_directories.insert(marker.parent_path().lexically_normal());
          }
        }
      }
      const auto segments_root = collection_root / "segments";
      if (std::filesystem::is_directory(segments_root)) {
        for (const auto &entry : std::filesystem::directory_iterator(segments_root)) {
          if (!entry.is_directory()) {
            continue;
          }
          const auto relative =
              std::filesystem::relative(entry.path(), collection_root).lexically_normal();
          if (live_directories.contains(relative)) {
            continue;
          }
          if (std::filesystem::is_regular_file(entry.path() / kOwnedArtifactManifestFilename) &&
              std::filesystem::is_regular_file(entry.path() / kReadyFilename)) {
            std::filesystem::remove_all(entry.path());
          }
        }
        platform::sync_directory_or_throw(segments_root);
      }
      return core::Status::success();
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::open,
                                 core::StatusDetail::none,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  static constexpr std::string_view kStagingDirectory{".alaya_staging"};
  static constexpr std::string_view kOwnedArtifactManifestFilename{"ARTIFACTS.v2"};
  static constexpr std::string_view kReadyFilename{"READY"};

 private:
  enum class State : std::uint8_t {
    created,
    artifacts_written,
    prepared,
    ready,
    published,
    aborted,
  };

  struct Binding {
    LogicalArtifactSpec spec{};
    std::filesystem::path absolute_path{};
    std::string absolute_path_string{};
  };

  ArtifactControlPlaneTransaction(ArtifactTransactionOptions options,
                                  core::IoCredits io_credits,
                                  core::Deadline deadline,
                                  core::CancellationToken cancellation)
      : options_(std::move(options)),
        io_credits_(std::move(io_credits)),
        deadline_(std::move(deadline)),
        cancellation_(std::move(cancellation)),
        staging_root_(options_.collection_root / kStagingDirectory / options_.transaction_id),
        staging_payload_(staging_root_ / options_.target_relative_directory.filename()),
        final_payload_(options_.collection_root / options_.target_relative_directory) {}

  [[nodiscard]] static auto safe_component(std::string_view value) -> bool {
    return !value.empty() && value != "." && value != ".." && value.find('/') == value.npos &&
           value.find('\\') == value.npos && value.find('\0') == value.npos;
  }

  [[nodiscard]] auto install_specs(std::vector<LogicalArtifactSpec> specs) -> core::Status {
    if (specs.empty()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "artifact transaction requires at least one logical artifact");
    }
    std::set<std::string> logical_names;
    std::set<std::string> relative_paths;
    bindings_.clear();
    bindings_.reserve(specs.size());
    for (auto &spec : specs) {
      const auto relative = spec.relative_path.generic_string();
      if (spec.logical_name.empty() || !logical_names.insert(spec.logical_name).second ||
          !artifact_manifest_v2_detail::safe_relative_path(relative) ||
          !relative_paths.insert(relative).second) {
        bindings_.clear();
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::save,
                                   core::StatusDetail::malformed_struct,
                                   "logical artifact names/paths must be unique safe relative paths");
      }
      Binding binding;
      binding.absolute_path = staging_payload_ / spec.relative_path;
      binding.absolute_path_string = binding.absolute_path.string();
      binding.spec = std::move(spec);
      bindings_.push_back(std::move(binding));
    }
    return core::Status::success();
  }

  void rebuild_locations() {
    locations_.clear();
    locations_.reserve(bindings_.size());
    for (const auto &binding : bindings_) {
      std::filesystem::create_directories(binding.absolute_path.parent_path());
      locations_.emplace_back(binding.spec.logical_name, binding.absolute_path_string);
    }
  }

  [[nodiscard]] auto validate_control(core::OperationStage stage) const -> core::Status {
    return core::validate_runtime_control(deadline_, cancellation_, stage);
  }

  [[nodiscard]] static auto serialize_owned_artifacts(const SegmentEntryV2 &segment)
      -> std::string {
    std::string body = "version=2\nsegment=" +
                       artifact_manifest_v2_detail::encode_string(segment.segment_id) +
                       "\ngeneration=" + std::to_string(segment.generation) +
                       "\nartifacts=" + std::to_string(segment.artifacts.size()) + "\n";
    for (std::size_t index = 0; index < segment.artifacts.size(); ++index) {
      const auto &artifact = segment.artifacts[index];
      const auto prefix = "artifact." + std::to_string(index) + ".";
      body += prefix + "name=" +
              artifact_manifest_v2_detail::encode_string(artifact.logical_name) + "\n";
      body += prefix + "path=" +
              artifact_manifest_v2_detail::encode_string(artifact.relative_path) + "\n";
      body += prefix + "required=" + (artifact.required ? "1\n" : "0\n");
      body += prefix + "bytes=" + std::to_string(artifact.size_bytes) + "\n";
      body += prefix + "checksum=sha256\n";
      body += prefix + "digest=" + artifact.digest.hex() + "\n";
    }
    return body;
  }

  [[nodiscard]] static auto invalid_state(core::OperationStage stage, std::string diagnostic)
      -> core::Status {
    return core::Status::error(core::StatusCode::conflict,
                               stage,
                               core::StatusDetail::none,
                               std::move(diagnostic));
  }

  [[nodiscard]] static auto injected_failure(std::string_view point) -> core::Status {
    return core::Status::error(core::StatusCode::io_error,
                               core::OperationStage::save,
                               core::StatusDetail::none,
                               "injected artifact transaction failure " + std::string(point));
  }

  void cleanup_owned_paths() noexcept {
    std::error_code ec;
    std::filesystem::remove_all(staging_root_, ec);
    if (payload_published_) {
      if (manifest_published_) {
        return;
      }
      std::filesystem::remove_all(final_payload_, ec);
    }
  }

  ArtifactTransactionOptions options_{};
  core::IoCredits io_credits_{};
  core::Deadline deadline_{};
  core::CancellationToken cancellation_{};
  std::filesystem::path staging_root_{};
  std::filesystem::path staging_payload_{};
  std::filesystem::path final_payload_{};
  std::vector<Binding> bindings_{};
  std::vector<core::ArtifactLocation> locations_{};
  SegmentEntryV2 prepared_segment_{};
  State state_{State::created};
  bool payload_published_{};
  bool manifest_published_{};
};

}  // namespace alaya::internal::collection

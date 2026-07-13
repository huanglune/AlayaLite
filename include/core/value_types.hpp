// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/resource_contexts.hpp"

namespace alaya::core {

// Contract v3 is frozen. Engine-specific boundary fields belong in keyed
// extensions; engines must not grow these structs privately.
using RowCount = std::uint64_t;

struct SegmentRowId {
  std::uint64_t value{};

  constexpr SegmentRowId() = default;
  constexpr explicit SegmentRowId(std::uint64_t row) : value(row) {}
  [[nodiscard]] constexpr explicit operator std::uint64_t() const noexcept { return value; }
  auto operator<=>(const SegmentRowId &) const = default;
};

using ExternalId [[deprecated("ExternalId was removed by contract v3; use SegmentRowId")]] =
    SegmentRowId;

enum class LogicalIdKind : std::uint8_t { utf8 = 1, legacy_uint64 = 2 };

struct LogicalIdView {
  VersionedStructHeader header{};
  std::uint32_t encoding_version{1};
  LogicalIdKind kind{LogicalIdKind::utf8};
  std::uint8_t reserved_bytes[3]{};
  std::span<const std::byte> canonical_bytes{};
  std::uint64_t reserved[3]{};

  LogicalIdView() : header(current_struct_header<LogicalIdView>()) {}
  LogicalIdView(LogicalIdKind id_kind, std::span<const std::byte> bytes)
      : header(current_struct_header<LogicalIdView>()), kind(id_kind), canonical_bytes(bytes) {}
};

class LogicalId {
 public:
  LogicalId() : header(current_struct_header<LogicalId>()) {}

  [[nodiscard]] static auto from_utf8(std::string_view value) -> LogicalId {
    LogicalId id;
    id.kind_ = LogicalIdKind::utf8;
    id.bytes_.resize(value.size());
    std::memcpy(id.bytes_.data(), value.data(), value.size());
    return id;
  }

  [[nodiscard]] static auto from_legacy_uint64(std::uint64_t value) -> LogicalId {
    LogicalId id;
    id.kind_ = LogicalIdKind::legacy_uint64;
    id.bytes_.resize(sizeof(value));
    for (std::size_t index = 0; index < sizeof(value); ++index) {
      const auto shift = static_cast<unsigned>((sizeof(value) - index - 1) * 8);
      id.bytes_[index] = static_cast<std::byte>((value >> shift) & 0xffU);
    }
    return id;
  }

  [[nodiscard]] auto view() const noexcept -> LogicalIdView { return {kind_, bytes_}; }
  [[nodiscard]] auto kind() const noexcept -> LogicalIdKind { return kind_; }
  [[nodiscard]] auto canonical_bytes() const noexcept -> std::span<const std::byte> {
    return bytes_;
  }

  [[nodiscard]] auto compare(const LogicalId &other) const noexcept -> int {
    if (kind_ != other.kind_) {
      return static_cast<unsigned>(kind_) < static_cast<unsigned>(other.kind_) ? -1 : 1;
    }
    const auto mismatch =
        std::mismatch(bytes_.begin(), bytes_.end(), other.bytes_.begin(), other.bytes_.end());
    if (mismatch.first == bytes_.end() && mismatch.second == other.bytes_.end()) {
      return 0;
    }
    if (mismatch.first == bytes_.end()) {
      return -1;
    }
    if (mismatch.second == other.bytes_.end()) {
      return 1;
    }
    return std::to_integer<unsigned>(*mismatch.first) < std::to_integer<unsigned>(*mismatch.second)
               ? -1
               : 1;
  }

  [[nodiscard]] auto operator==(const LogicalId &other) const noexcept -> bool {
    return kind_ == other.kind_ && bytes_ == other.bytes_;
  }

  VersionedStructHeader header{};

 private:
  std::uint32_t encoding_version_{1};
  LogicalIdKind kind_{LogicalIdKind::utf8};
  std::uint8_t reserved_bytes_[3]{};
  std::vector<std::byte> bytes_{};
  std::uint64_t reserved_[2]{};
};

enum class ScalarType : std::uint8_t { float32 = 1, int8 = 2, uint8 = 3 };

template <class T>
inline constexpr auto scalar_type_for = ScalarType::float32;
template <>
inline constexpr auto scalar_type_for<std::int8_t> = ScalarType::int8;
template <>
inline constexpr auto scalar_type_for<std::uint8_t> = ScalarType::uint8;
template <>
inline constexpr auto scalar_type_for<float> = ScalarType::float32;

[[nodiscard]] constexpr auto scalar_type_size(ScalarType scalar_type) noexcept -> std::uint32_t {
  switch (scalar_type) {
    case ScalarType::float32:
      return sizeof(float);
    case ScalarType::int8:
      return sizeof(std::int8_t);
    case ScalarType::uint8:
      return sizeof(std::uint8_t);
  }
  return 0;
}

struct TypedTensorView {
  VersionedStructHeader header{};
  const void *data{};
  ScalarType scalar_type{ScalarType::float32};
  std::uint8_t reserved_bytes[7]{};
  RowCount rows{};
  std::uint32_t dim{};
  std::uint32_t reserved_dim{};
  std::uint64_t row_stride{};
  std::uint64_t reserved[4]{};

  TypedTensorView() : header(current_struct_header<TypedTensorView>()) {}
  TypedTensorView(const void *values,
                  ScalarType type,
                  RowCount row_count,
                  std::uint32_t dimension,
                  std::uint64_t stride)
      : header(current_struct_header<TypedTensorView>()),
        data(values),
        scalar_type(type),
        rows(row_count),
        dim(dimension),
        row_stride(stride) {}

  template <class T>
    requires(std::is_same_v<std::remove_cv_t<T>, float> ||
             std::is_same_v<std::remove_cv_t<T>, std::int8_t> ||
             std::is_same_v<std::remove_cv_t<T>, std::uint8_t>)
  [[nodiscard]] static auto contiguous(const T *values, RowCount row_count, std::uint32_t dimension)
      -> TypedTensorView {
    std::uint64_t stride{};
    if (!checked_multiply(dimension, sizeof(T), stride)) {
      stride = std::numeric_limits<std::uint64_t>::max();
    }
    return {values, scalar_type_for<std::remove_cv_t<T>>, row_count, dimension, stride};
  }

  template <class T>
  [[nodiscard]] auto row(RowCount index) const noexcept -> const T * {
    const auto *bytes = static_cast<const std::byte *>(data);
    return reinterpret_cast<const T *>(bytes + index * row_stride);
  }
};

[[nodiscard]] inline auto validate_tensor(const TypedTensorView &tensor,
                                          std::uint32_t expected_dim,
                                          OperationStage stage) -> Status {
  if (!is_current_struct(tensor)) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::malformed_struct,
                         "TypedTensorView has an incompatible size or ABI version");
  }
  const auto scalar_bytes = scalar_type_size(tensor.scalar_type);
  if (scalar_bytes == 0) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::unsupported_scalar_type,
                         "TypedTensorView scalar type is unsupported");
  }
  if (tensor.dim != expected_dim) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::dimension_mismatch,
                         "tensor dimension does not match the segment descriptor");
  }
  if (tensor.rows != 0 && tensor.dim != 0 && tensor.data == nullptr) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::null_data,
                         "non-empty tensor data is null");
  }
  std::uint64_t row_bytes{};
  if (!checked_multiply(tensor.dim, scalar_bytes, row_bytes)) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::arithmetic_overflow,
                         "tensor row byte size overflows uint64");
  }
  if (tensor.rows != 0 && tensor.row_stride < row_bytes) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::invalid_stride,
                         "tensor row stride is smaller than one row");
  }
  if (tensor.rows != 0) {
    std::uint64_t prefix{};
    std::uint64_t extent{};
    if (!checked_multiply(tensor.rows - 1, tensor.row_stride, prefix) ||
        !checked_add(prefix, row_bytes, extent)) {
      return Status::error(StatusCode::invalid_argument,
                           stage,
                           StatusDetail::arithmetic_overflow,
                           "tensor byte extent overflows uint64");
    }
  }
  return Status::success();
}

enum class Metric : std::uint8_t { l2 = 0, inner_product = 1, cosine = 2 };
enum class Medium : std::uint8_t { memory = 0, disk = 1 };
enum class MetricPreprocessing : std::uint8_t {
  none = 0,
  l2_normalized = 1,
  engine_quantized = 2,
};

enum class FilterPolicy : std::uint8_t { automatic = 0, strict = 1, allow_partial = 2 };
enum class FilterExecution : std::uint8_t {
  prefilter = 0,
  traversal = 1,
  postfilter = 2,
};
enum class RerankPolicy : std::uint8_t { automatic = 0, disabled = 1, exact_required = 2 };
enum class PartialResultPolicy : std::uint8_t { discard = 0, retain = 1 };
enum class UnknownExtensionPolicy : std::uint8_t { ignore_safe = 0, reject = 1 };

struct AlgorithmSearchExtension {
  VersionedStructHeader header{};
  AlgorithmId algorithm_id{};
  std::uint32_t extension_version{1};
  UnknownExtensionPolicy unknown_policy{UnknownExtensionPolicy::reject};
  std::uint8_t reserved_bytes[3]{};
  const void *payload{};
  std::uint64_t payload_size{};
  std::uint64_t reserved[3]{};

  AlgorithmSearchExtension() : header(current_struct_header<AlgorithmSearchExtension>()) {}
};

struct SearchOptions {
  VersionedStructHeader header{};
  std::uint64_t top_k{10};
  std::uint64_t latency_goal_nanoseconds{};
  std::uint64_t deadline_steady_nanoseconds{};
  float quality_hint{1.0F};
  RerankPolicy rerank_policy{RerankPolicy::automatic};
  FilterPolicy filter_policy{FilterPolicy::automatic};
  PartialResultPolicy partial_result_policy{PartialResultPolicy::discard};
  std::span<const AlgorithmSearchExtension> extensions{};
  std::uint64_t reserved[4]{};

  SearchOptions() : header(current_struct_header<SearchOptions>()) {}
  explicit SearchOptions(std::uint64_t requested_top_k)
      : header(current_struct_header<SearchOptions>()), top_k(requested_top_k) {}
};

enum class SegmentFilterKind : std::uint8_t {
  none = 0,
  bitmap = 1,
  sorted_rows = 2,
  predicate = 3,
  composite = 4,
};

struct SegmentFilterView {
  VersionedStructHeader header{};
  SegmentFilterKind kind{SegmentFilterKind::none};
  bool exact{};
  std::uint8_t reserved_bytes[6]{};
  std::uint64_t metadata_epoch{};
  const void *payload{};
  std::uint64_t payload_size{};
  double selectivity_hint{1.0};
  std::uint64_t reserved[3]{};

  SegmentFilterView() : header(current_struct_header<SegmentFilterView>()) {}
};

enum class ScoreKind : std::uint8_t {
  distance = 0,
  similarity = 1,
  rank_only = 2,
};

enum class ResultFlag : std::uint32_t {
  none = 0,
  approximate = 1U << 0U,
  exact_reranked = 1U << 1U,
  filtered = 1U << 2U,
  version_checked = 1U << 3U,
};

[[nodiscard]] constexpr auto operator|(ResultFlag lhs, ResultFlag rhs) noexcept -> ResultFlag {
  return static_cast<ResultFlag>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

enum class SearchCompleteness : std::uint8_t {
  complete_k = 0,
  eligible_exhausted = 1,
  strategy_incomplete = 2,
  cancelled_partial = 3,
  failed = 4,
};

struct SearchHit {
  VersionedStructHeader header{};
  SegmentRowId row_id{};
  float score{};
  ScoreKind score_kind{ScoreKind::distance};
  Metric comparable_metric{Metric::l2};
  std::uint16_t reserved_score{};
  ResultFlag result_flags{ResultFlag::none};
  std::uint32_t reserved_flags{};
  std::uint64_t generation{};
  std::uint64_t row_version{};
  std::uint64_t reserved[2]{};

  SearchHit() : header(current_struct_header<SearchHit>()) {}
  SearchHit(SegmentRowId id,
            float value,
            ScoreKind kind,
            Metric metric,
            ResultFlag flags = ResultFlag::none)
      : header(current_struct_header<SearchHit>()),
        row_id(id),
        score(value),
        score_kind(kind),
        comparable_metric(metric),
        result_flags(flags) {}
};

using SearchSink = std::span<SearchHit>;

struct SearchResponse {
  VersionedStructHeader header{};
  RowCount query_count{};
  SearchSink hits{};
  std::span<RowCount> offsets{};
  std::span<RowCount> valid_counts{};
  std::span<Status> statuses{};
  std::span<SearchCompleteness> completeness{};
  ScoreKind score_kind{ScoreKind::distance};
  Metric comparable_metric{Metric::l2};
  std::uint16_t reserved_score{};
  ResultFlag result_flags{ResultFlag::none};
  std::uint32_t reserved_flags{};
  std::uint64_t reserved[4]{};

  SearchResponse() : header(current_struct_header<SearchResponse>()) {}

  void invalidate(const Status &failure) noexcept {
    std::fill(offsets.begin(), offsets.end(), RowCount{0});
    std::fill(valid_counts.begin(), valid_counts.end(), RowCount{0});
    std::fill(statuses.begin(), statuses.end(), failure);
    std::fill(completeness.begin(), completeness.end(), SearchCompleteness::failed);
    result_flags = ResultFlag::none;
  }
};

[[nodiscard]] inline auto validate_response(SearchResponse &response,
                                            RowCount query_count,
                                            std::uint64_t top_k,
                                            OperationStage stage) -> Status {
  if (!is_current_struct(response)) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::malformed_struct,
                         "SearchResponse has an incompatible size or ABI version");
  }
  std::uint64_t offset_count{};
  if (!checked_add(query_count, 1, offset_count) ||
      offset_count > std::numeric_limits<std::size_t>::max()) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::arithmetic_overflow,
                         "search response offset count is not representable");
  }
  if (response.offsets.size() < static_cast<std::size_t>(offset_count) ||
      response.valid_counts.size() < query_count || response.statuses.size() < query_count ||
      response.completeness.size() < query_count) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::sink_too_small,
                         "search response metadata arrays are too small");
  }
  std::uint64_t max_hits{};
  if (!checked_multiply(query_count, top_k, max_hits) ||
      max_hits > std::numeric_limits<std::size_t>::max()) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::arithmetic_overflow,
                         "rows * top_k is not representable");
  }
  if (response.hits.size() < static_cast<std::size_t>(max_hits)) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::sink_too_small,
                         "search hit sink is smaller than rows * top_k");
  }
  response.query_count = query_count;
  return Status::success();
}

inline void initialize_empty_response(SearchResponse &response,
                                      RowCount query_count,
                                      SearchCompleteness completeness) {
  response.query_count = query_count;
  std::fill_n(response.offsets.begin(), static_cast<std::size_t>(query_count + 1), RowCount{0});
  std::fill_n(response.valid_counts.begin(), static_cast<std::size_t>(query_count), RowCount{0});
  std::fill_n(response.statuses.begin(), static_cast<std::size_t>(query_count), Status::success());
  std::fill_n(response.completeness.begin(), static_cast<std::size_t>(query_count), completeness);
}

struct SearchRequest {
  VersionedStructHeader header{};
  TypedTensorView queries{};
  SearchOptions options{};
  SegmentFilterView filter{};
  SearchContext *context{};
  SearchResponse *response{};
  std::shared_ptr<void> lifetime_pin{};
  std::uint64_t reserved[4]{};

  SearchRequest() : header(current_struct_header<SearchRequest>()) {}
};

struct Descriptor {
  VersionedStructHeader header{};
  AlgorithmId algorithm_id{};
  std::uint32_t format_version{};
  std::uint32_t factory_version{};
  std::uint32_t dim{};
  Metric metric{Metric::l2};
  ScalarType stored_scalar_type{ScalarType::float32};
  Medium medium{Medium::memory};
  MetricPreprocessing preprocessing{MetricPreprocessing::none};
  std::uint64_t engine_factory_id{};
  std::uint64_t reserved[5]{};

  Descriptor() : header(current_struct_header<Descriptor>()) {}
};

struct ConcurrencyProfile {
  VersionedStructHeader header{};
  bool reentrant_search{true};
  bool search_with_stage{};
  bool search_with_publish{};
  bool serial_mutation{true};
  bool checkpoint_with_search{};
  bool freeze_with_search{};
  bool native_async{};
  bool cooperative_cancel{true};
  bool explicit_drain{true};
  std::uint8_t reserved_bytes[7]{};
  std::uint64_t reserved[3]{};

  ConcurrencyProfile() : header(current_struct_header<ConcurrencyProfile>()) {}
};

enum class OperationCapability : std::uint64_t {
  search = 1ULL << 0U,
  batch_search = 1ULL << 1U,
  mutation = 1ULL << 2U,
  save = 1ULL << 3U,
  export_rows = 1ULL << 4U,
  checkpoint = 1ULL << 5U,
  freeze = 1ULL << 6U,
  stats = 1ULL << 7U,
  close = 1ULL << 8U,
  drain = 1ULL << 9U,
};

[[nodiscard]] constexpr auto capability_bit(OperationCapability capability) noexcept
    -> std::uint64_t {
  return static_cast<std::uint64_t>(capability);
}

struct SegmentInstanceConfig {
  VersionedStructHeader header{};
  std::uint64_t enabled_operations{std::numeric_limits<std::uint64_t>::max()};
  bool readonly{true};
  std::uint8_t reserved_bytes[7]{};
  ConcurrencyProfile concurrency{};
  std::uint64_t reserved[4]{};

  SegmentInstanceConfig() : header(current_struct_header<SegmentInstanceConfig>()) {}
};

struct RuntimeCapabilities {
  VersionedStructHeader header{};
  std::uint64_t operations{};
  ConcurrencyProfile concurrency{};
  std::uint64_t reserved[4]{};

  RuntimeCapabilities() : header(current_struct_header<RuntimeCapabilities>()) {}
  [[nodiscard]] auto supports(OperationCapability capability) const noexcept -> bool {
    return (operations & capability_bit(capability)) != 0;
  }
};

enum class SegmentHealth : std::uint8_t { healthy = 0, degraded = 1, failed = 2, closed = 3 };

struct SegmentStats {
  VersionedStructHeader header{};
  std::uint64_t snapshot_version{};
  RowCount live_rows{};
  RowCount allocated_rows{};
  RowCount tombstone_rows{};
  RowCount pending_rows{};
  std::uint64_t resident_bytes{};
  std::uint64_t cache_bytes{};
  std::uint64_t dirty_bytes{};
  std::uint64_t inflight_search{};
  std::uint64_t inflight_mutation{};
  std::uint64_t inflight_io{};
  SegmentHealth health{SegmentHealth::healthy};
  StatusDetail last_error{StatusDetail::none};
  std::uint8_t reserved_bytes[5]{};
  std::uint64_t reserved[4]{};

  SegmentStats() : header(current_struct_header<SegmentStats>()) {}
};

struct Artifact {
  VersionedStructHeader header{};
  std::string_view name{};
  std::uint64_t size_bytes{};
  std::uint64_t checksum{};
  std::uint64_t reserved[3]{};

  Artifact() : header(current_struct_header<Artifact>()) {}
  Artifact(std::string_view logical_name, std::uint64_t bytes, std::uint64_t digest)
      : header(current_struct_header<Artifact>()),
        name(logical_name),
        size_bytes(bytes),
        checksum(digest) {}
};

struct ArtifactManifest {
  VersionedStructHeader header{};
  std::uint32_t schema_version{1};
  std::uint32_t format_version{};
  AlgorithmId algorithm_id{};
  std::span<const Artifact> artifacts{};
  std::uint64_t reserved[4]{};

  ArtifactManifest() : header(current_struct_header<ArtifactManifest>()) {}
};

struct ArtifactLocation {
  VersionedStructHeader header{};
  std::string_view name{};
  std::string_view path{};
  std::uint64_t reserved[2]{};

  ArtifactLocation() : header(current_struct_header<ArtifactLocation>()) {}
  ArtifactLocation(std::string_view logical_name, std::string_view file_path)
      : header(current_struct_header<ArtifactLocation>()), name(logical_name), path(file_path) {}
};

struct ArtifactView {
  VersionedStructHeader header{};
  std::span<const ArtifactLocation> locations{};
  std::uint64_t reserved[3]{};

  ArtifactView() : header(current_struct_header<ArtifactView>()) {}
  explicit ArtifactView(std::span<const ArtifactLocation> values)
      : header(current_struct_header<ArtifactView>()), locations(values) {}

  [[nodiscard]] auto find(std::string_view name) const noexcept -> std::string_view {
    for (const auto &location : locations) {
      if (location.name == name) {
        return location.path;
      }
    }
    return {};
  }
};

using ArtifactWriter = ArtifactView;

struct OpenOptions {
  VersionedStructHeader header{};
  std::uint64_t reserved[4]{};
  OpenOptions() : header(current_struct_header<OpenOptions>()) {}
};

struct SaveOptions {
  VersionedStructHeader header{};
  std::uint64_t reserved[4]{};
  SaveOptions() : header(current_struct_header<SaveOptions>()) {}
};

// Gate 2 freezes the operation shapes. The payloads become typed in the owner
// layers without changing the AnySegment slots.
struct OpaqueOperationRequest {
  VersionedStructHeader header{};
  const void *payload{};
  std::uint64_t payload_size{};
  std::uint64_t reserved[4]{};
  OpaqueOperationRequest() : header(current_struct_header<OpaqueOperationRequest>()) {}
};

struct MutationToken {
  VersionedStructHeader header{};
  std::uint64_t value{};
  std::uint64_t reserved[4]{};
  MutationToken() : header(current_struct_header<MutationToken>()) {}
};

struct CheckpointToken {
  VersionedStructHeader header{};
  std::uint64_t value{};
  std::uint64_t reserved[4]{};
  CheckpointToken() : header(current_struct_header<CheckpointToken>()) {}
};

struct FreezeToken {
  VersionedStructHeader header{};
  std::uint64_t value{};
  std::uint64_t reserved[4]{};
  FreezeToken() : header(current_struct_header<FreezeToken>()) {}
};

struct ExportCursor {
  VersionedStructHeader header{};
  void *state{};
  std::uint64_t reserved[4]{};
  ExportCursor() : header(current_struct_header<ExportCursor>()) {}
};

struct SealedArtifact {
  VersionedStructHeader header{};
  ArtifactManifest manifest{};
  std::uint64_t reserved[4]{};
  SealedArtifact() : header(current_struct_header<SealedArtifact>()) {}
};

static_assert(sizeof(SegmentRowId) == sizeof(std::uint64_t),
              "same-toolchain layout regression canary for SegmentRowId");
static_assert(std::is_trivially_copyable_v<SegmentRowId>);
static_assert(std::is_trivially_copyable_v<TypedTensorView>);
static_assert(std::is_trivially_copyable_v<SearchHit>);

}  // namespace alaya::core

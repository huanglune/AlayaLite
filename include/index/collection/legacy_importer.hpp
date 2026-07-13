// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include <rocksdb/options.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "index/collection/segmented_collection.hpp"
#include "index/disk/disk_flat_segment.hpp"
#include "utils/binary_io.hpp"
#include "utils/platform_fs.hpp"
#include "utils/scalar_data.hpp"

namespace alaya::internal::collection {

inline constexpr std::string_view kLegacyImportNamespace{"legacy_import_v1"};
inline constexpr std::string_view kLegacyImportIntentFilename{"intent.v1"};
inline constexpr std::string_view kLegacyImportAuditFilename{"audit.v1"};
inline constexpr std::string_view kLegacyImportMarkerFilename{"ACTIVE"};

enum class LegacyImporterFailPoint : std::uint8_t {
  none = 0,
  after_rocksdb_validation = 1,
  during_wal_decode = 2,
  after_op_id_reservation = 3,
  after_intent_write = 4,
  after_checkpoint_write = 5,
  before_marker = 6,
  after_marker = 7,
};

enum class LegacyOpenRoute : std::uint8_t {
  unavailable = 0,
  legacy_layout = 1,
  imported_layout = 2,
};

struct LegacyImportOptions {
  using ActiveRegistrationFactory =
      std::function<core::Result<SegmentRegistration>(const CollectionSchema &)>;

  std::filesystem::path source_root{};
  std::filesystem::path target_root{};
  CollectionFeatureFlags features{};
  LegacyImporterFailPoint fail_point{LegacyImporterFailPoint::none};
  // Optional Gate 9 facade composition seam.  The importer still owns and
  // validates the sealed legacy target; a canonical facade may append its
  // independently constructed active mutable registration without routing
  // the source through a legacy reader.
  ActiveRegistrationFactory active_registration_factory{};
};

struct LegacyImportAudit {
  std::uint32_t format_version{1};
  std::string source_fingerprint{};
  std::uint64_t source_file_count{};
  std::uint64_t source_bytes{};
  std::string source_kind{};
  core::ScalarType source_scalar_type{core::ScalarType::float32};
  std::string source_id_type{};
  std::uint32_t dim{};
  bool has_scalar_data{};
  std::string snapshot_id{};
  std::uint64_t snapshot_applied_through{};
  std::uint64_t maximum_seen_op_id{};
  std::uint64_t maximum_committed_op_id{};
  std::uint64_t minimum_next_op_id{1};
  std::uint64_t allocated_rows{};
  std::uint64_t live_rows{};
  std::uint64_t tombstone_rows{};
  std::uint64_t committed_wal_records{};
  bool torn_wal_tail{};
  std::uint64_t verified_wal_bytes{};
  std::string rocksdb_mode{"absent"};
  std::string checkpoint_name{};
  std::uint64_t wal_cut{};
  std::string manifest_sha256{};
  std::string segment_id{"seg_00000001"};
};

struct LegacyImportResult {
  std::shared_ptr<SegmentedCollection> collection{};
  LegacyImportAudit audit{};
  bool already_imported{};
};

namespace legacy_import_detail {

namespace fs = std::filesystem;

static_assert(std::endian::native == std::endian::little,
              "legacy PyIndex importer supports the pinned little-endian layout only");

inline constexpr std::uint64_t kMaximumSourceFileBytes = 1ULL << 30U;
inline constexpr std::uint64_t kSegmentId = 1;
inline constexpr std::uint64_t kSegmentGeneration = 1;
inline constexpr std::string_view kSegmentName{"seg_00000001"};

[[nodiscard]] inline auto failure(core::OperationStage stage,
                                  core::StatusCode code,
                                  std::string diagnostic,
                                  core::StatusDetail detail = core::StatusDetail::malformed_struct)
    -> core::Status {
  return core::Status::error(code, stage, detail, std::move(diagnostic));
}

[[nodiscard]] inline auto injected(LegacyImporterFailPoint point) -> core::Status {
  return failure(core::OperationStage::save,
                 core::StatusCode::io_error,
                 "injected legacy importer failure at cut " +
                     std::to_string(static_cast<unsigned>(point)),
                 core::StatusDetail::none);
}

[[nodiscard]] inline auto read_bytes(const fs::path &path,
                                     std::uint64_t maximum = kMaximumSourceFileBytes)
    -> std::vector<std::byte> {
  std::error_code ec;
  if (fs::is_symlink(path, ec) || !fs::is_regular_file(path, ec) || ec) {
    throw std::invalid_argument("legacy source file is absent, non-regular, or a symlink: " +
                                path.string());
  }
  const auto size = fs::file_size(path);
  if (size > maximum || size > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("legacy source file exceeds its bounded reader limit: " +
                                path.string());
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if ((!input && !input.eof()) || static_cast<std::size_t>(input.gcount()) != bytes.size()) {
    throw std::runtime_error("cannot read complete legacy source file: " + path.string());
  }
  return bytes;
}

[[nodiscard]] inline auto read_text(const fs::path &path, std::uint64_t maximum = 1ULL << 20U)
    -> std::string {
  const auto bytes = read_bytes(path, maximum);
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

[[nodiscard]] inline auto trim(std::string value) -> std::string {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.pop_back();
  }
  const auto first = std::find_if(value.begin(), value.end(), [](char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) == 0;
  });
  value.erase(value.begin(), first);
  return value;
}

[[nodiscard]] inline auto parse_u64(std::string_view value, std::string_view field)
    -> std::uint64_t {
  if (value.empty()) {
    throw std::invalid_argument(std::string(field) + " is empty");
  }
  std::uint64_t parsed{};
  for (const char digit : value) {
    if (digit < '0' || digit > '9' ||
        parsed >
            (std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(digit - '0')) /
                10U) {
      throw std::invalid_argument(std::string(field) + " is not a bounded uint64");
    }
    parsed = parsed * 10U + static_cast<std::uint64_t>(digit - '0');
  }
  return parsed;
}

[[nodiscard]] inline auto parse_lines(std::string_view body)
    -> std::map<std::string, std::string, std::less<>> {
  std::map<std::string, std::string, std::less<>> fields;
  std::istringstream lines{std::string(body)};
  for (std::string line; std::getline(lines, line);) {
    if (line.empty()) {
      continue;
    }
    const auto separator = line.find('=');
    if (separator == std::string::npos || separator == 0 ||
        !fields.emplace(line.substr(0, separator), line.substr(separator + 1)).second) {
      throw std::invalid_argument("legacy importer control file has a malformed/duplicate field");
    }
  }
  return fields;
}

[[nodiscard]] inline auto required_field(
    const std::map<std::string, std::string, std::less<>> &fields,
    std::string_view field) -> const std::string & {
  const auto found = fields.find(field);
  if (found == fields.end()) {
    throw std::invalid_argument("legacy importer control file misses field " + std::string(field));
  }
  return found->second;
}

[[nodiscard]] inline auto json_value_start(std::string_view body, std::string_view key)
    -> std::size_t {
  const auto quoted = "\"" + std::string(key) + "\"";
  const auto key_offset = body.find(quoted);
  if (key_offset == body.npos) {
    throw std::invalid_argument("legacy schema misses JSON key " + std::string(key));
  }
  const auto colon = body.find(':', key_offset + quoted.size());
  if (colon == body.npos) {
    throw std::invalid_argument("legacy schema key has no JSON value " + std::string(key));
  }
  auto offset = colon + 1;
  while (offset < body.size() && std::isspace(static_cast<unsigned char>(body[offset])) != 0) {
    ++offset;
  }
  return offset;
}

[[nodiscard]] inline auto json_string(std::string_view body, std::string_view key) -> std::string {
  auto offset = json_value_start(body, key);
  if (offset >= body.size() || body[offset++] != '"') {
    throw std::invalid_argument("legacy schema JSON key is not a string: " + std::string(key));
  }
  std::string value;
  for (; offset < body.size(); ++offset) {
    const auto ch = body[offset];
    if (ch == '"') {
      return value;
    }
    if (ch == '\\') {
      if (++offset >= body.size()) {
        break;
      }
      const auto escaped = body[offset];
      if (escaped != '"' && escaped != '\\' && escaped != '/') {
        throw std::invalid_argument("legacy schema uses an unsupported JSON escape");
      }
      value.push_back(escaped);
    } else {
      value.push_back(ch);
    }
  }
  throw std::invalid_argument("legacy schema contains an unterminated JSON string");
}

[[nodiscard]] inline auto json_bool(std::string_view body, std::string_view key) -> bool {
  const auto offset = json_value_start(body, key);
  if (body.substr(offset, 4) == "true") {
    return true;
  }
  if (body.substr(offset, 5) == "false") {
    return false;
  }
  throw std::invalid_argument("legacy schema JSON key is not a bool: " + std::string(key));
}

struct SourceSchema {
  std::string kind{};
  core::ScalarType scalar_type{core::ScalarType::float32};
  std::string scalar_name{};
  std::string id_type{};
  std::uint32_t id_bytes{};
  std::uint32_t dim{};
  bool has_scalar_data{};
};

[[nodiscard]] inline auto parse_schema(const fs::path &source_root) -> SourceSchema {
  const auto body = read_text(source_root / "schema.json");
  SourceSchema schema;
  schema.kind = json_string(body, "type");
  schema.scalar_name = json_string(body, "data_type");
  schema.id_type = json_string(body, "id_type");
  schema.has_scalar_data = json_bool(body, "has_scalar_data");
  if (json_string(body, "metric") != "l2" || json_string(body, "quantization_type") != "none") {
    throw std::invalid_argument("legacy importer supports only the pinned L2/raw corpus layout");
  }
  if (schema.kind != "index" && schema.kind != "collection") {
    throw std::invalid_argument("legacy schema type is not index or collection");
  }
  if (schema.scalar_name == "float32") {
    schema.scalar_type = core::ScalarType::float32;
  } else if (schema.scalar_name == "int8") {
    schema.scalar_type = core::ScalarType::int8;
  } else if (schema.scalar_name == "uint8") {
    schema.scalar_type = core::ScalarType::uint8;
  } else {
    throw std::invalid_argument("legacy schema data_type is outside the importer allowlist");
  }
  if (schema.id_type == "uint32") {
    schema.id_bytes = 4;
  } else if (schema.id_type == "uint64") {
    schema.id_bytes = 8;
  } else {
    throw std::invalid_argument("legacy schema id_type is outside the importer allowlist");
  }
  if ((schema.kind == "collection") != schema.has_scalar_data) {
    throw std::invalid_argument(
        "pinned legacy collection/index kind disagrees with scalar checkpoint presence");
  }
  return schema;
}

struct SnapshotManifest {
  std::string snapshot_id{};
  std::uint64_t applied_through{};
  fs::path snapshot_directory{};
  fs::path data_path{};
  fs::path rocksdb_path{};
};

[[nodiscard]] inline auto parse_current_snapshot(const fs::path &source_root) -> SnapshotManifest {
  const auto recovery = source_root / "recovery";
  auto current = trim(read_text(recovery / "CURRENT", 4096));
  if (current.empty() || current == "." || current == ".." || current.find('/') != current.npos ||
      current.find('\\') != current.npos) {
    throw std::invalid_argument("legacy recovery CURRENT is not a safe snapshot component");
  }
  SnapshotManifest result;
  result.snapshot_id = current;
  result.snapshot_directory = recovery / "snapshots" / current;
  const auto fields = parse_lines(read_text(result.snapshot_directory / "manifest.txt"));
  if (required_field(fields, "format_version") != "1" ||
      required_field(fields, "snapshot_id") != current) {
    throw std::invalid_argument("legacy snapshot manifest identity/version mismatch");
  }
  result.applied_through =
      parse_u64(required_field(fields, "applied_through_op_id"), "applied_through_op_id");
  const auto data_file = required_field(fields, "data_file");
  if (data_file.empty() || data_file.find('/') != data_file.npos ||
      data_file.find('\\') != data_file.npos) {
    throw std::invalid_argument("legacy snapshot data_file is not a safe component");
  }
  result.data_path = result.snapshot_directory / data_file;
  const auto rocksdb_dir = required_field(fields, "rocksdb_dir");
  if (!rocksdb_dir.empty()) {
    if (rocksdb_dir == "." || rocksdb_dir == ".." || rocksdb_dir.find('/') != rocksdb_dir.npos ||
        rocksdb_dir.find('\\') != rocksdb_dir.npos) {
      throw std::invalid_argument("legacy snapshot rocksdb_dir is not a safe component");
    }
    result.rocksdb_path = result.snapshot_directory / rocksdb_dir;
  }
  return result;
}

class ByteReader {
 public:
  explicit ByteReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  template <class T>
  [[nodiscard]] auto pod() -> T {
    if (remaining() < sizeof(T)) {
      throw std::invalid_argument("legacy snapshot is truncated");
    }
    T value{};
    std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  [[nodiscard]] auto unsigned_id(std::uint32_t width) -> std::uint64_t {
    if (width == 4) {
      return pod<std::uint32_t>();
    }
    if (width == 8) {
      return pod<std::uint64_t>();
    }
    throw std::invalid_argument("legacy snapshot ID width is unsupported");
  }

  void skip(std::uint64_t count) {
    if (count > remaining()) {
      throw std::invalid_argument("legacy snapshot length crosses EOF");
    }
    offset_ += static_cast<std::size_t>(count);
  }

  [[nodiscard]] auto bytes(std::size_t count) -> std::vector<std::byte> {
    if (count > remaining()) {
      throw std::invalid_argument("legacy snapshot vector crosses EOF");
    }
    std::vector<std::byte> result(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                                  bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
    offset_ += count;
    return result;
  }

  [[nodiscard]] auto offset() const noexcept -> std::size_t { return offset_; }
  [[nodiscard]] auto remaining() const noexcept -> std::size_t { return bytes_.size() - offset_; }

 private:
  std::span<const std::byte> bytes_{};
  std::size_t offset_{};
};

struct SnapshotVectorRow {
  std::uint64_t internal_id{};
  bool valid{};
  std::vector<std::byte> vector{};
};

struct SnapshotData {
  std::uint64_t allocated_rows{};
  std::vector<SnapshotVectorRow> rows{};
};

inline void skip_scalar_config(ByteReader &reader) {
  const auto path_size = reader.pod<std::uint64_t>();
  if (path_size > 1ULL << 20U) {
    throw std::invalid_argument("legacy scalar snapshot path is implausibly large");
  }
  reader.skip(path_size);
  (void)reader.pod<std::uint64_t>();  // write_buffer_size
  (void)reader.pod<std::int32_t>();   // max_write_buffer_number
  (void)reader.pod<std::uint64_t>();  // target_file_size_base
  (void)reader.pod<std::int32_t>();   // max_background_compactions
  (void)reader.pod<std::int32_t>();   // max_background_flushes
  (void)reader.pod<std::uint64_t>();  // block_cache_size_mb
  (void)reader.pod<std::uint8_t>();   // compression
  const auto fields = reader.pod<std::uint64_t>();
  if (fields > 1000) {
    throw std::invalid_argument("legacy scalar snapshot indexed field count is invalid");
  }
  for (std::uint64_t index = 0; index < fields; ++index) {
    const auto length = reader.pod<std::uint64_t>();
    if (length > 1ULL << 20U) {
      throw std::invalid_argument("legacy scalar snapshot indexed field is too large");
    }
    reader.skip(length);
  }
}

[[nodiscard]] inline auto parse_snapshot_data(const SnapshotManifest &manifest,
                                              SourceSchema &schema) -> SnapshotData {
  const auto bytes = read_bytes(manifest.data_path);
  ByteReader reader(bytes);
  const auto metric = reader.pod<std::int32_t>();
  const auto data_size = reader.pod<std::uint32_t>();
  schema.dim = reader.pod<std::uint32_t>();
  const auto item_count = reader.unsigned_id(schema.id_bytes);
  const auto delete_count = reader.unsigned_id(schema.id_bytes);
  const auto capacity_header = reader.unsigned_id(schema.id_bytes);
  if (metric != 0 || schema.dim == 0 ||
      data_size != schema.dim * core::scalar_type_size(schema.scalar_type) ||
      item_count > capacity_header || delete_count > item_count) {
    throw std::invalid_argument("legacy RawSpace header disagrees with the pinned schema");
  }
  if (schema.has_scalar_data) {
    skip_scalar_config(reader);
  }
  const auto item_size = reader.pod<std::uint64_t>();
  const auto aligned_item_size = reader.pod<std::uint64_t>();
  const auto capacity = reader.pod<std::uint64_t>();
  const auto position = reader.pod<std::uint64_t>();
  const auto alignment = reader.pod<std::uint64_t>();
  if (item_size != data_size || aligned_item_size < item_size || capacity != capacity_header ||
      position != item_count || position > capacity || alignment == 0 ||
      aligned_item_size > kMaximumSourceFileBytes ||
      capacity > kMaximumSourceFileBytes / aligned_item_size) {
    throw std::invalid_argument("legacy SequentialStorage header is invalid");
  }
  const auto data_offset = reader.offset();
  const auto storage_bytes = aligned_item_size * capacity;
  const auto bitmap_bytes = (capacity + 7U) / 8U;
  if (storage_bytes + bitmap_bytes > reader.remaining()) {
    throw std::invalid_argument("legacy SequentialStorage payload is truncated");
  }
  const auto bitmap_offset = data_offset + static_cast<std::size_t>(storage_bytes);
  SnapshotData result;
  result.allocated_rows = position;
  result.rows.reserve(static_cast<std::size_t>(position));
  for (std::uint64_t row = 0; row < position; ++row) {
    const auto vector_offset = data_offset + static_cast<std::size_t>(row * aligned_item_size);
    std::vector<std::byte> vector(static_cast<std::size_t>(item_size));
    std::memcpy(vector.data(), bytes.data() + vector_offset, vector.size());
    const auto bitmap = std::to_integer<std::uint8_t>(bytes[bitmap_offset + row / 8U]);
    result.rows.push_back({row, (bitmap & (1U << (row % 8U))) != 0, std::move(vector)});
  }
  return result;
}

using ScalarRows = std::map<std::uint64_t, ScalarData>;

[[nodiscard]] inline auto parse_data_key(std::string_view key) -> std::optional<std::uint64_t> {
  if (!key.starts_with("d_") || key.size() == 2) {
    return std::nullopt;
  }
  std::uint64_t id{};
  for (const char digit : key.substr(2)) {
    if (digit < '0' || digit > '9' ||
        id > (std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(digit - '0')) /
                 10U) {
      return std::nullopt;
    }
    id = id * 10U + static_cast<std::uint64_t>(digit - '0');
  }
  return id;
}

[[nodiscard]] inline auto read_rocksdb_checkpoint(const SnapshotManifest &manifest,
                                                  const SourceSchema &schema) -> ScalarRows {
  if (!schema.has_scalar_data) {
    if (!manifest.rocksdb_path.empty()) {
      throw std::invalid_argument("legacy scalar-off snapshot unexpectedly names RocksDB");
    }
    return {};
  }
  if (manifest.rocksdb_path.empty() || !fs::is_directory(manifest.rocksdb_path)) {
    throw std::invalid_argument("legacy scalar snapshot has no RocksDB checkpoint directory");
  }
  rocksdb::Options options;
  options.create_if_missing = false;
  rocksdb::DB *raw = nullptr;
  const auto status = rocksdb::DB::OpenForReadOnly(options, manifest.rocksdb_path.string(), &raw);
  if (!status.ok()) {
    throw std::runtime_error("cannot open legacy RocksDB checkpoint read-only: " +
                             status.ToString());
  }
  std::unique_ptr<rocksdb::DB> database(raw);
  std::unique_ptr<rocksdb::Iterator> iterator(database->NewIterator(rocksdb::ReadOptions{}));
  ScalarRows rows;
  for (iterator->Seek("d_"); iterator->Valid(); iterator->Next()) {
    const auto key = iterator->key().ToStringView();
    if (!key.starts_with("d_")) {
      break;
    }
    const auto id = parse_data_key(key);
    if (!id.has_value()) {
      throw std::invalid_argument("legacy RocksDB contains a malformed primary data key");
    }
    const auto value = iterator->value();
    auto scalar = ScalarData::deserialize(value.data(), value.size());
    auto found = rows.find(*id);
    if (found != rows.end() && found->second.item_id != scalar.item_id) {
      throw std::invalid_argument("legacy RocksDB contains conflicting duplicate data keys");
    }
    rows.insert_or_assign(*id, std::move(scalar));
  }
  if (!iterator->status().ok()) {
    throw std::runtime_error("legacy RocksDB iterator failed: " + iterator->status().ToString());
  }
  return rows;
}

struct LegacyWalRecord {
  enum class Mutation : std::uint8_t {
    insert = 1,
    upsert = 2,
    remove_by_item_id = 3,
    remove_by_internal_id = 4,
  };

  std::uint64_t op_id{};
  Mutation mutation{Mutation::insert};
  std::vector<std::byte> payload{};
};

struct LegacyWalScan {
  std::vector<LegacyWalRecord> committed{};
  std::uint64_t maximum_seen_op_id{};
  std::uint64_t verified_bytes{};
  bool torn_tail{};
};

template <class T>
[[nodiscard]] inline auto load_pod(std::span<const std::byte> bytes, std::size_t offset) -> T {
  if (offset > bytes.size() || bytes.size() - offset < sizeof(T)) {
    throw std::invalid_argument("legacy WAL field crosses EOF");
  }
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

[[nodiscard]] inline auto decode_wal(const fs::path &wal_path,
                                     std::uint64_t applied_through,
                                     bool inject_during_decode) -> LegacyWalScan {
  LegacyWalScan scan;
  scan.maximum_seen_op_id = applied_through;
  if (!fs::exists(wal_path)) {
    return scan;
  }
  const auto bytes = read_bytes(wal_path);
  constexpr std::size_t kHeaderBytes = 24;
  constexpr std::size_t kTrailerBytes = 4;
  std::map<std::uint64_t, LegacyWalRecord> pending;
  std::size_t offset{};
  std::size_t frame_count{};
  while (offset != bytes.size()) {
    const auto frame_start = offset;
    if (bytes.size() - offset < kHeaderBytes) {
      scan.torn_tail = true;
      break;
    }
    const auto magic = load_pod<std::uint32_t>(bytes, offset);
    const auto version = load_pod<std::uint8_t>(bytes, offset + 4);
    const auto raw_frame = load_pod<std::uint8_t>(bytes, offset + 5);
    const auto raw_mutation = load_pod<std::uint8_t>(bytes, offset + 6);
    const auto reserved = load_pod<std::uint8_t>(bytes, offset + 7);
    const auto op_id = load_pod<std::uint64_t>(bytes, offset + 8);
    const auto payload_size = load_pod<std::uint64_t>(bytes, offset + 16);
    const auto frame_valid = raw_frame == 1 || raw_frame == 2;
    const auto mutation_valid = raw_mutation >= 1 && raw_mutation <= 4;
    constexpr std::uint32_t kFrameMagic = 0x48454144U;    // "HEAD"
    constexpr std::uint32_t kTrailerMagic = 0x5441494CU;  // "TAIL"
    constexpr std::uint64_t kMaximumPayload = 1ULL << 30U;
    if (magic != kFrameMagic || version != 1 || reserved != 0 || !frame_valid || !mutation_valid ||
        payload_size > kMaximumPayload || payload_size > bytes.size() - offset - kHeaderBytes ||
        bytes.size() - offset - kHeaderBytes - static_cast<std::size_t>(payload_size) <
            kTrailerBytes) {
      scan.torn_tail = true;
      break;
    }
    const auto payload_offset = offset + kHeaderBytes;
    const auto trailer_offset = payload_offset + static_cast<std::size_t>(payload_size);
    if (load_pod<std::uint32_t>(bytes, trailer_offset) != kTrailerMagic) {
      scan.torn_tail = true;
      break;
    }
    offset = trailer_offset + kTrailerBytes;
    scan.verified_bytes = offset;
    scan.maximum_seen_op_id = std::max(scan.maximum_seen_op_id, op_id);
    const auto mutation = static_cast<LegacyWalRecord::Mutation>(raw_mutation);
    if (raw_frame == 1) {
      std::vector<std::byte> payload(static_cast<std::size_t>(payload_size));
      std::memcpy(payload.data(), bytes.data() + payload_offset, payload.size());
      pending.insert_or_assign(op_id, LegacyWalRecord{op_id, mutation, std::move(payload)});
    } else {
      const auto found = pending.find(op_id);
      if (found != pending.end() && found->second.mutation == mutation) {
        if (op_id > applied_through) {
          scan.committed.push_back(std::move(found->second));
        }
        pending.erase(found);
      }
    }
    ++frame_count;
    if (inject_during_decode && frame_count == 1) {
      throw std::runtime_error("injected legacy importer failure during WAL decode");
    }
    if (offset <= frame_start) {
      throw std::logic_error("legacy WAL decoder made no progress");
    }
  }
  std::ranges::sort(scan.committed, [](const auto &left, const auto &right) {
    return left.op_id < right.op_id;
  });
  return scan;
}

[[nodiscard]] inline auto convert_metadata(const MetadataMap &source) -> Metadata {
  Metadata result;
  for (const auto &[key, value] : source) {
    std::visit(
        [&](const auto &typed) {
          using T = std::decay_t<decltype(typed)>;
          if constexpr (std::is_same_v<T, std::int64_t>) {
            result.emplace(key, ScalarValue{typed});
          } else if constexpr (std::is_same_v<T, double>) {
            result.emplace(key, ScalarValue{typed});
          } else if constexpr (std::is_same_v<T, std::string>) {
            result.emplace(key, ScalarValue{typed});
          } else if constexpr (std::is_same_v<T, bool>) {
            result.emplace(key, ScalarValue{typed});
          }
        },
        value);
  }
  return result;
}

struct ImportedRow {
  core::LogicalId logical_id{};
  std::string item_id{};
  std::uint64_t internal_id{};
  std::uint64_t sequence{};
  VersionState state{VersionState::live};
  std::vector<std::byte> vector{};
  std::optional<ScalarData> scalar{};
};

[[nodiscard]] inline auto logical_id_for(const SourceSchema &schema,
                                         std::uint64_t internal_id,
                                         const std::optional<ScalarData> &scalar)
    -> core::LogicalId {
  if (schema.kind == "collection") {
    if (!scalar.has_value() || scalar->item_id.empty()) {
      throw std::invalid_argument("legacy collection live row has no scalar item_id");
    }
    return core::LogicalId::from_utf8(scalar->item_id);
  }
  return core::LogicalId::from_legacy_uint64(internal_id);
}

[[nodiscard]] inline auto snapshot_rows(const SourceSchema &schema,
                                        const SnapshotManifest &manifest,
                                        const SnapshotData &snapshot,
                                        const ScalarRows &scalars) -> std::vector<ImportedRow> {
  std::vector<ImportedRow> rows;
  rows.reserve(snapshot.rows.size());
  const auto inferred_first_mutation =
      schema.kind == "index" && manifest.applied_through <= snapshot.allocated_rows
          ? snapshot.allocated_rows - manifest.applied_through
          : snapshot.allocated_rows;
  for (const auto &source : snapshot.rows) {
    std::optional<ScalarData> scalar;
    const auto scalar_found = scalars.find(source.internal_id);
    if (scalar_found != scalars.end()) {
      scalar = scalar_found->second;
    }
    if (!source.valid && schema.kind == "collection") {
      // RocksDB deletion intentionally removes the only legacy external-ID
      // mapping. The committed visibility floor preserves the delete cut; no
      // synthetic logical ID is fabricated.
      continue;
    }
    ImportedRow row;
    row.internal_id = source.internal_id;
    row.vector = source.vector;
    row.scalar = scalar;
    row.logical_id = logical_id_for(schema, source.internal_id, scalar);
    row.item_id = scalar.has_value() ? scalar->item_id : std::string{};
    row.state = source.valid ? VersionState::live : VersionState::tombstone;
    if (schema.kind == "index" && source.internal_id >= inferred_first_mutation &&
        manifest.applied_through != 0) {
      row.sequence = source.internal_id - inferred_first_mutation + 1;
    } else if (!source.valid) {
      row.sequence = manifest.applied_through;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

struct InsertPayload {
  std::vector<std::byte> vector{};
  std::optional<ScalarData> scalar{};
};

[[nodiscard]] inline auto decode_insert_payload(const LegacyWalRecord &record,
                                                const SourceSchema &schema) -> InsertPayload {
  binary_io::BinaryReader reader(reinterpret_cast<const char *>(record.payload.data()),
                                 record.payload.size());
  const auto ef = reader.read_u32();
  const auto vector = reader.read_blob();
  const auto scalar = reader.read_blob();
  const auto expected =
      static_cast<std::size_t>(schema.dim) * core::scalar_type_size(schema.scalar_type);
  if (!ef.has_value() || !vector.has_value() || !scalar.has_value() || reader.remaining() != 0 ||
      vector->size() != expected) {
    throw std::invalid_argument("legacy WAL insert/upsert payload is malformed");
  }
  InsertPayload result;
  result.vector.resize(vector->size());
  std::memcpy(result.vector.data(), vector->data(), vector->size());
  if (!scalar->empty()) {
    auto decoded = ScalarData::deserialize(scalar->data(), scalar->size());
    if (schema.has_scalar_data) {
      result.scalar = std::move(decoded);
    } else if (!decoded.item_id.empty() || !decoded.document.empty() || !decoded.metadata.empty()) {
      throw std::invalid_argument("legacy scalar-off WAL carries non-empty scalar data");
    }
  }
  if (schema.has_scalar_data && !result.scalar.has_value()) {
    throw std::invalid_argument("legacy scalar-on WAL lacks scalar data");
  }
  return result;
}

inline void apply_wal(const SourceSchema &schema,
                      const LegacyWalScan &scan,
                      std::uint64_t &allocated_rows,
                      std::vector<ImportedRow> &rows) {
  auto find_numeric = [&](std::uint64_t id) {
    return std::find_if(rows.begin(), rows.end(), [&](const ImportedRow &row) {
      return row.internal_id == id;
    });
  };
  auto find_item = [&](std::string_view item_id) {
    return std::find_if(rows.begin(), rows.end(), [&](const ImportedRow &row) {
      return row.item_id == item_id && row.state == VersionState::live;
    });
  };
  for (const auto &record : scan.committed) {
    switch (record.mutation) {
      case LegacyWalRecord::Mutation::insert:
      case LegacyWalRecord::Mutation::upsert: {
        auto payload = decode_insert_payload(record, schema);
        if (record.mutation == LegacyWalRecord::Mutation::upsert) {
          if (!payload.scalar.has_value()) {
            throw std::invalid_argument("legacy upsert lacks its external item ID");
          }
          const auto previous = find_item(payload.scalar->item_id);
          if (previous != rows.end()) {
            rows.erase(previous);
          }
        }
        ImportedRow row;
        row.internal_id = allocated_rows++;
        row.sequence = record.op_id;
        row.vector = std::move(payload.vector);
        row.scalar = std::move(payload.scalar);
        row.logical_id = logical_id_for(schema, row.internal_id, row.scalar);
        row.item_id = row.scalar.has_value() ? row.scalar->item_id : std::string{};
        rows.push_back(std::move(row));
        break;
      }
      case LegacyWalRecord::Mutation::remove_by_internal_id: {
        binary_io::BinaryReader reader(reinterpret_cast<const char *>(record.payload.data()),
                                       record.payload.size());
        const auto id = reader.read_u64();
        if (!id.has_value() || reader.remaining() != 0) {
          throw std::invalid_argument("legacy remove-by-internal-id payload is malformed");
        }
        const auto found = find_numeric(*id);
        if (found != rows.end()) {
          found->state = VersionState::tombstone;
          found->sequence = record.op_id;
        }
        break;
      }
      case LegacyWalRecord::Mutation::remove_by_item_id: {
        binary_io::BinaryReader reader(reinterpret_cast<const char *>(record.payload.data()),
                                       record.payload.size());
        const auto item_id = reader.read_string();
        if (!item_id.has_value() || reader.remaining() != 0) {
          throw std::invalid_argument("legacy remove-by-item-id payload is malformed");
        }
        const auto found = find_item(*item_id);
        if (found != rows.end()) {
          found->state = VersionState::tombstone;
          found->sequence = record.op_id;
        }
        break;
      }
    }
  }
}

[[nodiscard]] inline auto owned_vector(const ImportedRow &row, const SourceSchema &schema)
    -> OwnedVector {
  core::TypedTensorView view(row.vector.data(),
                             schema.scalar_type,
                             1,
                             schema.dim,
                             row.vector.size());
  auto copied = OwnedVector::copy_row(view, 0);
  if (!copied.ok()) {
    throw std::invalid_argument(copied.status().diagnostic());
  }
  return std::move(copied).value();
}

[[nodiscard]] inline auto to_registered_rows(const std::vector<ImportedRow> &rows,
                                             const SourceSchema &schema)
    -> std::vector<RegisteredRow> {
  std::vector<RegisteredRow> registered;
  registered.reserve(rows.size());
  for (const auto &row : rows) {
    RecordPayload payload;
    payload.vector = owned_vector(row, schema);
    if (row.scalar.has_value()) {
      payload.document = row.scalar->document;
      payload.metadata = convert_metadata(row.scalar->metadata);
    }
    registered.push_back({row.logical_id,
                          core::SegmentRowId(row.internal_id),
                          row.sequence,
                          row.state,
                          std::move(payload)});
  }
  return registered;
}

[[nodiscard]] inline auto flat_vectors(const std::vector<ImportedRow> &rows,
                                       const SourceSchema &schema,
                                       std::vector<std::uint64_t> &labels) -> std::vector<float> {
  std::vector<float> output;
  const auto live = std::count_if(rows.begin(), rows.end(), [](const auto &row) {
    return row.state == VersionState::live;
  });
  output.reserve(static_cast<std::size_t>(live) * schema.dim);
  labels.reserve(static_cast<std::size_t>(live));
  for (const auto &row : rows) {
    if (row.state != VersionState::live) {
      continue;
    }
    labels.push_back(row.internal_id);
    for (std::uint32_t column = 0; column < schema.dim; ++column) {
      switch (schema.scalar_type) {
        case core::ScalarType::float32: {
          float value{};
          std::memcpy(&value, row.vector.data() + column * sizeof(float), sizeof(value));
          output.push_back(value);
          break;
        }
        case core::ScalarType::int8:
          output.push_back(static_cast<float>(
              std::bit_cast<std::int8_t>(std::to_integer<std::uint8_t>(row.vector[column]))));
          break;
        case core::ScalarType::uint8:
          output.push_back(static_cast<float>(std::to_integer<std::uint8_t>(row.vector[column])));
          break;
      }
    }
  }
  return output;
}

class SourceTypedFlatSegment {
 public:
  SourceTypedFlatSegment(std::shared_ptr<::alaya::disk::DiskFlatSegment> flat,
                         core::ScalarType scalar_type)
      : flat_(std::move(flat)), scalar_type_(scalar_type) {}

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    auto descriptor = flat_->descriptor();
    descriptor.stored_scalar_type = scalar_type_;
    return descriptor;
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::invalid_argument,
                     "legacy Flat adapter single search requires one query");
    }
    return execute(request, false);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute(request, true);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    return flat_->stats(stats);
  }

 private:
  [[nodiscard]] auto execute(const core::SearchRequest &request, bool batch) const -> core::Status {
    auto status =
        core::validate_tensor(request.queries, descriptor().dim, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != scalar_type_) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::not_supported,
                     "legacy Flat adapter query dtype disagrees with imported schema",
                     core::StatusDetail::unsupported_scalar_type);
    }
    std::vector<float> converted(static_cast<std::size_t>(request.queries.rows) *
                                 request.queries.dim);
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      for (std::uint32_t column = 0; column < request.queries.dim; ++column) {
        const auto output = static_cast<std::size_t>(row * request.queries.dim + column);
        switch (scalar_type_) {
          case core::ScalarType::float32:
            converted[output] = request.queries.row<float>(row)[column];
            break;
          case core::ScalarType::int8:
            converted[output] = static_cast<float>(request.queries.row<std::int8_t>(row)[column]);
            break;
          case core::ScalarType::uint8:
            converted[output] = static_cast<float>(request.queries.row<std::uint8_t>(row)[column]);
            break;
        }
      }
    }
    auto delegated = request;
    delegated.queries = core::TypedTensorView::contiguous(converted.data(),
                                                          request.queries.rows,
                                                          request.queries.dim);
    return batch ? flat_->batch_search(delegated) : flat_->search(delegated);
  }

  std::shared_ptr<::alaya::disk::DiskFlatSegment> flat_{};
  core::ScalarType scalar_type_{core::ScalarType::float32};
};

[[nodiscard]] inline auto erase_flat(std::unique_ptr<::alaya::disk::DiskFlatSegment> flat,
                                     core::ScalarType scalar_type)
    -> core::Result<core::AnySegment> {
  auto shared_flat = std::shared_ptr<::alaya::disk::DiskFlatSegment>(std::move(flat));
  auto adapter = std::make_shared<SourceTypedFlatSegment>(std::move(shared_flat), scalar_type);
  core::SegmentInstanceConfig config;
  config.readonly = true;
  config.concurrency.reentrant_search = true;
  config.concurrency.search_with_stage = false;
  config.concurrency.search_with_publish = false;
  config.concurrency.serial_mutation = true;
  config.concurrency.native_async = false;
  config.concurrency.cooperative_cancel = true;
  config.concurrency.explicit_drain = false;
  return core::AnySegment::from_sync(std::move(adapter), std::move(config));
}

struct SourceFingerprint {
  std::string digest{};
  std::uint64_t files{};
  std::uint64_t bytes{};
};

[[nodiscard]] inline auto output_relative_path(std::string_view relative) -> bool {
  return relative == kCollectionManifestFilename || relative == "segments" ||
         relative.starts_with("segments/") || relative == ".alaya_internal" ||
         relative.starts_with(".alaya_internal/");
}

[[nodiscard]] inline auto fingerprint_source(const fs::path &source_root,
                                             const fs::path &target_root) -> SourceFingerprint {
  if (!fs::is_directory(source_root)) {
    throw std::invalid_argument("legacy source root is not a directory");
  }
  const auto same_root = fs::weakly_canonical(source_root) == fs::weakly_canonical(target_root);
  std::vector<fs::path> files;
  for (const auto &entry : fs::recursive_directory_iterator(source_root)) {
    std::error_code ec;
    if (entry.is_symlink(ec)) {
      throw std::invalid_argument("legacy source tree contains a symlink: " +
                                  entry.path().string());
    }
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    const auto relative = fs::relative(entry.path(), source_root).generic_string();
    if (same_root && output_relative_path(relative)) {
      continue;
    }
    files.push_back(entry.path());
  }
  std::ranges::sort(files, [&](const fs::path &left, const fs::path &right) {
    return fs::relative(left, source_root).generic_string() <
           fs::relative(right, source_root).generic_string();
  });
  Sha256 hasher;
  SourceFingerprint result;
  for (const auto &path : files) {
    const auto relative = fs::relative(path, source_root).generic_string();
    const auto size = static_cast<std::uint64_t>(fs::file_size(path));
    const auto digest = sha256_file(path).hex();
    const auto record = relative + "\0" + std::to_string(size) + "\0" + digest + "\n";
    hasher.update(record);
    ++result.files;
    if (size > std::numeric_limits<std::uint64_t>::max() - result.bytes) {
      throw std::overflow_error("legacy source byte accounting overflow");
    }
    result.bytes += size;
  }
  result.digest = hasher.finalize().hex();
  return result;
}

[[nodiscard]] inline auto import_directory(const fs::path &target_root) -> fs::path {
  return target_root / ".alaya_internal" / kLegacyImportNamespace;
}

inline void atomic_write(const fs::path &path, std::string_view body) {
  fs::create_directories(path.parent_path());
  const auto temporary = path.string() + ".tmp";
  platform::write_all_fsync(temporary, body.data(), body.size());
  platform::atomic_replace(temporary, path);
  platform::sync_directory_or_throw(path.parent_path());
}

inline void cleanup_incomplete(const fs::path &target_root) {
  std::error_code ec;
  fs::remove_all(import_directory(target_root), ec);
  ec.clear();
  fs::remove_all(target_root / ".alaya_internal" / kCollectionWalNamespace, ec);
  ec.clear();
  fs::remove_all(target_root / "segments" / kSegmentName, ec);
  ec.clear();
  fs::remove(target_root / kCollectionManifestFilename, ec);
  ec.clear();
  fs::remove_all(target_root / ".alaya_staging", ec);
}

[[nodiscard]] inline auto has_import_intent(const fs::path &target_root) -> bool {
  return fs::is_regular_file(import_directory(target_root) / kLegacyImportIntentFilename);
}

[[nodiscard]] inline auto has_import_owned_output(const fs::path &target_root) -> bool {
  return fs::exists(target_root / kCollectionManifestFilename) ||
         fs::exists(target_root / "segments" / kSegmentName) ||
         fs::exists(target_root / ".alaya_internal" / kCollectionWalNamespace) ||
         fs::exists(import_directory(target_root));
}

[[nodiscard]] inline auto serialize_intent(const SourceFingerprint &fingerprint,
                                           const SourceSchema &schema,
                                           const SnapshotManifest &snapshot,
                                           const LegacyWalScan &wal,
                                           std::uint64_t minimum_next_op_id) -> std::string {
  return "format=1\nstate=prepared\nsource_fingerprint=" + fingerprint.digest +
         "\nsource_kind=" + schema.kind + "\nsource_dtype=" + schema.scalar_name +
         "\nsource_id_type=" + schema.id_type + "\nsnapshot_id=" + snapshot.snapshot_id +
         "\nsnapshot_applied=" + std::to_string(snapshot.applied_through) +
         "\nmaximum_seen_op_id=" + std::to_string(wal.maximum_seen_op_id) +
         "\nminimum_next_op_id=" + std::to_string(minimum_next_op_id) + "\n";
}

[[nodiscard]] inline auto scalar_name(core::ScalarType type) -> std::string_view {
  switch (type) {
    case core::ScalarType::float32:
      return "float32";
    case core::ScalarType::int8:
      return "int8";
    case core::ScalarType::uint8:
      return "uint8";
  }
  return "unknown";
}

[[nodiscard]] inline auto serialize_audit(const LegacyImportAudit &audit) -> std::string {
  return "format=" + std::to_string(audit.format_version) +
         "\nstate=ready\nsource_fingerprint=" + audit.source_fingerprint +
         "\nsource_file_count=" + std::to_string(audit.source_file_count) +
         "\nsource_bytes=" + std::to_string(audit.source_bytes) +
         "\nsource_kind=" + audit.source_kind +
         "\nsource_dtype=" + std::string(scalar_name(audit.source_scalar_type)) +
         "\nsource_id_type=" + audit.source_id_type + "\ndim=" + std::to_string(audit.dim) +
         "\nhas_scalar_data=" + (audit.has_scalar_data ? "1" : "0") +
         "\nsnapshot_id=" + audit.snapshot_id +
         "\nsnapshot_applied_through=" + std::to_string(audit.snapshot_applied_through) +
         "\nmaximum_seen_op_id=" + std::to_string(audit.maximum_seen_op_id) +
         "\nmaximum_committed_op_id=" + std::to_string(audit.maximum_committed_op_id) +
         "\nminimum_next_op_id=" + std::to_string(audit.minimum_next_op_id) +
         "\nallocated_rows=" + std::to_string(audit.allocated_rows) +
         "\nlive_rows=" + std::to_string(audit.live_rows) +
         "\ntombstone_rows=" + std::to_string(audit.tombstone_rows) +
         "\ncommitted_wal_records=" + std::to_string(audit.committed_wal_records) +
         "\ntorn_wal_tail=" + (audit.torn_wal_tail ? "1" : "0") +
         "\nverified_wal_bytes=" + std::to_string(audit.verified_wal_bytes) +
         "\nrocksdb_mode=" + audit.rocksdb_mode + "\ncheckpoint_name=" + audit.checkpoint_name +
         "\nwal_cut=" + std::to_string(audit.wal_cut) +
         "\nmanifest_sha256=" + audit.manifest_sha256 + "\nsegment_id=" + audit.segment_id + "\n";
}

[[nodiscard]] inline auto parse_scalar_type(std::string_view value) -> core::ScalarType {
  if (value == "float32") {
    return core::ScalarType::float32;
  }
  if (value == "int8") {
    return core::ScalarType::int8;
  }
  if (value == "uint8") {
    return core::ScalarType::uint8;
  }
  throw std::invalid_argument("legacy import audit contains an unsupported dtype");
}

[[nodiscard]] inline auto parse_audit(const fs::path &path) -> LegacyImportAudit {
  const auto fields = parse_lines(read_text(path));
  LegacyImportAudit audit;
  audit.format_version =
      static_cast<std::uint32_t>(parse_u64(required_field(fields, "format"), "audit format"));
  if (audit.format_version != 1 || required_field(fields, "state") != "ready") {
    throw std::invalid_argument("legacy import audit version/state is invalid");
  }
  audit.source_fingerprint = required_field(fields, "source_fingerprint");
  audit.source_file_count = parse_u64(required_field(fields, "source_file_count"), "source files");
  audit.source_bytes = parse_u64(required_field(fields, "source_bytes"), "source bytes");
  audit.source_kind = required_field(fields, "source_kind");
  audit.source_scalar_type = parse_scalar_type(required_field(fields, "source_dtype"));
  audit.source_id_type = required_field(fields, "source_id_type");
  audit.dim = static_cast<std::uint32_t>(parse_u64(required_field(fields, "dim"), "dimension"));
  audit.has_scalar_data = required_field(fields, "has_scalar_data") == "1";
  audit.snapshot_id = required_field(fields, "snapshot_id");
  audit.snapshot_applied_through =
      parse_u64(required_field(fields, "snapshot_applied_through"), "snapshot cut");
  audit.maximum_seen_op_id =
      parse_u64(required_field(fields, "maximum_seen_op_id"), "maximum seen op id");
  audit.maximum_committed_op_id =
      parse_u64(required_field(fields, "maximum_committed_op_id"), "maximum committed op id");
  audit.minimum_next_op_id =
      parse_u64(required_field(fields, "minimum_next_op_id"), "minimum next op id");
  audit.allocated_rows = parse_u64(required_field(fields, "allocated_rows"), "allocated rows");
  audit.live_rows = parse_u64(required_field(fields, "live_rows"), "live rows");
  audit.tombstone_rows = parse_u64(required_field(fields, "tombstone_rows"), "tombstone rows");
  audit.committed_wal_records =
      parse_u64(required_field(fields, "committed_wal_records"), "committed WAL records");
  audit.torn_wal_tail = required_field(fields, "torn_wal_tail") == "1";
  audit.verified_wal_bytes =
      parse_u64(required_field(fields, "verified_wal_bytes"), "verified WAL bytes");
  audit.rocksdb_mode = required_field(fields, "rocksdb_mode");
  audit.checkpoint_name = required_field(fields, "checkpoint_name");
  audit.wal_cut = parse_u64(required_field(fields, "wal_cut"), "WAL cut");
  audit.manifest_sha256 = required_field(fields, "manifest_sha256");
  audit.segment_id = required_field(fields, "segment_id");
  if (audit.source_fingerprint.size() != 64 || audit.manifest_sha256.size() != 64 ||
      audit.dim == 0 || audit.minimum_next_op_id == 0 ||
      (audit.source_kind != "index" && audit.source_kind != "collection") ||
      audit.segment_id != kSegmentName) {
    throw std::invalid_argument("legacy import audit invariant is invalid");
  }
  return audit;
}

[[nodiscard]] inline auto marker_prefix(std::string_view source_fingerprint,
                                        std::string_view audit_sha) -> std::string {
  return "format=1\nstate=active\nsource_fingerprint=" + std::string(source_fingerprint) +
         "\naudit_sha256=" + std::string(audit_sha) + "\n";
}

[[nodiscard]] inline auto valid_marker(const fs::path &target_root) -> bool {
  try {
    const auto directory = import_directory(target_root);
    const auto marker_path = directory / kLegacyImportMarkerFilename;
    const auto audit_path = directory / kLegacyImportAuditFilename;
    if (!fs::is_regular_file(marker_path) || !fs::is_regular_file(audit_path)) {
      return false;
    }
    const auto fields = parse_lines(read_text(marker_path, 4096));
    if (fields.size() != 5 || required_field(fields, "format") != "1" ||
        required_field(fields, "state") != "active") {
      return false;
    }
    const auto &source = required_field(fields, "source_fingerprint");
    const auto &audit_sha = required_field(fields, "audit_sha256");
    const auto expected = sha256(marker_prefix(source, audit_sha)).hex();
    if (required_field(fields, "checksum") != expected ||
        sha256_file(audit_path).hex() != audit_sha) {
      return false;
    }
    const auto audit = parse_audit(audit_path);
    return audit.source_fingerprint == source &&
           sha256_file(target_root / kCollectionManifestFilename).hex() == audit.manifest_sha256;
  } catch (...) {
    return false;
  }
}

[[nodiscard]] inline auto open_imported(
    const fs::path &target_root,
    const LegacyImportOptions::ActiveRegistrationFactory &active_registration_factory = {})
    -> core::Result<LegacyImportResult> {
  if (!valid_marker(target_root)) {
    return failure(core::OperationStage::open,
                   core::StatusCode::not_found,
                   "legacy import marker is absent, partial, or invalid",
                   core::StatusDetail::none);
  }
  try {
    auto audit = parse_audit(import_directory(target_root) / kLegacyImportAuditFilename);
    core::OpenContext open_context;
    auto flat = ::alaya::disk::DiskFlatSegment::open_collection(target_root,
                                                                audit.segment_id,
                                                                {},
                                                                open_context);
    if (!flat.ok()) {
      return flat.status();
    }
    auto erased = erase_flat(std::move(flat).value(), audit.source_scalar_type);
    if (!erased.ok()) {
      return erased.status();
    }
    SegmentRegistration registration;
    registration.segment_id = kSegmentId;
    registration.generation = kSegmentGeneration;
    registration.role = SegmentRole::sealed;
    registration.segment = std::move(erased).value();
    CollectionConfig config;
    config.features.wal_coordinator = true;
    config.wal.root = target_root;
    config.recovery.minimum_next_op_id = audit.minimum_next_op_id;
    config.recovery.minimum_visibility_watermark = audit.maximum_committed_op_id;
    CollectionSchema schema{audit.dim, core::Metric::l2, audit.source_scalar_type};
    std::vector<SegmentRegistration> registrations;
    registrations.push_back(std::move(registration));
    if (active_registration_factory) {
      auto active = active_registration_factory(schema);
      if (!active.ok()) {
        return active.status();
      }
      registrations.push_back(std::move(active).value());
    }
    auto collection =
        SegmentedCollection::open(schema, std::move(registrations), std::move(config));
    if (!collection.ok()) {
      return collection.status();
    }
    return LegacyImportResult{std::move(collection).value(), std::move(audit), true};
  } catch (const std::invalid_argument &error) {
    return failure(core::OperationStage::open, core::StatusCode::corruption, error.what());
  } catch (const std::exception &error) {
    return failure(core::OperationStage::open,
                   core::StatusCode::io_error,
                   error.what(),
                   core::StatusDetail::none);
  } catch (...) {
    return core::status_from_exception(core::OperationStage::open);
  }
}

}  // namespace legacy_import_detail

class LegacyImporter {
 public:
  [[nodiscard]] static auto import(const LegacyImportOptions &options)
      -> core::Result<LegacyImportResult> {
    namespace detail = legacy_import_detail;
    if (options.target_root.empty()) {
      return detail::failure(core::OperationStage::save,
                             core::StatusCode::invalid_argument,
                             "legacy importer target root is empty");
    }
    if (detail::valid_marker(options.target_root)) {
      return detail::open_imported(options.target_root, options.active_registration_factory);
    }
    if (!options.features.legacy_importer) {
      return detail::failure(core::OperationStage::save,
                             core::StatusCode::not_supported,
                             "legacy importer feature gate is disabled",
                             core::StatusDetail::operation_slot_absent);
    }
    if (options.source_root.empty()) {
      return detail::failure(core::OperationStage::save,
                             core::StatusCode::invalid_argument,
                             "legacy importer source root is empty");
    }

    if (detail::has_import_owned_output(options.target_root) &&
        !detail::has_import_intent(options.target_root)) {
      return detail::failure(core::OperationStage::save,
                             core::StatusCode::conflict,
                             "legacy importer target contains unowned collection output",
                             core::StatusDetail::already_exists);
    }
    detail::cleanup_incomplete(options.target_root);
    try {
      const auto before = detail::fingerprint_source(options.source_root, options.target_root);
      auto schema = detail::parse_schema(options.source_root);
      const auto snapshot_manifest = detail::parse_current_snapshot(options.source_root);
      const auto snapshot = detail::parse_snapshot_data(snapshot_manifest, schema);
      const auto scalars = detail::read_rocksdb_checkpoint(snapshot_manifest, schema);
      if (options.fail_point == LegacyImporterFailPoint::after_rocksdb_validation) {
        return detail::injected(options.fail_point);
      }
      const auto wal =
          detail::decode_wal(options.source_root / "recovery" / "wal.bin",
                             snapshot_manifest.applied_through,
                             options.fail_point == LegacyImporterFailPoint::during_wal_decode);
      auto rows = detail::snapshot_rows(schema, snapshot_manifest, snapshot, scalars);
      auto allocated_rows = snapshot.allocated_rows;
      detail::apply_wal(schema, wal, allocated_rows, rows);
      const auto maximum_committed =
          wal.committed.empty()
              ? snapshot_manifest.applied_through
              : std::max(snapshot_manifest.applied_through, wal.committed.back().op_id);
      if (wal.maximum_seen_op_id == std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument("legacy source op-id range has no representable successor");
      }
      const auto minimum_next_op_id = wal.maximum_seen_op_id + 1;
      if (options.fail_point == LegacyImporterFailPoint::after_op_id_reservation) {
        return detail::injected(options.fail_point);
      }
      const auto import_dir = detail::import_directory(options.target_root);
      detail::atomic_write(import_dir / kLegacyImportIntentFilename,
                           detail::serialize_intent(before,
                                                    schema,
                                                    snapshot_manifest,
                                                    wal,
                                                    minimum_next_op_id));
      if (options.fail_point == LegacyImporterFailPoint::after_intent_write) {
        return detail::injected(options.fail_point);
      }

      std::vector<std::uint64_t> labels;
      auto vectors = detail::flat_vectors(rows, schema, labels);
      if (labels.empty()) {
        throw std::invalid_argument("legacy importer cannot materialize an empty DiskFlat segment");
      }
      ::alaya::disk::DiskFlatBuildInput
          build_input{core::TypedTensorView::contiguous(vectors.data(), labels.size(), schema.dim),
                      labels};
      ::alaya::disk::DiskFlatPublicationOptions publication;
      publication.collection_root = options.target_root;
      publication.segment_id = std::string(detail::kSegmentName);
      publication.segment_generation = detail::kSegmentGeneration;
      publication.collection_features.manifest_v2_writer = true;
      publication.row_versions = {0, maximum_committed};
      core::BuildContext build_context;
      auto flat = ::alaya::disk::DiskFlatSegment::build(build_input,
                                                        core::Metric::l2,
                                                        publication,
                                                        build_context);
      if (!flat.ok()) {
        return flat.status();
      }
      auto erased = detail::erase_flat(std::move(flat).value(), schema.scalar_type);
      if (!erased.ok()) {
        return erased.status();
      }
      SegmentRegistration registration;
      registration.segment_id = detail::kSegmentId;
      registration.generation = detail::kSegmentGeneration;
      registration.role = SegmentRole::sealed;
      registration.segment = std::move(erased).value();
      registration.rows = detail::to_registered_rows(rows, schema);
      registration.next_row_id = allocated_rows;
      CollectionConfig config;
      config.features.wal_coordinator = true;
      config.wal.root = options.target_root;
      config.recovery.minimum_next_op_id = minimum_next_op_id;
      config.recovery.minimum_visibility_watermark = maximum_committed;
      CollectionSchema collection_schema{schema.dim, core::Metric::l2, schema.scalar_type};
      std::vector<SegmentRegistration> registrations;
      registrations.push_back(std::move(registration));
      if (options.active_registration_factory) {
        auto active = options.active_registration_factory(collection_schema);
        if (!active.ok()) {
          return active.status();
        }
        registrations.push_back(std::move(active).value());
      }
      auto collection =
          SegmentedCollection::open(collection_schema, std::move(registrations), std::move(config));
      if (!collection.ok()) {
        return collection.status();
      }
      core::CheckpointContext checkpoint_context;
      checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
      auto checkpoint = collection.value()->checkpoint(checkpoint_context);
      if (!checkpoint.ok()) {
        return checkpoint.status();
      }
      if (options.fail_point == LegacyImporterFailPoint::after_checkpoint_write) {
        return detail::injected(options.fail_point);
      }

      auto manifest = ArtifactManifestV2::load(options.target_root / kCollectionManifestFilename);
      manifest.collection.scalar_type = schema.scalar_type;
      manifest.collection.logical_id_encoding = schema.kind == "collection"
                                                    ? LogicalIdEncodingV2::canonical_kind_and_bytes
                                                    : LogicalIdEncodingV2::legacy_u64_le;
      SegmentedCollection::apply_checkpoint_to_manifest(checkpoint.value(), manifest);
      manifest.row_versions.minimum = 0;
      manifest.extensions.insert_or_assign("legacy_import.format", "1");
      manifest.extensions.insert_or_assign("legacy_import.source_fingerprint", before.digest);
      manifest.extensions.insert_or_assign("legacy_import.minimum_next_op_id",
                                           std::to_string(minimum_next_op_id));
      for (auto &segment : manifest.segments) {
        segment.wal_cut = checkpoint.value().wal_cut;
        segment.row_versions = manifest.row_versions;
        segment.id_map_checkpoint = checkpoint.value().checkpoint_name;
      }
      detail::atomic_write(options.target_root / kCollectionManifestFilename, manifest.serialize());

      const auto after = detail::fingerprint_source(options.source_root, options.target_root);
      if (after.digest != before.digest || after.files != before.files ||
          after.bytes != before.bytes) {
        throw std::runtime_error("legacy source changed while importer was running");
      }
      LegacyImportAudit audit;
      audit.source_fingerprint = before.digest;
      audit.source_file_count = before.files;
      audit.source_bytes = before.bytes;
      audit.source_kind = schema.kind;
      audit.source_scalar_type = schema.scalar_type;
      audit.source_id_type = schema.id_type;
      audit.dim = schema.dim;
      audit.has_scalar_data = schema.has_scalar_data;
      audit.snapshot_id = snapshot_manifest.snapshot_id;
      audit.snapshot_applied_through = snapshot_manifest.applied_through;
      audit.maximum_seen_op_id = wal.maximum_seen_op_id;
      audit.maximum_committed_op_id = maximum_committed;
      audit.minimum_next_op_id = minimum_next_op_id;
      audit.allocated_rows = allocated_rows;
      audit.live_rows =
          static_cast<std::uint64_t>(std::count_if(rows.begin(), rows.end(), [](const auto &row) {
            return row.state == VersionState::live;
          }));
      audit.tombstone_rows = static_cast<std::uint64_t>(rows.size()) - audit.live_rows;
      audit.committed_wal_records = wal.committed.size();
      audit.torn_wal_tail = wal.torn_tail;
      audit.verified_wal_bytes = wal.verified_bytes;
      audit.rocksdb_mode = schema.has_scalar_data ? "checkpoint_read_only" : "absent";
      audit.checkpoint_name = checkpoint.value().checkpoint_name;
      audit.wal_cut = checkpoint.value().wal_cut;
      audit.manifest_sha256 = sha256_file(options.target_root / kCollectionManifestFilename).hex();
      const auto audit_body = detail::serialize_audit(audit);
      detail::atomic_write(import_dir / kLegacyImportAuditFilename, audit_body);
      if (options.fail_point == LegacyImporterFailPoint::before_marker) {
        return detail::injected(options.fail_point);
      }
      const auto audit_sha = sha256(audit_body).hex();
      const auto prefix = detail::marker_prefix(audit.source_fingerprint, audit_sha);
      detail::atomic_write(import_dir / kLegacyImportMarkerFilename,
                           prefix + "checksum=" + sha256(prefix).hex() + "\n");
      if (options.fail_point == LegacyImporterFailPoint::after_marker) {
        return detail::injected(options.fail_point);
      }
      return LegacyImportResult{std::move(collection).value(), std::move(audit), false};
    } catch (const std::invalid_argument &error) {
      return detail::failure(core::OperationStage::save,
                             core::StatusCode::corruption,
                             error.what());
    } catch (const std::exception &error) {
      return detail::failure(core::OperationStage::save,
                             core::StatusCode::io_error,
                             error.what(),
                             core::StatusDetail::none);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  [[nodiscard]] static auto open(const std::filesystem::path &target_root)
      -> core::Result<LegacyImportResult> {
    return legacy_import_detail::open_imported(target_root);
  }

  [[nodiscard]] static auto resolve_open_route(const std::filesystem::path &source_root,
                                               const std::filesystem::path &target_root)
      -> LegacyOpenRoute {
    if (legacy_import_detail::valid_marker(target_root)) {
      return LegacyOpenRoute::imported_layout;
    }
    try {
      (void)legacy_import_detail::parse_schema(source_root);
      (void)legacy_import_detail::parse_current_snapshot(source_root);
      return LegacyOpenRoute::legacy_layout;
    } catch (...) {
      return LegacyOpenRoute::unavailable;
    }
  }

  [[nodiscard]] static auto marker_valid(const std::filesystem::path &target_root) -> bool {
    return legacy_import_detail::valid_marker(target_root);
  }
};

}  // namespace alaya::internal::collection

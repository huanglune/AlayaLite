// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/collection/legacy_importer.hpp"

namespace alaya::internal::collection {
namespace {

namespace fs = std::filesystem;

const fs::path kCorpusRoot =
    fs::path(ALAYA_SOURCE_DIR) / "python/tests/fixtures/legacy_recovery_corpus";

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = fs::temp_directory_path() /
            ("alaya-legacy-importer-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    fs::remove_all(path_);
    fs::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  [[nodiscard]] auto path() const -> const fs::path & { return path_; }

 private:
  fs::path path_{};
};

struct Case {
  std::string_view name;
  std::string_view kind;
  core::ScalarType scalar_type;
  std::uint64_t allocated;
  std::uint64_t live;
  std::uint64_t tombstones;
  std::uint64_t max_seen;
  std::uint64_t max_committed;
  std::uint64_t committed_records;
  bool torn;
  std::vector<std::string> live_ids;
  std::vector<std::string> vector_sha256;
  std::vector<std::string> query_ids;
  std::vector<std::uint64_t> query_internal_ids;
};

[[nodiscard]] auto cases() -> std::vector<Case> {
  return {{"f32_u32_insert_clean",
           "index",
           core::ScalarType::float32,
           8,
           8,
           0,
           2,
           2,
           0,
           false,
           {"0", "1", "2", "3", "4", "5", "6", "7"},
           {"f2ea2840eea7598ad9548d993d4bfe5548f879f14345436de760d5acef65cf13",
            "5ce7903abf0bad9f9643cc03cda5ed23f2337709e8737e3bc67c453fb3150b24",
            "d5a4d6c636923b94e73516d8b2ef2974f0326ce43f4a61d2acf0a92b7c32342d",
            "a51bcadd144dcb214241f11f50c8bda4fdfc560476758412be16aead9c2a43a4",
            "094a19021e81303c60fb0d0b005f6c61ee1007cc4e515e83adafd6d0fcaf40ff",
            "5b3eab28a91b2bbae1327a2a063a06365af2d0d4df611d2316a7449edd11a66d",
            "dfcaa8c163353474f1a0d47f539a42e08c24abfe9f718914e7439078e42ed484",
            "a2879326db662f7c2ab707037eafe2268f3c69b5b997cafaffbba030206c6aae"},
           {"0", "7"},
           {0, 7}},
          {"i8_u64_insert_remove_wal",
           "index",
           core::ScalarType::int8,
           7,
           6,
           1,
           2,
           2,
           2,
           false,
           {"0", "1", "3", "4", "5", "6"},
           {"b14a1a6891502ce1e117d4decd8a5f1634af660c84993a107e9c0ba0d46cfea8",
            "eac2049c46ead4010a71ae3852f66cbbd645a3e07238311b0d8b706aec125939",
            "fa626e72e6b10b7f03d48ed39cc1a4662a5d3afd6f70dcd4f2df3631d971e223",
            "92afb92e2343c95ec09d2d538cf256fb48e63cba91d81c6ccb98d55f0c137a0c",
            "71324139c90cfdb543ee7a785b2f00170d0d9bbf2213a9d57518095477a01cbc",
            "cc9a17dcb3783ace72d4fcc3dd1a4d552aa79daa3e7cd9c430d6d90c8dde382d"},
           {"0", "6"},
           {0, 6}},
          {"u8_u32_snapshot_insert_wal",
           "index",
           core::ScalarType::uint8,
           8,
           8,
           0,
           2,
           2,
           1,
           false,
           {"0", "1", "2", "3", "4", "5", "6", "7"},
           {"e8b9713233aac669b9d44ed2f765f87b8d21d512e8a1cb4b3f2026c3685dbae2",
            "27930a5dfea9875a6bcbd28fe230bd1280ecaa6d65277442ee5e745f8e83203f",
            "11abf654a52b751b1e7bbbae4aab11364b4ea8a62078dc231db7ad7016199b5b",
            "5809cc949dcdd350b5f793a1c6046f2d6cd7aecc62440912d61022dbfd1f51d7",
            "883c49e38a16620df347defe3f3263758af40e7004341665dc1878944c43c0a7",
            "6033d019de289c2a6fd1eb5c5e08df42e39977ea311598692ba010b219af53d4",
            "df20ec5138ea96764b692b405bd128f4b35279bcd98765c3f7ffbd757590f153",
            "74746e529cdf3f9613a7cd997576bcc83b16d17eddd14bb4c663997fe8c9afa9"},
           {"0", "7"},
           {0, 7}},
          {"f32_u64_torn_tail",
           "index",
           core::ScalarType::float32,
           7,
           7,
           0,
           2,
           1,
           1,
           true,
           {"0", "1", "2", "3", "4", "5", "6"},
           {"7badda7616ab4b05f1d14213251f52aa6d259c42528a5de5aeb216bb777b2a9e",
            "d6fa5da639b4374a834df73394bdce26a69cc599c5764e9f19fbd2bd4db6dbaf",
            "445f142cde2e71a0c9fd64e8897fe5e4755ac2c3603b4e21bf528abbfba7ca74",
            "8af1de8cf61b79fa16a5d92d2c7b333a1d8141e1f01a09f6a527fd2f4a425d70",
            "95b6390bfd3b72f60d2ce5d47d0acf26aef7df61596a8e4aedf5563e05cdd419",
            "b909b0ff5ce543bce57aeb16e011be0e28971cf8f65d02d4db994a9f3742ae37",
            "29aa997b1206492e534ee4f24e4a41c6caf086adc2896c1d84d44260e3e5edf6"},
           {"0", "6"},
           {0, 6}},
          {"i8_u32_collection_upsert_wal",
           "collection",
           core::ScalarType::int8,
           5,
           4,
           0,
           1,
           1,
           1,
           false,
           {"item-0", "item-1", "item-2", "item-3"},
           {"f7343c9114b4140e294606f045aa4199eafbac1a37e4fd8fb2ee8f87028ef330",
            "b07dc7697c9f758ed09e3161ee5d571017ff9a1c222a3ae589afdaeb425d75f0",
            "42ebccebea3cc43d7147d0862515a37bf3efad311d9e9044e8943fc35387c04a",
            "50119e597b3b222a36cb13081f8f62f9e8da1e5596397d106088ca0be3944f70"},
           {"item-1"},
           {4}},
          {"u8_u64_collection_delete_clean",
           "collection",
           core::ScalarType::uint8,
           5,
           4,
           0,
           1,
           1,
           0,
           false,
           {"item-0", "item-1", "item-2", "item-4"},
           {"b0c9e381bc0749901f81cd2c4c8867d44dc68a303015e77592197155c40500cb",
            "6da7cf430967abc7b7ce37309cd2c850abba3c416b424d44d6bcb9749accc5ed",
            "66112bbc94297c0213fffacd7b39f0d9abe0b0a577d68a25797e552e9d87221e",
            "5ce211814d536f0d6a8c53a26649db0cc4f7c966009e090c4e3cf1b52e83ee02"},
           {"item-0"},
           {0}}};
}

[[nodiscard]] auto copy_case(const TemporaryDirectory &temporary, std::string_view name)
    -> fs::path {
  const auto destination = temporary.path() / name;
  fs::copy(kCorpusRoot / name,
           destination,
           fs::copy_options::recursive | fs::copy_options::copy_symlinks);
  return destination;
}

[[nodiscard]] auto read_text(const fs::path &path) -> std::string {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct FrozenFile {
  std::uint64_t bytes{};
  std::string digest{};
};

[[nodiscard]] auto frozen_fixture_files(const fs::path &case_root)
    -> std::map<std::string, FrozenFile, std::less<>> {
  const auto manifest = read_text(case_root / "sha256.json");
  const std::regex entry(
      R"re("([^"]+)"\s*:\s*\{\s*"bytes"\s*:\s*([0-9]+)\s*,\s*"sha256"\s*:\s*"([0-9a-f]{64})"\s*\})re");
  std::map<std::string, FrozenFile, std::less<>> result;
  for (auto found = std::sregex_iterator(manifest.begin(), manifest.end(), entry);
       found != std::sregex_iterator();
       ++found) {
    result.emplace((*found)[1].str(),
                   FrozenFile{std::stoull((*found)[2].str()), (*found)[3].str()});
  }
  if (result.empty()) {
    throw std::runtime_error("fixture checksum manifest did not parse");
  }
  return result;
}

void expect_frozen_fixture_unchanged(const fs::path &case_root,
                                     const std::map<std::string, FrozenFile, std::less<>> &files) {
  for (const auto &[relative, expected] : files) {
    const auto path = case_root / relative;
    ASSERT_TRUE(fs::is_regular_file(path)) << relative;
    EXPECT_EQ(fs::file_size(path), expected.bytes) << relative;
    EXPECT_EQ(sha256_file(path).hex(), expected.digest) << relative;
  }
}

[[nodiscard]] auto id(std::string_view value, std::string_view kind) -> core::LogicalId {
  return kind == "collection"
             ? core::LogicalId::from_utf8(value)
             : core::LogicalId::from_legacy_uint64(std::stoull(std::string(value)));
}

[[nodiscard]] auto query_vectors(const fs::path &expected_path)
    -> std::vector<std::vector<double>> {
  const auto body = read_text(expected_path);
  std::vector<std::vector<double>> result;
  std::size_t offset{};
  while ((offset = body.find("\"vector\"", offset)) != std::string::npos) {
    const auto begin = body.find('[', offset);
    const auto end = body.find(']', begin);
    if (begin == std::string::npos || end == std::string::npos) {
      throw std::runtime_error("fixture query vector array is malformed");
    }
    std::vector<double> vector;
    std::string values = body.substr(begin + 1, end - begin - 1);
    std::replace(values.begin(), values.end(), ',', ' ');
    std::istringstream input(values);
    for (double value{}; input >> value;) {
      vector.push_back(value);
    }
    result.push_back(std::move(vector));
    offset = end + 1;
  }
  return result;
}

[[nodiscard]] auto search_one(const std::shared_ptr<SegmentedCollection> &collection,
                              core::ScalarType scalar_type,
                              const std::vector<double> &query)
    -> core::Result<CollectionSearchResult> {
  core::SearchContext context;
  CollectionSearchRequest request;
  request.options.top_k = 1;
  request.context = &context;
  if (scalar_type == core::ScalarType::float32) {
    std::vector<float> typed(query.begin(), query.end());
    request.queries = core::TypedTensorView::contiguous(typed.data(), 1, typed.size());
    return collection->search(request);
  }
  if (scalar_type == core::ScalarType::int8) {
    std::vector<std::int8_t> typed;
    for (const auto value : query) {
      typed.push_back(static_cast<std::int8_t>(value));
    }
    request.queries = core::TypedTensorView::contiguous(typed.data(), 1, typed.size());
    return collection->search(request);
  }
  std::vector<std::uint8_t> typed;
  for (const auto value : query) {
    typed.push_back(static_cast<std::uint8_t>(value));
  }
  request.queries = core::TypedTensorView::contiguous(typed.data(), 1, typed.size());
  return collection->search(request);
}

[[nodiscard]] auto flat_search_one(const fs::path &root, const std::vector<double> &query)
    -> core::Result<core::SearchHit> {
  core::OpenContext open_context;
  auto opened =
      ::alaya::disk::DiskFlatSegment::open_collection(root, "seg_00000001", {}, open_context);
  if (!opened.ok()) {
    return opened.status();
  }
  std::vector<float> typed(query.begin(), query.end());
  std::array<core::SearchHit, 1> hits{};
  std::array<core::RowCount, 2> offsets{};
  std::array<core::RowCount, 1> counts{};
  std::array<core::Status, 1> statuses{};
  std::array<core::SearchCompleteness, 1> completeness{};
  core::SearchResponse response;
  response.hits = hits;
  response.offsets = offsets;
  response.valid_counts = counts;
  response.statuses = statuses;
  response.completeness = completeness;
  core::SearchContext context;
  core::SearchRequest request;
  request.queries = core::TypedTensorView::contiguous(typed.data(), 1, typed.size());
  request.options.top_k = 1;
  request.context = &context;
  request.response = &response;
  const auto status = opened.value()->search(request);
  if (!status.ok()) {
    return status;
  }
  if (!statuses[0].ok() || counts[0] != 1) {
    return statuses[0].ok() ? core::Status::error(core::StatusCode::internal,
                                                  core::OperationStage::search,
                                                  core::StatusDetail::none,
                                                  "DiskFlat importer query returned no row")
                            : statuses[0];
  }
  return hits[0];
}

[[nodiscard]] auto logical_name(const core::LogicalId &logical_id,
                                const VersionEntry &version,
                                std::string_view kind) -> std::string {
  if (kind == "index") {
    return std::to_string(static_cast<std::uint64_t>(version.address.row_id));
  }
  return {reinterpret_cast<const char *>(logical_id.canonical_bytes().data()),
          logical_id.canonical_bytes().size()};
}

[[nodiscard]] auto expected_versions(std::string_view name)
    -> std::map<std::string, std::pair<std::uint64_t, VersionState>, std::less<>> {
  if (name == "f32_u32_insert_clean") {
    return {{"0", {0, VersionState::live}},
            {"1", {0, VersionState::live}},
            {"2", {0, VersionState::live}},
            {"3", {0, VersionState::live}},
            {"4", {0, VersionState::live}},
            {"5", {0, VersionState::live}},
            {"6", {1, VersionState::live}},
            {"7", {2, VersionState::live}}};
  }
  if (name == "i8_u64_insert_remove_wal") {
    return {{"0", {0, VersionState::live}},
            {"1", {0, VersionState::live}},
            {"2", {2, VersionState::tombstone}},
            {"3", {0, VersionState::live}},
            {"4", {0, VersionState::live}},
            {"5", {0, VersionState::live}},
            {"6", {1, VersionState::live}}};
  }
  if (name == "u8_u32_snapshot_insert_wal") {
    return {{"0", {0, VersionState::live}},
            {"1", {0, VersionState::live}},
            {"2", {0, VersionState::live}},
            {"3", {0, VersionState::live}},
            {"4", {0, VersionState::live}},
            {"5", {0, VersionState::live}},
            {"6", {1, VersionState::live}},
            {"7", {2, VersionState::live}}};
  }
  if (name == "f32_u64_torn_tail") {
    return {{"0", {0, VersionState::live}},
            {"1", {0, VersionState::live}},
            {"2", {0, VersionState::live}},
            {"3", {0, VersionState::live}},
            {"4", {0, VersionState::live}},
            {"5", {0, VersionState::live}},
            {"6", {1, VersionState::live}}};
  }
  if (name == "i8_u32_collection_upsert_wal") {
    return {{"item-0", {0, VersionState::live}},
            {"item-1", {1, VersionState::live}},
            {"item-2", {0, VersionState::live}},
            {"item-3", {0, VersionState::live}}};
  }
  return {{"item-0", {0, VersionState::live}},
          {"item-1", {0, VersionState::live}},
          {"item-2", {0, VersionState::live}},
          {"item-4", {0, VersionState::live}}};
}

void expect_metadata(const std::shared_ptr<SegmentedCollection> &collection) {
  struct Expected {
    std::string id;
    std::string document;
    std::string group;
    std::int64_t version;
    std::int64_t ordinal;
  };
  const std::array upsert{Expected{"item-0", "document-0", "base", 1, 0},
                          Expected{"item-1", "document-1-v2", "updated", 2, 1},
                          Expected{"item-2", "document-2", "base", 1, 2},
                          Expected{"item-3", "document-3", "base", 1, 3}};
  for (const auto &expected : upsert) {
    auto record = collection->get_by_id(core::LogicalId::from_utf8(expected.id));
    ASSERT_TRUE(record.ok()) << expected.id;
    EXPECT_EQ(record.value().document, expected.document);
    EXPECT_EQ(record.value().metadata.at("group"), ScalarValue{expected.group});
    EXPECT_EQ(record.value().metadata.at("version"), ScalarValue{expected.version});
    EXPECT_EQ(record.value().metadata.at("ordinal"), ScalarValue{expected.ordinal});
  }
}

void expect_delete_checkpoint_metadata(const std::shared_ptr<SegmentedCollection> &collection) {
  for (const auto ordinal : {0, 1, 2, 4}) {
    const auto item = "item-" + std::to_string(ordinal);
    auto record = collection->get_by_id(core::LogicalId::from_utf8(item));
    ASSERT_TRUE(record.ok()) << item;
    EXPECT_EQ(record.value().document, "document-" + std::to_string(ordinal));
    EXPECT_EQ(record.value().metadata.at("group"), ScalarValue{std::string("base")});
    EXPECT_EQ(record.value().metadata.at("version"), ScalarValue{std::int64_t{1}});
    EXPECT_EQ(record.value().metadata.at("ordinal"),
              ScalarValue{static_cast<std::int64_t>(ordinal)});
  }
}

TEST(LegacyImporter, SixCorpusCasesPreserveTerminalStateQueryAndEverySourceByte) {
  for (const auto &test_case : cases()) {
    SCOPED_TRACE(test_case.name);
    TemporaryDirectory temporary;
    const auto root = copy_case(temporary, test_case.name);
    const auto frozen = frozen_fixture_files(root);
    expect_frozen_fixture_unchanged(root, frozen);

    LegacyImportOptions options;
    options.source_root = root;
    options.target_root = root;
    options.features.legacy_importer = true;
    options.fail_point = LegacyImporterFailPoint::after_intent_write;
    EXPECT_FALSE(LegacyImporter::import(options).ok());
    EXPECT_EQ(LegacyImporter::resolve_open_route(root, root), LegacyOpenRoute::legacy_layout);
    expect_frozen_fixture_unchanged(root, frozen);
    options.fail_point = LegacyImporterFailPoint::none;
    auto imported = LegacyImporter::import(options);
    ASSERT_TRUE(imported.ok()) << imported.status().diagnostic();
    EXPECT_FALSE(imported.value().already_imported);
    const auto &audit = imported.value().audit;
    EXPECT_EQ(audit.source_kind, test_case.kind);
    EXPECT_EQ(audit.source_scalar_type, test_case.scalar_type);
    EXPECT_EQ(audit.allocated_rows, test_case.allocated);
    EXPECT_EQ(audit.live_rows, test_case.live);
    EXPECT_EQ(audit.tombstone_rows, test_case.tombstones);
    EXPECT_EQ(audit.maximum_seen_op_id, test_case.max_seen);
    EXPECT_EQ(audit.maximum_committed_op_id, test_case.max_committed);
    EXPECT_EQ(audit.minimum_next_op_id, test_case.max_seen + 1);
    EXPECT_EQ(audit.committed_wal_records, test_case.committed_records);
    EXPECT_EQ(audit.torn_wal_tail, test_case.torn);
    EXPECT_EQ(imported.value().collection->stats().size, test_case.live);
    EXPECT_EQ(imported.value().collection->stats().tombstone_count, test_case.tombstones);
    const auto manifest = ArtifactManifestV2::load(root / kCollectionManifestFilename);
    EXPECT_EQ(manifest.collection.scalar_type, test_case.scalar_type);
    EXPECT_EQ(manifest.wal_cut, test_case.max_committed);
    EXPECT_EQ(manifest.collection.metadata_checkpoint, audit.checkpoint_name);
    EXPECT_EQ(manifest.id_map_checkpoint, audit.checkpoint_name);
    EXPECT_EQ(manifest.extensions.at("legacy_import.minimum_next_op_id"),
              std::to_string(test_case.max_seen + 1));
    ASSERT_EQ(manifest.segments.size(), 1U);
    EXPECT_EQ(manifest.segments[0].algorithm_id, core::algorithm::flat);
    EXPECT_EQ(manifest.segments[0].wal_cut, test_case.max_committed);
    EXPECT_EQ(manifest.segments[0].id_map_checkpoint, audit.checkpoint_name);
    auto checkpoint =
        CollectionCheckpointStore::load(root / ".alaya_internal" / kCollectionWalNamespace);
    ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().diagnostic();
    ASSERT_TRUE(checkpoint.value().has_value());
    EXPECT_EQ(checkpoint.value()->wal_cut, test_case.max_committed);
    EXPECT_EQ(checkpoint.value()->state.rows.size(), test_case.live + test_case.tombstones);
    auto logical_wal = CollectionLogicalWal::scan_file(
        root / ".alaya_internal" / kCollectionWalNamespace / kCollectionWalFilename);
    ASSERT_TRUE(logical_wal.ok()) << logical_wal.status().diagnostic();
    ASSERT_EQ(logical_wal.value().frames.size(), 1U);
    EXPECT_EQ(logical_wal.value().frames[0].type, LogicalWalRecordType::checkpoint);

    const auto snapshot = imported.value().collection->pin_routing_snapshot();
    EXPECT_EQ(snapshot->visibility_watermark, test_case.max_committed);
    EXPECT_EQ(snapshot->versions.size(), test_case.live + test_case.tombstones);
    std::vector<std::string> actual_live;
    std::map<std::string, std::pair<std::uint64_t, VersionState>, std::less<>> actual_versions;
    for (const auto &[logical_id, version] : snapshot->versions) {
      ASSERT_TRUE(version.payload.vector.has_value());
      EXPECT_EQ(version.payload.vector->scalar_type(), test_case.scalar_type);
      EXPECT_EQ(version.payload.vector->dim(), 8U);
      actual_versions.emplace(logical_name(logical_id, version, test_case.kind),
                              std::pair{version.upsert_sequence, version.state});
      if (version.state != VersionState::live) {
        continue;
      }
      if (test_case.kind == "collection") {
        actual_live.emplace_back(reinterpret_cast<const char *>(
                                     logical_id.canonical_bytes().data()),
                                 logical_id.canonical_bytes().size());
      } else {
        actual_live.push_back(std::to_string(static_cast<std::uint64_t>(version.address.row_id)));
      }
    }
    EXPECT_EQ(actual_live, test_case.live_ids);
    EXPECT_EQ(actual_versions, expected_versions(test_case.name));

    std::vector<std::string> actual_vector_sha256;
    for (const auto &live_id : test_case.live_ids) {
      auto record = imported.value().collection->get_by_id(id(live_id, test_case.kind));
      ASSERT_TRUE(record.ok()) << live_id;
      ASSERT_TRUE(record.value().vector.has_value()) << live_id;
      actual_vector_sha256.push_back(sha256(record.value().vector->bytes()).hex());
    }
    EXPECT_EQ(actual_vector_sha256, test_case.vector_sha256);

    const auto queries = query_vectors(root / "expected.json");
    ASSERT_EQ(queries.size(), test_case.query_ids.size());
    for (std::size_t index = 0; index < queries.size(); ++index) {
      auto result = search_one(imported.value().collection, test_case.scalar_type, queries[index]);
      ASSERT_TRUE(result.ok()) << result.status().diagnostic();
      ASSERT_EQ(result.value().queries.size(), 1U);
      ASSERT_EQ(result.value().queries[0].hits.size(), 1U);
      EXPECT_EQ(result.value().queries[0].hits[0].logical_id,
                id(test_case.query_ids[index], test_case.kind));
      EXPECT_EQ(result.value().queries[0].hits[0].score, 0.0F);
      auto flat_hit = flat_search_one(root, queries[index]);
      ASSERT_TRUE(flat_hit.ok()) << flat_hit.status().diagnostic();
      EXPECT_EQ(static_cast<std::uint64_t>(flat_hit.value().row_id),
                test_case.query_internal_ids[index]);
      EXPECT_EQ(flat_hit.value().score, 0.0F);
    }
    if (test_case.name == "i8_u32_collection_upsert_wal") {
      expect_metadata(imported.value().collection);
    }
    if (test_case.name == "u8_u64_collection_delete_clean") {
      expect_delete_checkpoint_metadata(imported.value().collection);
      EXPECT_EQ(imported.value()
                    .collection->get_by_id(core::LogicalId::from_utf8("item-3"))
                    .status()
                    .code(),
                core::StatusCode::not_found);
    }

    expect_frozen_fixture_unchanged(root, frozen);
    EXPECT_TRUE(LegacyImporter::marker_valid(root));
    EXPECT_EQ(LegacyImporter::resolve_open_route(root, root), LegacyOpenRoute::imported_layout);

    const auto checkpoint_path =
        root / ".alaya_internal" / kCollectionWalNamespace / audit.checkpoint_name;
    const auto checkpoint_digest = sha256_file(checkpoint_path);
    imported.value().collection.reset();
    auto repeated = LegacyImporter::import(options);
    ASSERT_TRUE(repeated.ok()) << repeated.status().diagnostic();
    EXPECT_TRUE(repeated.value().already_imported);
    EXPECT_EQ(repeated.value().audit.minimum_next_op_id, audit.minimum_next_op_id);
    EXPECT_EQ(repeated.value().collection->stats().size, test_case.live);
    EXPECT_EQ(sha256_file(checkpoint_path), checkpoint_digest);
    expect_frozen_fixture_unchanged(root, frozen);
  }
}

TEST(LegacyImporter, EveryStateCutRestartsIdempotentlyAndSelectsOnlyAnAtomicMarker) {
  const std::array points{LegacyImporterFailPoint::after_rocksdb_validation,
                          LegacyImporterFailPoint::during_wal_decode,
                          LegacyImporterFailPoint::after_op_id_reservation,
                          LegacyImporterFailPoint::after_intent_write,
                          LegacyImporterFailPoint::after_checkpoint_write,
                          LegacyImporterFailPoint::before_marker,
                          LegacyImporterFailPoint::after_marker};
  for (const auto point : points) {
    SCOPED_TRACE(static_cast<unsigned>(point));
    TemporaryDirectory temporary;
    const auto root = copy_case(temporary, "f32_u64_torn_tail");
    const auto frozen = frozen_fixture_files(root);
    LegacyImportOptions options;
    options.source_root = root;
    options.target_root = root;
    options.features.legacy_importer = true;
    options.fail_point = point;
    auto interrupted = LegacyImporter::import(options);
    EXPECT_FALSE(interrupted.ok());
    expect_frozen_fixture_unchanged(root, frozen);
    if (point == LegacyImporterFailPoint::after_marker) {
      EXPECT_TRUE(LegacyImporter::marker_valid(root));
      EXPECT_EQ(LegacyImporter::resolve_open_route(root, root), LegacyOpenRoute::imported_layout);
    } else {
      EXPECT_FALSE(LegacyImporter::marker_valid(root));
      EXPECT_EQ(LegacyImporter::resolve_open_route(root, root), LegacyOpenRoute::legacy_layout);
    }
    options.fail_point = LegacyImporterFailPoint::none;
    auto resumed = LegacyImporter::import(options);
    ASSERT_TRUE(resumed.ok()) << resumed.status().diagnostic();
    EXPECT_EQ(resumed.value().collection->stats().size, 7U);
    EXPECT_EQ(resumed.value().audit.minimum_next_op_id, 3U);
    expect_frozen_fixture_unchanged(root, frozen);
  }

  TemporaryDirectory temporary;
  const auto root = copy_case(temporary, "f32_u64_torn_tail");
  LegacyImportOptions options;
  options.source_root = root;
  options.target_root = root;
  options.features.legacy_importer = true;
  options.fail_point = LegacyImporterFailPoint::before_marker;
  ASSERT_FALSE(LegacyImporter::import(options).ok());
  const auto marker = legacy_import_detail::import_directory(root) / kLegacyImportMarkerFilename;
  {
    std::ofstream partial(marker, std::ios::binary | std::ios::trunc);
    partial << "format=1\nstate=active\n";
  }
  EXPECT_FALSE(LegacyImporter::marker_valid(root));
  EXPECT_EQ(LegacyImporter::resolve_open_route(root, root), LegacyOpenRoute::legacy_layout);
  options.fail_point = LegacyImporterFailPoint::none;
  ASSERT_TRUE(LegacyImporter::import(options).ok());
  EXPECT_TRUE(LegacyImporter::marker_valid(root));
}

TEST(LegacyImporter, IndependentGateStopsOnlyNewImportsAndMarkerIsRollForwardOnly) {
  TemporaryDirectory temporary;
  const auto source = copy_case(temporary, "i8_u32_collection_upsert_wal");
  const auto target = temporary.path() / "new-layout";
  LegacyImportOptions options;
  options.source_root = source;
  options.target_root = target;
  auto disabled = LegacyImporter::import(options);
  ASSERT_FALSE(disabled.ok());
  EXPECT_EQ(disabled.status().code(), core::StatusCode::not_supported);
  EXPECT_EQ(LegacyImporter::resolve_open_route(source, target), LegacyOpenRoute::legacy_layout);

  options.features.legacy_importer = true;
  auto imported = LegacyImporter::import(options);
  ASSERT_TRUE(imported.ok()) << imported.status().diagnostic();
  EXPECT_EQ(LegacyImporter::resolve_open_route(source, target), LegacyOpenRoute::imported_layout);
  imported.value().collection.reset();
  fs::rename(source, temporary.path() / "legacy-source-retained-but-offline");
  auto reopened = LegacyImporter::open(target);
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value().collection->stats().size, 4U);

  options.features.legacy_importer = false;
  auto gate_disabled_after_marker = LegacyImporter::import(options);
  ASSERT_TRUE(gate_disabled_after_marker.ok());
  EXPECT_TRUE(gate_disabled_after_marker.value().already_imported);
  const auto wal_path =
      target / ".alaya_internal" / kCollectionWalNamespace / std::string(kCollectionWalFilename);
  auto scan = CollectionLogicalWal::scan_file(wal_path);
  ASSERT_TRUE(scan.ok());
  ASSERT_EQ(scan.value().frames.size(), 1U);
  EXPECT_EQ(scan.value().frames.front().type, LogicalWalRecordType::checkpoint);
}

}  // namespace
}  // namespace alaya::internal::collection

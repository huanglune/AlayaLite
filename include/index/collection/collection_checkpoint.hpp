// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/platform_fs.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/mutation_wal_codec.hpp"
#include "index/collection/routing_snapshot.hpp"

namespace alaya::internal::collection {

struct CollectionCheckpointImage {
  std::uint64_t generation{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  std::uint64_t metadata_epoch{};
  std::uint64_t wal_cut{};
  WalMutationTransaction state{};
  std::map<std::string, MutationReceipt, std::less<>> retry_receipts{};
  std::map<std::string, BatchMutationReceipt, std::less<>> batch_retry_receipts{};
  std::string checkpoint_name{};
};

class CollectionCheckpointStore {
 public:
  [[nodiscard]] static auto write(
      const std::filesystem::path &wal_directory,
      const RoutingSnapshot &snapshot,
      const std::map<std::string, MutationReceipt, std::less<>> &retry_receipts,
      const std::map<std::string, BatchMutationReceipt, std::less<>> &batch_retry_receipts)
      -> core::Result<CheckpointReceipt> {
    try {
      std::filesystem::create_directories(wal_directory);
      CollectionCheckpointImage image;
      image.generation = snapshot.generation;
      image.visibility_watermark = snapshot.visibility_watermark;
      image.durable_watermark = snapshot.visibility_watermark;
      image.metadata_epoch = snapshot.metadata_epoch;
      image.wal_cut = snapshot.visibility_watermark;
      image.checkpoint_name = "checkpoint_" + std::to_string(image.wal_cut) + ".bin";
      image.retry_receipts = retry_receipts;
      image.batch_retry_receipts = batch_retry_receipts;
      image.state.batch_op_id = image.wal_cut;
      image.state.batch_mode = BatchMutationMode::all_or_nothing;
      image.state.durability = WriteDurability::wal_fsync;
      image.state.rows.reserve(snapshot.versions.size());
      for (const auto &[logical_id, version] : snapshot.versions) {
        WalMutationRow row;
        row.op_id = version.upsert_sequence;
        row.action = version.state == VersionState::live ? SegmentMutationAction::write
                                                         : SegmentMutationAction::erase;
        row.status = version.state == VersionState::live ? RowMutationStatus::inserted
                                                         : RowMutationStatus::deleted;
        row.logical_id = logical_id;
        row.target = version.address;
        row.payload = version.payload;
        image.state.rows.push_back(std::move(row));
      }
      const auto bytes = encode(image);
      const auto temporary = wal_directory / (image.checkpoint_name + ".tmp");
      const auto final = wal_directory / image.checkpoint_name;
      write_file(temporary, bytes);
      platform::sync_file_or_throw(temporary);
      platform::atomic_replace(temporary, final);
      platform::sync_directory_or_throw(wal_directory);

      const auto current_temporary = wal_directory / "CURRENT.tmp";
      const auto current = wal_directory / "CURRENT";
      const auto current_body = "checkpoint=" + image.checkpoint_name +
                                "\nwal_cut=" + std::to_string(image.wal_cut) + "\n";
      write_file(current_temporary,
                 std::span(reinterpret_cast<const std::byte *>(current_body.data()),
                           current_body.size()));
      platform::sync_file_or_throw(current_temporary);
      platform::atomic_replace(current_temporary, current);
      platform::sync_directory_or_throw(wal_directory);
      return CheckpointReceipt{image.durable_watermark,
                               image.wal_cut,
                               image.metadata_epoch,
                               image.checkpoint_name};
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::checkpoint,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::checkpoint,
                                 core::StatusDetail::none,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::checkpoint);
    }
  }

  [[nodiscard]] static auto load(const std::filesystem::path &wal_directory)
      -> core::Result<std::optional<CollectionCheckpointImage>> {
    try {
      const auto current = wal_directory / "CURRENT";
      if (!std::filesystem::exists(current)) {
        return std::optional<CollectionCheckpointImage>{};
      }
      const auto body = read_file(current, 4096);
      const std::string text(reinterpret_cast<const char *>(body.data()), body.size());
      std::string checkpoint_name;
      std::optional<std::uint64_t> wal_cut;
      std::istringstream lines(text);
      for (std::string line; std::getline(lines, line);) {
        if (line.starts_with("checkpoint=")) {
          checkpoint_name = line.substr(std::string("checkpoint=").size());
        } else if (line.starts_with("wal_cut=")) {
          wal_cut = parse_u64(line.substr(std::string("wal_cut=").size()));
        } else if (!line.empty()) {
          throw std::invalid_argument("checkpoint CURRENT contains an unknown field");
        }
      }
      if (checkpoint_name.empty() || checkpoint_name.find('/') != std::string::npos ||
          checkpoint_name.find('\\') != std::string::npos || !wal_cut.has_value()) {
        throw std::invalid_argument("checkpoint CURRENT is malformed");
      }
      auto image = decode(
          read_file(wal_directory / checkpoint_name, logical_wal_detail::kMaximumPayloadBytes));
      if (image.wal_cut != *wal_cut) {
        throw std::invalid_argument("checkpoint CURRENT WAL cut does not match the image");
      }
      image.checkpoint_name = std::move(checkpoint_name);
      return std::optional<CollectionCheckpointImage>(std::move(image));
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::open,
                                 core::StatusDetail::none,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  static void apply_to_manifest(const CheckpointReceipt &checkpoint, ArtifactManifestV2 &manifest) {
    manifest.wal_cut = checkpoint.wal_cut;
    manifest.collection.metadata_epoch = checkpoint.metadata_epoch;
    manifest.collection.metadata_checkpoint = checkpoint.checkpoint_name;
    manifest.id_map_checkpoint = checkpoint.checkpoint_name;
    if (manifest.row_versions.minimum == 0 || manifest.row_versions.minimum > checkpoint.wal_cut) {
      manifest.row_versions.minimum = checkpoint.wal_cut == 0 ? 0 : 1;
    }
    manifest.row_versions.maximum = checkpoint.wal_cut;
  }

 private:
  inline static constexpr std::uint32_t kMagic = 0x37504B43U;    // "CKP7".
  inline static constexpr std::uint32_t kTrailer = 0x37444E45U;  // "END7".
  inline static constexpr std::uint16_t kVersion = 1;

  [[nodiscard]] static auto encode(const CollectionCheckpointImage &image)
      -> std::vector<std::byte> {
    std::vector<std::byte> payload;
    logical_wal_detail::put_u64(payload, image.generation);
    logical_wal_detail::put_u64(payload, image.visibility_watermark);
    logical_wal_detail::put_u64(payload, image.durable_watermark);
    logical_wal_detail::put_u64(payload, image.metadata_epoch);
    logical_wal_detail::put_u64(payload, image.wal_cut);
    const auto state = encode_wal_transaction(image.state);
    logical_wal_detail::put_u64(payload, state.size());
    payload.insert(payload.end(), state.begin(), state.end());
    mutation_wal_codec_detail::put_u32(payload,
                                       static_cast<std::uint32_t>(image.retry_receipts.size()));
    for (const auto &[token, receipt] : image.retry_receipts) {
      mutation_wal_codec_detail::put_string(payload, token);
      encode_receipt(payload, receipt);
    }
    mutation_wal_codec_detail::put_u32(payload,
                                       static_cast<std::uint32_t>(
                                           image.batch_retry_receipts.size()));
    for (const auto &[token, receipt] : image.batch_retry_receipts) {
      mutation_wal_codec_detail::put_string(payload, token);
      mutation_wal_codec_detail::put_u64(payload, receipt.batch_op_id);
      mutation_wal_codec_detail::put_u64(payload, receipt.visibility_watermark);
      mutation_wal_codec_detail::put_u64(payload, receipt.durable_watermark);
      mutation_wal_codec_detail::put_u8(payload, receipt.searchable ? 1 : 0);
      mutation_wal_codec_detail::put_u8(payload, static_cast<std::uint8_t>(receipt.durability));
      mutation_wal_codec_detail::put_u32(payload, static_cast<std::uint32_t>(receipt.rows.size()));
      for (const auto &row : receipt.rows) {
        encode_receipt(payload, row);
      }
    }
    std::vector<std::byte> output;
    logical_wal_detail::put_u32(output, kMagic);
    logical_wal_detail::put_u16(output, kVersion);
    logical_wal_detail::put_u16(output, 0);
    logical_wal_detail::put_u64(output, payload.size());
    logical_wal_detail::put_u32(output, logical_wal_detail::crc32(payload));
    output.insert(output.end(), payload.begin(), payload.end());
    logical_wal_detail::put_u32(output, kTrailer);
    return output;
  }

  [[nodiscard]] static auto decode(std::span<const std::byte> bytes) -> CollectionCheckpointImage {
    constexpr std::size_t kHeader = 20;
    if (bytes.size() < kHeader + 4 || logical_wal_detail::get_u32(bytes, 0) != kMagic ||
        logical_wal_detail::get_u16(bytes, 4) != kVersion) {
      throw std::invalid_argument("checkpoint image header is invalid");
    }
    const auto payload_size = logical_wal_detail::get_u64(bytes, 8);
    if (payload_size != bytes.size() - kHeader - 4 ||
        logical_wal_detail::get_u32(bytes, bytes.size() - 4) != kTrailer) {
      throw std::invalid_argument("checkpoint image length/trailer is invalid");
    }
    const auto payload = bytes.subspan(kHeader, payload_size);
    if (logical_wal_detail::crc32(payload) != logical_wal_detail::get_u32(bytes, 16)) {
      throw std::invalid_argument("checkpoint image checksum is invalid");
    }
    if (payload.size() < 48) {
      throw std::invalid_argument("checkpoint image payload is truncated");
    }
    CollectionCheckpointImage image;
    image.generation = logical_wal_detail::get_u64(payload, 0);
    image.visibility_watermark = logical_wal_detail::get_u64(payload, 8);
    image.durable_watermark = logical_wal_detail::get_u64(payload, 16);
    image.metadata_epoch = logical_wal_detail::get_u64(payload, 24);
    image.wal_cut = logical_wal_detail::get_u64(payload, 32);
    const auto state_size = logical_wal_detail::get_u64(payload, 40);
    if (state_size > payload.size() - 48) {
      throw std::invalid_argument("checkpoint state payload is truncated");
    }
    image.state = decode_wal_transaction(payload.subspan(48, state_size));
    mutation_wal_codec_detail::Decoder decoder(payload.subspan(48 + state_size));
    const auto retry_count = decoder.u32();
    if (retry_count > mutation_wal_codec_detail::kMaximumRows) {
      throw std::invalid_argument("checkpoint retry ledger is too large");
    }
    for (std::uint32_t index = 0; index < retry_count; ++index) {
      auto token = decoder.string();
      auto receipt = decode_receipt(decoder);
      if (!image.retry_receipts.emplace(std::move(token), std::move(receipt)).second) {
        throw std::invalid_argument("checkpoint retry ledger contains a duplicate token");
      }
    }
    const auto batch_count = decoder.u32();
    if (batch_count > mutation_wal_codec_detail::kMaximumRows) {
      throw std::invalid_argument("checkpoint batch retry ledger is too large");
    }
    for (std::uint32_t index = 0; index < batch_count; ++index) {
      auto token = decoder.string();
      BatchMutationReceipt receipt;
      receipt.batch_op_id = decoder.u64();
      receipt.visibility_watermark = decoder.u64();
      receipt.durable_watermark = decoder.u64();
      receipt.searchable = decoder.u8() != 0;
      receipt.durability = decode_durability(decoder.u8());
      receipt.retry_token = token;
      const auto row_count = decoder.u32();
      if (row_count > mutation_wal_codec_detail::kMaximumRows) {
        throw std::invalid_argument("checkpoint batch receipt has too many rows");
      }
      receipt.rows.reserve(row_count);
      for (std::uint32_t row = 0; row < row_count; ++row) {
        receipt.rows.push_back(decode_receipt(decoder));
      }
      if (!image.batch_retry_receipts.emplace(std::move(token), std::move(receipt)).second) {
        throw std::invalid_argument("checkpoint batch retry ledger contains a duplicate token");
      }
    }
    if (!decoder.empty()) {
      throw std::invalid_argument("checkpoint image has trailing receipt bytes");
    }
    return image;
  }

  static void encode_receipt(std::vector<std::byte> &output, const MutationReceipt &receipt) {
    mutation_wal_codec_detail::put_u64(output, receipt.op_id);
    mutation_wal_codec_detail::put_u64(output, receipt.batch_op_id);
    mutation_wal_codec_detail::put_u64(output, receipt.row_op_id);
    mutation_wal_codec_detail::put_u64(output, receipt.visibility_watermark);
    mutation_wal_codec_detail::put_u64(output, receipt.durable_watermark);
    mutation_wal_codec_detail::put_u8(output, receipt.searchable ? 1 : 0);
    mutation_wal_codec_detail::put_u8(output, static_cast<std::uint8_t>(receipt.durability));
    mutation_wal_codec_detail::put_u8(output, static_cast<std::uint8_t>(receipt.row_status));
    mutation_wal_codec_detail::put_string(output, receipt.retry_token);
  }

  [[nodiscard]] static auto decode_durability(std::uint8_t raw) -> DurabilityState {
    if (raw > static_cast<std::uint8_t>(DurabilityState::wal_fsync)) {
      throw std::invalid_argument("checkpoint receipt durability is invalid");
    }
    return static_cast<DurabilityState>(raw);
  }

  [[nodiscard]] static auto decode_receipt(mutation_wal_codec_detail::Decoder &decoder)
      -> MutationReceipt {
    MutationReceipt receipt;
    receipt.op_id = decoder.u64();
    receipt.batch_op_id = decoder.u64();
    receipt.row_op_id = decoder.u64();
    receipt.visibility_watermark = decoder.u64();
    receipt.durable_watermark = decoder.u64();
    receipt.searchable = decoder.u8() != 0;
    receipt.durability = decode_durability(decoder.u8());
    const auto status = decoder.u8();
    if (status > static_cast<std::uint8_t>(RowMutationStatus::aborted)) {
      throw std::invalid_argument("checkpoint receipt row status is invalid");
    }
    receipt.row_status = static_cast<RowMutationStatus>(status);
    receipt.retry_token = decoder.string();
    return receipt;
  }

  static void write_file(const std::filesystem::path &path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) {
      throw std::runtime_error("cannot write collection checkpoint file");
    }
  }

  [[nodiscard]] static auto read_file(const std::filesystem::path &path, std::uint64_t limit)
      -> std::vector<std::byte> {
    const auto size = std::filesystem::file_size(path);
    if (size > limit || size > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("collection checkpoint file exceeds its size limit");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input && !bytes.empty()) {
      throw std::runtime_error("cannot read collection checkpoint file");
    }
    return bytes;
  }

  [[nodiscard]] static auto parse_u64(std::string_view value) -> std::uint64_t {
    if (value.empty()) {
      throw std::invalid_argument("checkpoint CURRENT has an empty WAL cut");
    }
    std::uint64_t result{};
    for (const auto digit : value) {
      if (digit < '0' || digit > '9' ||
          result > (std::numeric_limits<std::uint64_t>::max() -
                    static_cast<std::uint64_t>(digit - '0')) /
                       10U) {
        throw std::invalid_argument("checkpoint CURRENT WAL cut is invalid");
      }
      result = result * 10U + static_cast<std::uint64_t>(digit - '0');
    }
    return result;
  }
};

}  // namespace alaya::internal::collection

// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "index/collection/logical_wal.hpp"
#include "index/collection/types.hpp"
#include "wal/frame.hpp"

namespace alaya::internal::collection {

struct WalMutationRow {
  std::uint64_t op_id{};
  SegmentMutationAction action{SegmentMutationAction::write};
  RowMutationStatus status{RowMutationStatus::aborted};
  core::LogicalId logical_id{};
  RowAddress target{};
  std::optional<RowAddress> previous{};
  RecordPayload payload{};
  std::string retry_token{};
};

struct WalMutationTransaction {
  std::uint64_t batch_op_id{};
  BatchMutationMode batch_mode{BatchMutationMode::per_row_independent};
  WriteDurability durability{WriteDurability::wal_fsync};
  std::string retry_token{};
  std::vector<WalMutationRow> rows{};
};

namespace mutation_wal_codec_detail {

inline constexpr std::uint16_t kPayloadVersion = 1;
inline constexpr std::uint32_t kMaximumRows = 1U << 20U;
inline constexpr std::uint32_t kMaximumStringBytes = 16U << 20U;

inline void put_u8(std::vector<std::byte> &output, std::uint8_t value) {
  output.push_back(static_cast<std::byte>(value));
}

inline void put_u32(std::vector<std::byte> &output, std::uint32_t value) {
  logical_wal_detail::put_u32(output, value);
}

inline void put_u64(std::vector<std::byte> &output, std::uint64_t value) {
  logical_wal_detail::put_u64(output, value);
}

inline void put_bytes(std::vector<std::byte> &output, std::span<const std::byte> value) {
  if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("mutation WAL byte field exceeds uint32");
  }
  put_u32(output, static_cast<std::uint32_t>(value.size()));
  output.insert(output.end(), value.begin(), value.end());
}

inline void put_string(std::vector<std::byte> &output, std::string_view value) {
  put_bytes(output, std::span(reinterpret_cast<const std::byte *>(value.data()), value.size()));
}

inline void put_address(std::vector<std::byte> &output, const RowAddress &address) {
  put_u64(output, address.segment_id);
  put_u64(output, address.generation);
  put_u64(output, static_cast<std::uint64_t>(address.row_id));
}

// The byte Decoder is hoisted into the framework layer (alaya::wal, see
// unified-wal-vocabulary.md section 5) so both op families share one reader.
using alaya::wal::Decoder;

// RowAddress is a collection type, so its composite decode helper stays here;
// the hoisted alaya::wal::Decoder only knows scalar/byte primitives.
[[nodiscard]] inline auto decode_address(Decoder &decoder) -> RowAddress {
  return {decoder.u64(), decoder.u64(), core::SegmentRowId(decoder.u64())};
}

inline void encode_metadata(std::vector<std::byte> &output, const Metadata &metadata) {
  if (metadata.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("mutation WAL metadata has too many fields");
  }
  put_u32(output, static_cast<std::uint32_t>(metadata.size()));
  for (const auto &[key, value] : metadata) {
    put_string(output, key);
    put_u8(output, static_cast<std::uint8_t>(value.index()));
    std::visit(
        [&](const auto &typed) {
          using Value = std::decay_t<decltype(typed)>;
          if constexpr (std::same_as<Value, bool>) {
            put_u8(output, typed ? 1 : 0);
          } else if constexpr (std::same_as<Value, std::int64_t>) {
            put_u64(output, std::bit_cast<std::uint64_t>(typed));
          } else if constexpr (std::same_as<Value, double>) {
            put_u64(output, std::bit_cast<std::uint64_t>(typed));
          } else {
            put_string(output, typed);
          }
        },
        value);
  }
}

[[nodiscard]] inline auto decode_metadata(Decoder &decoder) -> Metadata {
  const auto count = decoder.u32();
  if (count > kMaximumRows) {
    throw std::invalid_argument("mutation WAL metadata field count exceeds the decoder limit");
  }
  Metadata metadata;
  for (std::uint32_t index = 0; index < count; ++index) {
    auto key = decoder.string();
    const auto type = decoder.u8();
    ScalarValue value;
    switch (type) {
      case 0:
        value = decoder.u8() != 0;
        break;
      case 1:
        value = std::bit_cast<std::int64_t>(decoder.u64());
        break;
      case 2:
        value = std::bit_cast<double>(decoder.u64());
        break;
      case 3:
        value = decoder.string();
        break;
      default:
        throw std::invalid_argument("mutation WAL metadata variant is invalid");
    }
    if (!metadata.emplace(std::move(key), std::move(value)).second) {
      throw std::invalid_argument("mutation WAL metadata contains a duplicate key");
    }
  }
  return metadata;
}

inline void encode_payload(std::vector<std::byte> &output, const RecordPayload &payload) {
  put_u8(output, payload.vector.has_value() ? 1 : 0);
  if (payload.vector.has_value()) {
    put_u8(output, static_cast<std::uint8_t>(payload.vector->scalar_type()));
    put_u32(output, payload.vector->dim());
    put_bytes(output, payload.vector->bytes());
  }
  encode_metadata(output, payload.metadata);
  put_string(output, payload.document);
}

[[nodiscard]] inline auto decode_payload(Decoder &decoder) -> RecordPayload {
  RecordPayload payload;
  if (decoder.u8() != 0) {
    const auto scalar = static_cast<core::ScalarType>(decoder.u8());
    const auto dim = decoder.u32();
    const auto bytes = decoder.bytes();
    std::uint64_t expected{};
    if (core::scalar_type_size(scalar) == 0 ||
        !core::checked_multiply(dim, core::scalar_type_size(scalar), expected) ||
        expected != bytes.size()) {
      throw std::invalid_argument("mutation WAL vector shape does not match its byte payload");
    }
    const core::TypedTensorView view{bytes.data(), scalar, 1, dim, expected};
    auto owned = OwnedVector::copy_row(view, 0);
    if (!owned.ok()) {
      throw std::invalid_argument(owned.status().diagnostic());
    }
    payload.vector = std::move(owned).value();
  }
  payload.metadata = decode_metadata(decoder);
  payload.document = decoder.string();
  return payload;
}

inline void encode_logical_id(std::vector<std::byte> &output, const core::LogicalId &id) {
  put_u8(output, static_cast<std::uint8_t>(id.kind()));
  put_bytes(output, id.canonical_bytes());
}

[[nodiscard]] inline auto decode_logical_id(Decoder &decoder) -> core::LogicalId {
  const auto kind = static_cast<core::LogicalIdKind>(decoder.u8());
  const auto bytes = decoder.bytes();
  if (kind == core::LogicalIdKind::utf8) {
    return core::LogicalId::from_utf8({reinterpret_cast<const char *>(bytes.data()), bytes.size()});
  }
  if (kind == core::LogicalIdKind::legacy_uint64 && bytes.size() == sizeof(std::uint64_t)) {
    std::uint64_t value{};
    for (const auto byte : bytes) {
      value = (value << 8U) | std::to_integer<std::uint8_t>(byte);
    }
    return core::LogicalId::from_legacy_uint64(value);
  }
  throw std::invalid_argument("mutation WAL LogicalId kind/bytes are invalid");
}

}  // namespace mutation_wal_codec_detail

[[nodiscard]] inline auto encode_wal_transaction(const WalMutationTransaction &transaction)
    -> std::vector<std::byte> {
  using namespace mutation_wal_codec_detail;  // NOLINT(build/namespaces)
  if (transaction.rows.size() > kMaximumRows) {
    throw std::invalid_argument("mutation WAL transaction has too many rows");
  }
  std::vector<std::byte> output;
  output.reserve(256);
  logical_wal_detail::put_u16(output, kPayloadVersion);
  put_u8(output, static_cast<std::uint8_t>(transaction.batch_mode));
  put_u8(output, static_cast<std::uint8_t>(transaction.durability));
  put_u64(output, transaction.batch_op_id);
  put_string(output, transaction.retry_token);
  put_u32(output, static_cast<std::uint32_t>(transaction.rows.size()));
  for (const auto &row : transaction.rows) {
    put_u64(output, row.op_id);
    put_u8(output, static_cast<std::uint8_t>(row.action));
    put_u8(output, static_cast<std::uint8_t>(row.status));
    encode_logical_id(output, row.logical_id);
    put_address(output, row.target);
    put_u8(output, row.previous.has_value() ? 1 : 0);
    if (row.previous.has_value()) {
      put_address(output, *row.previous);
    }
    encode_payload(output, row.payload);
    put_string(output, row.retry_token);
  }
  return output;
}

[[nodiscard]] inline auto decode_wal_transaction(std::span<const std::byte> payload)
    -> WalMutationTransaction {
  using namespace mutation_wal_codec_detail;  // NOLINT(build/namespaces)
  Decoder decoder(payload);
  if (decoder.u16() != kPayloadVersion) {
    throw std::invalid_argument("mutation WAL payload version is unsupported");
  }
  const auto mode = decoder.u8();
  const auto durability = decoder.u8();
  if (mode > static_cast<std::uint8_t>(BatchMutationMode::all_or_nothing) ||
      durability > static_cast<std::uint8_t>(WriteDurability::wal_fsync)) {
    throw std::invalid_argument("mutation WAL mode/durability enum is invalid");
  }
  WalMutationTransaction transaction;
  transaction.batch_mode = static_cast<BatchMutationMode>(mode);
  transaction.durability = static_cast<WriteDurability>(durability);
  transaction.batch_op_id = decoder.u64();
  transaction.retry_token = decoder.string();
  const auto count = decoder.u32();
  if (count > kMaximumRows) {
    throw std::invalid_argument("mutation WAL row count is invalid");
  }
  transaction.rows.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    WalMutationRow row;
    row.op_id = decoder.u64();
    const auto action = decoder.u8();
    const auto status = decoder.u8();
    if (action > static_cast<std::uint8_t>(SegmentMutationAction::erase) ||
        status > static_cast<std::uint8_t>(RowMutationStatus::aborted)) {
      throw std::invalid_argument("mutation WAL row enum is invalid");
    }
    row.action = static_cast<SegmentMutationAction>(action);
    row.status = static_cast<RowMutationStatus>(status);
    row.logical_id = decode_logical_id(decoder);
    row.target = decode_address(decoder);
    if (decoder.u8() != 0) {
      row.previous = decode_address(decoder);
    }
    row.payload = decode_payload(decoder);
    row.retry_token = decoder.string();
    transaction.rows.push_back(std::move(row));
  }
  if (!decoder.empty()) {
    throw std::invalid_argument("mutation WAL payload has trailing bytes");
  }
  return transaction;
}

[[nodiscard]] inline auto wal_transaction_fingerprint(const WalMutationTransaction &transaction)
    -> std::uint32_t {
  const auto encoded = encode_wal_transaction(transaction);
  return logical_wal_detail::crc32(encoded);
}

[[nodiscard]] inline auto encode_batch_receipt_marker(const BatchMutationReceipt &receipt)
    -> std::vector<std::byte> {
  using namespace mutation_wal_codec_detail;  // NOLINT(build/namespaces)
  if (receipt.rows.size() > kMaximumRows) {
    throw std::invalid_argument("batch receipt marker has too many rows");
  }
  std::vector<std::byte> output;
  logical_wal_detail::put_u16(output, 1);
  put_string(output, receipt.retry_token);
  put_u64(output, receipt.batch_op_id);
  put_u64(output, receipt.visibility_watermark);
  put_u64(output, receipt.durable_watermark);
  put_u8(output, receipt.searchable ? 1 : 0);
  put_u8(output, static_cast<std::uint8_t>(receipt.durability));
  put_u32(output, static_cast<std::uint32_t>(receipt.rows.size()));
  for (const auto &row : receipt.rows) {
    put_u64(output, row.op_id);
    put_u64(output, row.batch_op_id);
    put_u64(output, row.row_op_id);
    put_u64(output, row.visibility_watermark);
    put_u64(output, row.durable_watermark);
    put_u8(output, row.searchable ? 1 : 0);
    put_u8(output, static_cast<std::uint8_t>(row.durability));
    put_u8(output, static_cast<std::uint8_t>(row.row_status));
    put_string(output, row.retry_token);
  }
  return output;
}

[[nodiscard]] inline auto decode_batch_receipt_marker(std::span<const std::byte> payload)
    -> BatchMutationReceipt {
  using namespace mutation_wal_codec_detail;  // NOLINT(build/namespaces)
  Decoder decoder(payload);
  if (decoder.u16() != 1) {
    throw std::invalid_argument("batch receipt marker version is unsupported");
  }
  BatchMutationReceipt receipt;
  receipt.retry_token = decoder.string();
  receipt.batch_op_id = decoder.u64();
  receipt.visibility_watermark = decoder.u64();
  receipt.durable_watermark = decoder.u64();
  receipt.searchable = decoder.u8() != 0;
  const auto batch_durability = decoder.u8();
  if (batch_durability > static_cast<std::uint8_t>(DurabilityState::wal_fsync)) {
    throw std::invalid_argument("batch receipt marker durability is invalid");
  }
  receipt.durability = static_cast<DurabilityState>(batch_durability);
  const auto count = decoder.u32();
  if (count > kMaximumRows) {
    throw std::invalid_argument("batch receipt marker row count is invalid");
  }
  receipt.rows.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    MutationReceipt row;
    row.op_id = decoder.u64();
    row.batch_op_id = decoder.u64();
    row.row_op_id = decoder.u64();
    row.visibility_watermark = decoder.u64();
    row.durable_watermark = decoder.u64();
    row.searchable = decoder.u8() != 0;
    const auto durability = decoder.u8();
    const auto status = decoder.u8();
    if (durability > static_cast<std::uint8_t>(DurabilityState::wal_fsync) ||
        status > static_cast<std::uint8_t>(RowMutationStatus::aborted)) {
      throw std::invalid_argument("batch receipt marker row enum is invalid");
    }
    row.durability = static_cast<DurabilityState>(durability);
    row.row_status = static_cast<RowMutationStatus>(status);
    row.retry_token = decoder.string();
    receipt.rows.push_back(std::move(row));
  }
  if (!decoder.empty()) {
    throw std::invalid_argument("batch receipt marker has trailing bytes");
  }
  return receipt;
}

}  // namespace alaya::internal::collection

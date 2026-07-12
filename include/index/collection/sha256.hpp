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
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace alaya::internal::collection {

struct Sha256Digest {
  std::array<std::byte, 32> bytes{};

  [[nodiscard]] auto hex() const -> std::string {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(64, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      const auto value = std::to_integer<unsigned>(bytes[i]);
      output[i * 2] = kHex[value >> 4U];
      output[i * 2 + 1] = kHex[value & 0x0fU];
    }
    return output;
  }

  [[nodiscard]] static auto from_hex(std::string_view value) -> Sha256Digest {
    if (value.size() != 64) {
      throw std::invalid_argument("SHA-256 digest must contain exactly 64 hexadecimal digits");
    }
    auto nibble = [](char digit) -> unsigned {
      if (digit >= '0' && digit <= '9') {
        return static_cast<unsigned>(digit - '0');
      }
      if (digit >= 'a' && digit <= 'f') {
        return static_cast<unsigned>(digit - 'a' + 10);
      }
      if (digit >= 'A' && digit <= 'F') {
        return static_cast<unsigned>(digit - 'A' + 10);
      }
      throw std::invalid_argument("SHA-256 digest contains a non-hexadecimal digit");
    };
    Sha256Digest digest;
    for (std::size_t i = 0; i < digest.bytes.size(); ++i) {
      digest.bytes[i] =
          static_cast<std::byte>((nibble(value[i * 2]) << 4U) | nibble(value[i * 2 + 1]));
    }
    return digest;
  }

  auto operator<=>(const Sha256Digest &) const = default;
};

// A dependency-free, streaming SHA-256 implementation for control-plane
// manifests. Engine hot paths never call this type.
class Sha256 {
 public:
  Sha256() = default;

  void update(std::span<const std::byte> input) {
    if (finalized_) {
      throw std::logic_error("SHA-256 update called after finalize");
    }
    if (input.size() > (UINT64_MAX - total_bytes_)) {
      throw std::overflow_error("SHA-256 input byte count overflows uint64");
    }
    total_bytes_ += static_cast<std::uint64_t>(input.size());
    std::size_t offset = 0;
    if (buffer_size_ != 0) {
      const auto copied = std::min(input.size(), buffer_.size() - buffer_size_);
      std::memcpy(buffer_.data() + buffer_size_, input.data(), copied);
      buffer_size_ += copied;
      offset += copied;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_.data());
        buffer_size_ = 0;
      }
    }
    while (input.size() - offset >= buffer_.size()) {
      transform(input.data() + offset);
      offset += buffer_.size();
    }
    if (offset != input.size()) {
      buffer_size_ = input.size() - offset;
      std::memcpy(buffer_.data(), input.data() + offset, buffer_size_);
    }
  }

  void update(std::string_view input) {
    update(std::as_bytes(std::span(input.data(), input.size())));
  }

  [[nodiscard]] auto finalize() -> Sha256Digest {
    if (finalized_) {
      return digest_;
    }
    if (total_bytes_ > UINT64_MAX / 8U) {
      throw std::overflow_error("SHA-256 bit length overflows uint64");
    }
    const auto bit_length = total_bytes_ * 8U;
    buffer_[buffer_size_++] = std::byte{0x80};
    if (buffer_size_ > 56) {
      std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                buffer_.end(),
                std::byte{});
      transform(buffer_.data());
      buffer_size_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
              buffer_.begin() + 56,
              std::byte{});
    for (std::size_t i = 0; i < 8; ++i) {
      buffer_[63 - i] = static_cast<std::byte>((bit_length >> (i * 8U)) & 0xffU);
    }
    transform(buffer_.data());
    for (std::size_t i = 0; i < state_.size(); ++i) {
      digest_.bytes[i * 4] = static_cast<std::byte>((state_[i] >> 24U) & 0xffU);
      digest_.bytes[i * 4 + 1] = static_cast<std::byte>((state_[i] >> 16U) & 0xffU);
      digest_.bytes[i * 4 + 2] = static_cast<std::byte>((state_[i] >> 8U) & 0xffU);
      digest_.bytes[i * 4 + 3] = static_cast<std::byte>(state_[i] & 0xffU);
    }
    finalized_ = true;
    return digest_;
  }

 private:
  [[nodiscard]] static constexpr auto choose(std::uint32_t x,
                                             std::uint32_t y,
                                             std::uint32_t z) noexcept -> std::uint32_t {
    return (x & y) ^ (~x & z);
  }

  [[nodiscard]] static constexpr auto majority(std::uint32_t x,
                                               std::uint32_t y,
                                               std::uint32_t z) noexcept -> std::uint32_t {
    return (x & y) ^ (x & z) ^ (y & z);
  }

  [[nodiscard]] static constexpr auto big_sigma0(std::uint32_t value) noexcept -> std::uint32_t {
    return std::rotr(value, 2) ^ std::rotr(value, 13) ^ std::rotr(value, 22);
  }

  [[nodiscard]] static constexpr auto big_sigma1(std::uint32_t value) noexcept -> std::uint32_t {
    return std::rotr(value, 6) ^ std::rotr(value, 11) ^ std::rotr(value, 25);
  }

  [[nodiscard]] static constexpr auto small_sigma0(std::uint32_t value) noexcept -> std::uint32_t {
    return std::rotr(value, 7) ^ std::rotr(value, 18) ^ (value >> 3U);
  }

  [[nodiscard]] static constexpr auto small_sigma1(std::uint32_t value) noexcept -> std::uint32_t {
    return std::rotr(value, 17) ^ std::rotr(value, 19) ^ (value >> 10U);
  }

  void transform(const std::byte *block) noexcept {
    static constexpr std::array<std::uint32_t, 64>
        kRound{0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
               0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
               0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
               0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
               0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
               0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
               0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
               0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
               0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
               0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
               0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      words[i] = (std::to_integer<std::uint32_t>(block[i * 4]) << 24U) |
                 (std::to_integer<std::uint32_t>(block[i * 4 + 1]) << 16U) |
                 (std::to_integer<std::uint32_t>(block[i * 4 + 2]) << 8U) |
                 std::to_integer<std::uint32_t>(block[i * 4 + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      words[i] =
          small_sigma1(words[i - 2]) + words[i - 7] + small_sigma0(words[i - 15]) + words[i - 16];
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const auto temp1 = h + big_sigma1(e) + choose(e, f, g) + kRound[i] + words[i];
      const auto temp2 = big_sigma0(a) + majority(a, b, c);
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{0x6a09e667U,
                                      0xbb67ae85U,
                                      0x3c6ef372U,
                                      0xa54ff53aU,
                                      0x510e527fU,
                                      0x9b05688cU,
                                      0x1f83d9abU,
                                      0x5be0cd19U};
  std::array<std::byte, 64> buffer_{};
  std::size_t buffer_size_{};
  std::uint64_t total_bytes_{};
  bool finalized_{};
  Sha256Digest digest_{};
};

[[nodiscard]] inline auto sha256(std::span<const std::byte> value) -> Sha256Digest {
  Sha256 hasher;
  hasher.update(value);
  return hasher.finalize();
}

[[nodiscard]] inline auto sha256(std::string_view value) -> Sha256Digest {
  Sha256 hasher;
  hasher.update(value);
  return hasher.finalize();
}

[[nodiscard]] inline auto sha256_file(const std::filesystem::path &path) -> Sha256Digest {
  std::error_code ec;
  if (std::filesystem::is_symlink(path, ec)) {
    throw std::invalid_argument("SHA-256 refuses symlink artifact: " + path.string());
  }
  if (!std::filesystem::is_regular_file(path, ec) || ec) {
    throw std::invalid_argument("SHA-256 artifact is not a regular file: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open artifact for SHA-256: " + path.string());
  }
  Sha256 hasher;
  std::array<std::byte, 1U << 16U> buffer{};
  for (;;) {
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      hasher.update(std::span(buffer.data(), static_cast<std::size_t>(count)));
    }
    if (input.eof()) {
      break;
    }
    if (!input) {
      throw std::runtime_error("failed while hashing artifact: " + path.string());
    }
  }
  return hasher.finalize();
}

}  // namespace alaya::internal::collection

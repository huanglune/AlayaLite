// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// collection_runtime_10: facade schema parse/read.
// One compile-cost-balanced Collection runtime unit; see CMakeLists.txt.

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::parse_u64(std::string_view value) -> std::uint64_t {
  if (value.empty()) {
    throw std::invalid_argument("canonical facade schema integer is empty");
  }
  std::uint64_t result{};
  for (const auto digit : value) {
    if (digit < '0' || digit > '9' ||
        result >
            (std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(digit - '0')) /
                10U) {
      throw std::invalid_argument("canonical facade schema integer is invalid");
    }
    result = result * 10U + static_cast<std::uint64_t>(digit - '0');
  }
  return result;
}

[[nodiscard]] auto Collection::read_facade_schema(const std::filesystem::path &root)
    -> core::Result<CollectionOptions> {
  try {
    const auto path = facade_schema_path(root);
    if (std::filesystem::file_size(path) > 64U * 1024U) {
      throw std::invalid_argument("canonical facade schema exceeds its size limit");
    }
    std::ifstream input(path, std::ios::binary);
    const std::string body{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (!input.eof() && !input) {
      throw std::runtime_error("cannot read canonical facade schema");
    }
    std::map<std::string, std::string, std::less<>> fields;
    std::string prefix;
    std::istringstream lines(body);
    for (std::string line; std::getline(lines, line);) {
      const auto equal = line.find('=');
      if (equal == std::string::npos || equal == 0 || equal + 1 == line.size() ||
          !fields.emplace(line.substr(0, equal), line.substr(equal + 1)).second) {
        throw std::invalid_argument("canonical facade schema contains an invalid field");
      }
      if (!line.starts_with("checksum=")) {
        prefix += line + "\n";
      }
    }
    const auto required = [&](std::string_view key) -> const std::string & {
      const auto found = fields.find(key);
      if (found == fields.end()) {
        throw std::invalid_argument("canonical facade schema is missing field " + std::string(key));
      }
      return found->second;
    };
    if ((fields.size() != 14 && fields.size() != 15) || required("format") != "1" ||
        required("public_version") != kCollectionPublicVersion ||
        required("checksum") != internal::collection::sha256(prefix).hex() ||
        parse_u64(required("active_segment_id")) != kActiveSegmentId ||
        parse_u64(required("active_generation")) != kActiveSegmentGeneration) {
      throw std::invalid_argument("canonical facade schema identity/checksum is invalid");
    }
    CollectionOptions options;
    options.root = root;
    options.dim = static_cast<std::uint32_t>(parse_u64(required("dim")));
    options.metric = static_cast<core::Metric>(parse_u64(required("metric")));
    options.scalar_type = static_cast<core::ScalarType>(parse_u64(required("scalar_type")));
    options.target_algorithm = parse_u64(required("target_algorithm"));
    options.quantization = static_cast<CollectionQuantization>(parse_u64(required("quantization")));
    options.build_threads = static_cast<std::uint32_t>(parse_u64(required("build_threads")));
    options.max_neighbors = static_cast<std::uint32_t>(parse_u64(required("max_neighbors")));
    options.ef_construction = static_cast<std::uint32_t>(parse_u64(required("ef_construction")));
    options.max_logical_id_bytes = parse_u64(required("max_logical_id_bytes"));
    // B-08: 14 fields = pre-2B / flat active (default flat); 15 fields carries the
    // explicit active engine. An old binary rejects the 15th field on the strict
    // count above, so a laser collection fails-closed rather than silently
    // reverting to flat.
    options.active_engine =
        fields.size() == 15 ? parse_u64(required("active_engine")) : core::algorithm::flat;
    auto status = validate_options(options, core::OperationStage::open);
    if (!status.ok()) {
      return status;
    }
    return options;
  } catch (const std::invalid_argument &exception) {
    return error(core::StatusCode::corruption,
                 core::OperationStage::open,
                 core::StatusDetail::malformed_struct,
                 exception.what());
  } catch (...) {
    return core::status_from_exception(core::OperationStage::open);
  }
}
}  // namespace alaya

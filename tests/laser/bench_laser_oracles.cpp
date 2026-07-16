// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bench_laser_oracles.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bench_laser_update_sift_support.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_query.hpp"
#include "index/graph/laser/qg/qg_scanner.hpp"
// The oracle is read-only. This include supplies the public v2 row-trailer and
// FastScan block-unpack helpers; qg_updater.hpp itself is deliberately unchanged.
#include "index/graph/laser/qg/qg_updater.hpp"
#include "index/graph/laser/space/l2.hpp"
#include "index/graph/laser/utils/buffer.hpp"

namespace alaya::laser::bench {
namespace {

struct DiskRow {
  std::vector<char> page;
  size_t slot = 0;
};

class DiskQGReader {
 public:
  DiskQGReader(const std::string &prefix, uint32_t degree, uint32_t main_dim, uint32_t full_dim)
      : degree_(degree),
        main_dim_(main_dim),
        full_dim_(full_dim),
        residual_dim_(full_dim - main_dim),
        padded_dim_(size_t{1} << ceil_log2(main_dim)),
        scanner_(padded_dim_, degree_),
        rotator_(main_dim_),
        node_len_((32 * main_dim_ + 32 * residual_dim_ + 128 * degree_ + degree_ * padded_dim_) /
                  8),
        code_offset_(full_dim_ * sizeof(float)),
        factor_offset_(code_offset_ + (padded_dim_ / 64) * 2 * degree_ * sizeof(float)),
        neighbor_offset_(factor_offset_ + 3 * degree_ * sizeof(float)),
        path_(prefix + "_R" + std::to_string(degree_) + "_MD" + std::to_string(main_dim_) +
              ".index") {
    if (main_dim_ == 0 || full_dim_ < main_dim_ || padded_dim_ % 64 != 0 || degree_ == 0 ||
        degree_ % kBatchSize != 0) {
      throw std::invalid_argument("oracle: unsupported QG dimensions/degree");
    }
    if (neighbor_offset_ + degree_ * sizeof(PID) != node_len_) {
      throw std::logic_error("oracle: computed row layout does not match node length");
    }
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("oracle: cannot open " + path_ + " errno=" + std::to_string(errno));
    }
    try {
      load_header();
      std::ifstream rotator_input(path_ + "_rotator", std::ios::binary);
      if (!rotator_input) throw std::runtime_error("oracle: cannot open rotator");
      rotator_.load(rotator_input);
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }

  DiskQGReader(const DiskQGReader &) = delete;
  auto operator=(const DiskQGReader &) -> DiskQGReader & = delete;

  ~DiskQGReader() {
    if (fd_ >= 0) ::close(fd_);
  }

  [[nodiscard]] size_t num_points() const { return num_points_; }
  [[nodiscard]] size_t degree() const { return degree_; }
  [[nodiscard]] size_t main_dim() const { return main_dim_; }
  [[nodiscard]] size_t full_dim() const { return full_dim_; }
  [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
  [[nodiscard]] PID entry_point() const { return entry_point_; }
  [[nodiscard]] bool is_v2() const { return v2_; }
  [[nodiscard]] size_t page_size() const { return page_size_; }
  [[nodiscard]] size_t nodes_per_page() const { return nodes_per_page_; }
  [[nodiscard]] const QGScanner &scanner() const { return scanner_; }
  [[nodiscard]] const FHTRotator &rotator() const { return rotator_; }

  void read_row(PID id, DiskRow &out) const {
    if (id >= num_points_) throw std::out_of_range("oracle: row id outside index");
    out.slot = id % nodes_per_page_;
    read_page(id / nodes_per_page_, out.page);
  }

  [[nodiscard]] const char *payload(const DiskRow &row) const {
    return row.page.data() + row.slot * node_len_;
  }

  [[nodiscard]] const float *raw(const DiskRow &row) const {
    return reinterpret_cast<const float *>(payload(row));
  }

  [[nodiscard]] const PID *ids(const DiskRow &row) const {
    return reinterpret_cast<const PID *>(payload(row) + neighbor_offset_);
  }

  [[nodiscard]] QGRowTrailer trailer(const DiskRow &row) const {
    if (!v2_) return QGRowTrailer{static_cast<uint16_t>(degree_), 0};
    const QGRowTrailer value =
        qg_read_page_trailer(row.page.data(), page_size_, nodes_per_page_, row.slot);
    if (value.valid_degree > degree_) {
      throw std::runtime_error("oracle: v2 valid_degree exceeds degree bound");
    }
    return value;
  }

  [[nodiscard]] bool hidden(const DiskRow &row) const {
    return v2_ && (trailer(row).flags & (kQGRowTombstone | kQGRowFree)) != 0;
  }

  [[nodiscard]] std::vector<size_t> valid_slots(const DiskRow &row) const {
    if (v2_) {
      const size_t count = trailer(row).valid_degree;
      std::vector<size_t> slots(count);
      std::iota(slots.begin(), slots.end(), size_t{0});
      return slots;
    }

    // Static v1 has no authoritative valid_degree. Mirror the updater's
    // migration-only ghost test so an actual edge to PID 0 is not mistaken
    // for an empty tail slot.
    const size_t words = padded_dim_ / 64;
    std::vector<uint64_t> bins(degree_ * words);
    const auto *code = reinterpret_cast<const uint8_t *>(payload(row) + code_offset_);
    for (size_t block = 0; block < degree_ / kBatchSize; ++block) {
      unpack_codes_block(padded_dim_,
                         code + block * padded_dim_ * 4,
                         bins.data() + block * kBatchSize * words);
    }
    const auto *factor = reinterpret_cast<const float *>(payload(row) + factor_offset_);
    const PID *neighbors = ids(row);
    std::vector<size_t> slots;
    slots.reserve(degree_);
    for (size_t slot = 0; slot < degree_; ++slot) {
      bool ghost = neighbors[slot] == 0 && factor[slot] == 0 && factor[degree_ + slot] == 0 &&
                   factor[2 * degree_ + slot] == 0;
      for (size_t word = 0; ghost && word < words; ++word) {
        ghost = bins[slot * words + word] == 0;
      }
      if (!ghost) slots.push_back(slot);
    }
    return slots;
  }

  [[nodiscard]] QGQuery prepare_query(const float *query) const {
    QGQuery prepared(query, padded_dim_);
    prepared.query_prepare(rotator_, scanner_);
    float residual_sqr = 0;
    for (size_t j = main_dim_; j < full_dim_; ++j) residual_sqr += query[j] * query[j];
    prepared.set_sqr_qr(residual_sqr);
    return prepared;
  }

  void scan(const QGQuery &query, const DiskRow &row, std::vector<float> &distances) const {
    distances.resize(degree_);
    scanner_.scan_neighbors(distances.data(),
                            query.lut().data(),
                            space::l2_sqr(query.query_data(), raw(row), main_dim_),
                            query.lower_val(),
                            query.width(),
                            query.sqr_qr(),
                            query.sumq(),
                            reinterpret_cast<const uint8_t *>(payload(row) + code_offset_),
                            reinterpret_cast<const float *>(payload(row) + factor_offset_));
  }

  void load_live_and_indegree(std::vector<uint8_t> &live, std::vector<uint32_t> &indegree) const {
    if (!v2_) throw std::runtime_error("twohop_oracle requires a post-churn v2 index");
    live.assign(num_points_, 0);
    indegree.assign(num_points_, 0);
    const size_t pages = (num_points_ + nodes_per_page_ - 1) / nodes_per_page_;
    std::vector<char> page;
    for (size_t page_index = 0; page_index < pages; ++page_index) {
      read_page(page_index, page);
      for (size_t slot = 0; slot < nodes_per_page_; ++slot) {
        const size_t raw_id = page_index * nodes_per_page_ + slot;
        if (raw_id >= num_points_) break;
        const QGRowTrailer row_trailer =
            qg_read_page_trailer(page.data(), page_size_, nodes_per_page_, slot);
        if (row_trailer.valid_degree > degree_) {
          throw std::runtime_error("oracle: corrupt valid_degree during indegree scan");
        }
        const bool source_live = (row_trailer.flags & (kQGRowTombstone | kQGRowFree)) == 0;
        live[raw_id] = static_cast<uint8_t>(source_live);
        if (!source_live) continue;
        const char *row = page.data() + slot * node_len_;
        const auto *neighbors = reinterpret_cast<const PID *>(row + neighbor_offset_);
        for (size_t j = 0; j < row_trailer.valid_degree; ++j) {
          if (neighbors[j] < indegree.size()) ++indegree[neighbors[j]];
        }
      }
    }
  }

 private:
  void read_exact(uint64_t offset, void *buffer, size_t length) const {
    size_t done = 0;
    while (done < length) {
      const ssize_t count = ::pread(fd_,
                                    static_cast<char *>(buffer) + done,
                                    length - done,
                                    static_cast<off_t>(offset + done));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) {
        throw std::runtime_error("oracle: short pread at offset " + std::to_string(offset));
      }
      done += static_cast<size_t>(count);
    }
  }

  void read_page(size_t page_index, std::vector<char> &page) const {
    page.resize(page_size_);
    read_exact(kSectorLen + page_index * page_size_, page.data(), page.size());
  }

  void load_header() {
    std::array<char, kSectorLen> header{};
    read_exact(0, header.data(), header.size());
    struct stat stat_buf{};
    if (::fstat(fd_, &stat_buf) != 0 || stat_buf.st_size < static_cast<off_t>(kSectorLen)) {
      throw std::runtime_error("oracle: cannot stat index");
    }
    const uint64_t file_size = static_cast<uint64_t>(stat_buf.st_size);

    QGSuperblockV2 superblock;
    const int superblock_slot = select_qg_superblock(header.data(), superblock);
    if (superblock_slot >= 0) {
      v2_ = true;
      num_points_ = static_cast<size_t>(superblock.num_points);
      entry_point_ = static_cast<PID>(superblock.entry_point);
      nodes_per_page_ = static_cast<size_t>(superblock.node_per_page);
      page_size_ = static_cast<size_t>(superblock.page_size);
      if (superblock.dimension != main_dim_ || superblock.node_len != node_len_ ||
          superblock.file_size != file_size) {
        throw std::runtime_error("oracle: v2 header/layout mismatch");
      }
    } else {
      if (qg_header_has_v2_magic(header.data())) {
        throw std::runtime_error("oracle: invalid v2 superblocks");
      }
      std::array<uint64_t, kSectorLen / sizeof(uint64_t)> metadata{};
      std::memcpy(metadata.data(), header.data(), header.size());
      num_points_ = static_cast<size_t>(metadata[0]);
      entry_point_ = static_cast<PID>(metadata[2]);
      nodes_per_page_ = static_cast<size_t>(metadata[4]);
      if (metadata[1] != main_dim_ || metadata[3] != node_len_ || metadata[8] != file_size ||
          nodes_per_page_ == 0) {
        throw std::runtime_error("oracle: v1 header/layout mismatch");
      }
      const size_t pages = (num_points_ + nodes_per_page_ - 1) / nodes_per_page_;
      if (pages == 0 || (file_size - kSectorLen) % pages != 0) {
        throw std::runtime_error("oracle: invalid v1 file geometry");
      }
      page_size_ = static_cast<size_t>((file_size - kSectorLen) / pages);
    }
    if (nodes_per_page_ * node_len_ > page_size_ || page_size_ % kSectorLen != 0) {
      throw std::runtime_error("oracle: unsupported page geometry");
    }
  }

  uint32_t degree_;
  uint32_t main_dim_;
  uint32_t full_dim_;
  size_t residual_dim_;
  size_t padded_dim_;
  QGScanner scanner_;
  FHTRotator rotator_;
  size_t node_len_;
  size_t code_offset_;
  size_t factor_offset_;
  size_t neighbor_offset_;
  std::string path_;
  int fd_ = -1;
  size_t num_points_ = 0;
  size_t nodes_per_page_ = 0;
  size_t page_size_ = 0;
  PID entry_point_ = 0;
  bool v2_ = false;
};

float exact_distance(const float *lhs, const float *rhs, size_t dimension) {
  return space::l2_sqr(lhs, rhs, dimension);
}

// This bench is compiled with -Ofast, whose finite-math-only assumption can
// fold std::isfinite() to true. Inspect the IEEE-754 exponent bits instead so
// degenerate RaBitQ payloads remain observable in the oracle statistics.
bool finite_float(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000U) != 0x7f800000U;
}

std::vector<double> average_ranks(const std::vector<float> &values) {
  std::vector<size_t> order(values.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    if (values[lhs] != values[rhs]) return values[lhs] < values[rhs];
    return lhs < rhs;
  });
  std::vector<double> ranks(values.size());
  size_t begin = 0;
  while (begin < order.size()) {
    size_t end = begin + 1;
    while (end < order.size() && values[order[end]] == values[order[begin]]) ++end;
    const double rank = 0.5 * static_cast<double>(begin + end - 1);
    for (size_t i = begin; i < end; ++i) ranks[order[i]] = rank;
    begin = end;
  }
  return ranks;
}

double spearman(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size() || lhs.size() < 2) return 0;
  const std::vector<double> lhs_rank = average_ranks(lhs);
  const std::vector<double> rhs_rank = average_ranks(rhs);
  const double mean = 0.5 * static_cast<double>(lhs.size() - 1);
  double covariance = 0;
  double lhs_variance = 0;
  double rhs_variance = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const double l = lhs_rank[i] - mean;
    const double r = rhs_rank[i] - mean;
    covariance += l * r;
    lhs_variance += l * l;
    rhs_variance += r * r;
  }
  const double denominator = std::sqrt(lhs_variance * rhs_variance);
  return denominator == 0 ? 0 : covariance / denominator;
}

double topk_overlap(const std::vector<float> &estimated,
                    const std::vector<float> &exact,
                    size_t requested_k) {
  const size_t k = std::min(requested_k, exact.size());
  if (k == 0) return 0;
  std::vector<size_t> estimated_order(exact.size());
  std::vector<size_t> exact_order(exact.size());
  std::iota(estimated_order.begin(), estimated_order.end(), size_t{0});
  std::iota(exact_order.begin(), exact_order.end(), size_t{0});
  const auto rank_by = [](const std::vector<float> &values, size_t lhs, size_t rhs) {
    if (values[lhs] != values[rhs]) return values[lhs] < values[rhs];
    return lhs < rhs;
  };
  std::partial_sort(estimated_order.begin(),
                    estimated_order.begin() + static_cast<std::ptrdiff_t>(k),
                    estimated_order.end(),
                    [&](size_t lhs, size_t rhs) {
                      return rank_by(estimated, lhs, rhs);
                    });
  std::partial_sort(exact_order.begin(),
                    exact_order.begin() + static_cast<std::ptrdiff_t>(k),
                    exact_order.end(),
                    [&](size_t lhs, size_t rhs) {
                      return rank_by(exact, lhs, rhs);
                    });
  std::unordered_set<size_t> truth(exact_order.begin(),
                                   exact_order.begin() + static_cast<std::ptrdiff_t>(k));
  size_t hits = 0;
  for (size_t i = 0; i < k; ++i) hits += static_cast<size_t>(truth.count(estimated_order[i]) != 0);
  return static_cast<double>(hits) / static_cast<double>(k);
}

struct RankStats {
  size_t samples = 0;
  uint64_t candidate_count = 0;
  double top1 = 0;
  double top4 = 0;
  double top8 = 0;
  double rank_correlation = 0;
  double relative_error = 0;
  double absolute_error = 0;
  double exact_distance_sum = 0;
  uint64_t error_terms = 0;
  uint64_t invalid_estimates = 0;
  uint64_t distance_terms = 0;
  uint64_t near_zero_exact = 0;
  std::vector<double> relative_errors;
  double anchor_l2 = 0;

  void add(const std::vector<float> &estimated,
           const std::vector<float> &exact,
           float anchor_distance_sqr) {
    ++samples;
    candidate_count += exact.size();
    top1 += topk_overlap(estimated, exact, 1);
    top4 += topk_overlap(estimated, exact, 4);
    top8 += topk_overlap(estimated, exact, 8);
    rank_correlation += spearman(estimated, exact);
    anchor_l2 += std::sqrt(std::max(0.0F, anchor_distance_sqr));
    for (size_t i = 0; i < exact.size(); ++i) {
      ++distance_terms;
      if (estimated[i] == FLT_MAX) {
        ++invalid_estimates;
        continue;
      }
      const double exact_value = static_cast<double>(exact[i]);
      const double abs_error = std::abs(static_cast<double>(estimated[i]) - exact_value);
      const double rel_error = abs_error / std::max(1e-6, exact_value);
      relative_error += rel_error;
      relative_errors.push_back(rel_error);
      absolute_error += abs_error;
      exact_distance_sum += exact_value;
      near_zero_exact += static_cast<uint64_t>(exact_value <= 1e-6);
      ++error_terms;
    }
  }

  void print(const char *relation) const {
    const double denominator = static_cast<double>(samples);
    std::vector<double> sorted_relative_errors = relative_errors;
    std::sort(sorted_relative_errors.begin(), sorted_relative_errors.end());
    const double median_relative_error =
        sorted_relative_errors.empty()
            ? 0
            : sorted_relative_errors[(sorted_relative_errors.size() - 1) / 2];
    std::cout << "oracle2," << relation << ',' << samples << ','
              << (samples == 0 ? 0 : static_cast<double>(candidate_count) / denominator) << ','
              << (samples == 0 ? 0 : top1 / denominator) << ','
              << (samples == 0 ? 0 : top4 / denominator) << ','
              << (samples == 0 ? 0 : top8 / denominator) << ','
              << (samples == 0 ? 0 : rank_correlation / denominator) << ','
              << (error_terms == 0 ? 0 : relative_error / static_cast<double>(error_terms)) << ','
              << median_relative_error << ','
              << (exact_distance_sum == 0 ? 0 : absolute_error / exact_distance_sum) << ','
              << (error_terms == 0 ? 0 : static_cast<double>(near_zero_exact) / error_terms) << ','
              << (distance_terms == 0 ? 0 : static_cast<double>(invalid_estimates) / distance_terms)
              << ',' << (samples == 0 ? 0 : anchor_l2 / denominator) << '\n';
  }
};

std::vector<PID> row_neighbors(const DiskQGReader &reader, const DiskRow &row, size_t base_size) {
  const PID *ids = reader.ids(row);
  std::vector<PID> result;
  std::unordered_set<PID> seen;
  for (size_t slot : reader.valid_slots(row)) {
    const PID id = ids[slot];
    if (id < reader.num_points() && id < base_size && seen.insert(id).second) result.push_back(id);
  }
  return result;
}

bool evaluate_rank_sample(const DiskQGReader &reader,
                          const MappedFloatMatrix &base,
                          PID query_id,
                          PID anchor_id,
                          RankStats &stats) {
  DiskRow anchor_row;
  reader.read_row(anchor_id, anchor_row);
  if (reader.hidden(anchor_row)) return false;
  const PID *ids = reader.ids(anchor_row);
  const std::vector<size_t> slots = reader.valid_slots(anchor_row);
  std::unordered_set<PID> seen;
  std::vector<size_t> usable_slots;
  for (size_t slot : slots) {
    const PID id = ids[slot];
    // Maintenance filters the query node itself after scanning an anchor row;
    // exclude that trivial zero-distance candidate from the ranking oracle.
    if (id != query_id && id < base.n && id < reader.num_points() && seen.insert(id).second) {
      usable_slots.push_back(slot);
    }
  }
  if (usable_slots.size() < 8) return false;

  const float *query = base.row(query_id);
  QGQuery prepared = reader.prepare_query(query);
  std::vector<float> all_estimated;
  reader.scan(prepared, anchor_row, all_estimated);
  std::vector<float> estimated;
  std::vector<float> exact;
  estimated.reserve(usable_slots.size());
  exact.reserve(usable_slots.size());
  for (size_t slot : usable_slots) {
    const float estimate = all_estimated[slot];
    estimated.push_back(finite_float(estimate) ? estimate : FLT_MAX);
    exact.push_back(exact_distance(query, base.row(ids[slot]), reader.full_dim()));
  }
  stats.add(estimated, exact, exact_distance(query, base.row(anchor_id), reader.full_dim()));
  return true;
}

RankStats sample_relation(const DiskQGReader &reader,
                          const MappedFloatMatrix &base,
                          size_t requested_samples,
                          bool two_hop,
                          uint64_t seed) {
  std::mt19937_64 random(seed + (two_hop ? 0x9e3779b97f4a7c15ULL : 0));
  std::uniform_int_distribution<uint64_t> node_distribution(0,
                                                            std::min<uint64_t>(reader.num_points(),
                                                                               base.n) -
                                                                1);
  RankStats stats;
  const size_t max_attempts = std::max<size_t>(10000, requested_samples * 200);
  for (size_t attempt = 0; attempt < max_attempts && stats.samples < requested_samples; ++attempt) {
    const PID query_id = static_cast<PID>(node_distribution(random));
    DiskRow query_row;
    reader.read_row(query_id, query_row);
    if (reader.hidden(query_row)) continue;
    const std::vector<PID> first_hop = row_neighbors(reader, query_row, base.n);
    if (first_hop.empty()) continue;
    std::uniform_int_distribution<size_t> first_distribution(0, first_hop.size() - 1);
    PID anchor_id = first_hop[first_distribution(random)];
    if (two_hop) {
      const PID middle_id = anchor_id;
      DiskRow middle_row;
      reader.read_row(middle_id, middle_row);
      const std::vector<PID> second_hop = row_neighbors(reader, middle_row, base.n);
      if (second_hop.empty()) continue;
      std::unordered_set<PID> direct(first_hop.begin(), first_hop.end());
      std::vector<PID> strict_second_hop;
      for (PID id : second_hop) {
        if (id != query_id && direct.count(id) == 0) strict_second_hop.push_back(id);
      }
      if (strict_second_hop.empty()) continue;
      std::uniform_int_distribution<size_t> second_distribution(0, strict_second_hop.size() - 1);
      anchor_id = strict_second_hop[second_distribution(random)];
    }
    (void)evaluate_rank_sample(reader, base, query_id, anchor_id, stats);
  }
  if (stats.samples < requested_samples) {
    std::cerr << "[fastscan_oracle] accepted only " << stats.samples << '/' << requested_samples
              << " samples for " << (two_hop ? "2hop" : "1hop") << '\n';
  }
  return stats;
}

std::unordered_set<PID> as_set(const std::vector<PID> &values) {
  return std::unordered_set<PID>(values.begin(), values.end());
}

size_t intersection_size(const std::unordered_set<PID> &lhs, const std::unordered_set<PID> &rhs) {
  const auto *small = &lhs;
  const auto *large = &rhs;
  if (small->size() > large->size()) std::swap(small, large);
  size_t count = 0;
  for (PID id : *small) count += static_cast<size_t>(large->count(id) != 0);
  return count;
}

std::vector<PID> beam_candidates(const DiskQGReader &reader,
                                 const MappedFloatMatrix &base,
                                 PID query_id,
                                 size_t ef,
                                 size_t cap,
                                 const std::vector<uint8_t> &live) {
  const float *query = base.row(query_id);
  QGQuery prepared = reader.prepare_query(query);
  buffer::SearchBuffer search_pool(ef);
  search_pool.insert(reader.entry_point(), FLT_MAX);
  std::unordered_set<PID> visited;
  visited.reserve(ef * 8);
  std::vector<std::pair<float, PID>> captured;
  std::vector<float> approximate;
  DiskRow row;
  while (search_pool.has_next()) {
    const PID current = search_pool.pop();
    if (current >= reader.num_points() || visited.count(current) != 0) continue;
    visited.insert(current);
    reader.read_row(current, row);
    reader.scan(prepared, row, approximate);
    const PID *neighbors = reader.ids(row);
    for (size_t slot : reader.valid_slots(row)) {
      const PID neighbor = neighbors[slot];
      const float distance = approximate[slot];
      if (neighbor >= reader.num_points() || visited.count(neighbor) != 0 ||
          !finite_float(distance) || search_pool.is_full(distance)) {
        continue;
      }
      search_pool.insert(neighbor, distance);
    }
    captured.emplace_back(exact_distance(query, reader.raw(row), reader.full_dim()), current);
  }
  std::sort(captured.begin(), captured.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first != rhs.first ? lhs.first < rhs.first : lhs.second < rhs.second;
  });
  if (captured.size() > cap) captured.resize(cap);
  std::vector<PID> result;
  std::unordered_set<PID> seen;
  for (const auto &[distance, id] : captured) {
    (void)distance;
    if (id != query_id && id < live.size() && live[id] != 0 && seen.insert(id).second) {
      result.push_back(id);
    }
  }
  return result;
}

std::vector<PID> robust_prune(const MappedFloatMatrix &base,
                              PID query_id,
                              const std::unordered_set<PID> &candidates,
                              size_t target,
                              float alpha) {
  const float *query = base.row(query_id);
  std::vector<std::pair<float, PID>> ordered;
  ordered.reserve(candidates.size());
  for (PID id : candidates) {
    if (id != query_id && id < base.n) {
      ordered.emplace_back(exact_distance(query, base.row(id), base.dim), id);
    }
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first != rhs.first ? lhs.first < rhs.first : lhs.second < rhs.second;
  });
  std::vector<PID> selected;
  selected.reserve(target);
  const float alpha_sqr = alpha * alpha;
  for (const auto &[query_distance, id] : ordered) {
    if (selected.size() >= target) break;
    bool occluded = false;
    for (PID prior : selected) {
      if (alpha_sqr * exact_distance(base.row(prior), base.row(id), base.dim) <= query_distance) {
        occluded = true;
        break;
      }
    }
    if (!occluded) selected.push_back(id);
  }
  return selected;
}

struct AnchorExpansion {
  std::vector<PID> estimated;
  std::vector<PID> exact;
};

std::vector<AnchorExpansion> expand_anchors(const DiskQGReader &reader,
                                            const MappedFloatMatrix &base,
                                            PID query_id,
                                            const std::vector<uint8_t> &live,
                                            std::vector<PID> &old_neighbors) {
  DiskRow query_row;
  reader.read_row(query_id, query_row);
  const float *query = base.row(query_id);
  QGQuery prepared = reader.prepare_query(query);
  std::vector<float> approximate;
  reader.scan(prepared, query_row, approximate);
  const PID *query_ids = reader.ids(query_row);
  std::vector<std::pair<float, PID>> ranked_anchors;
  std::unordered_set<PID> seen_anchor;
  for (size_t slot : reader.valid_slots(query_row)) {
    const PID anchor = query_ids[slot];
    if (anchor >= live.size() || live[anchor] == 0 || anchor == query_id ||
        !finite_float(approximate[slot]) || !seen_anchor.insert(anchor).second) {
      continue;
    }
    ranked_anchors.emplace_back(approximate[slot], anchor);
    old_neighbors.push_back(anchor);
  }
  std::sort(ranked_anchors.begin(), ranked_anchors.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first != rhs.first ? lhs.first < rhs.first : lhs.second < rhs.second;
  });

  std::vector<AnchorExpansion> expansions;
  expansions.reserve(ranked_anchors.size());
  DiskRow anchor_row;
  for (const auto &[anchor_distance, anchor] : ranked_anchors) {
    (void)anchor_distance;
    reader.read_row(anchor, anchor_row);
    reader.scan(prepared, anchor_row, approximate);
    const PID *candidate_ids = reader.ids(anchor_row);
    std::vector<std::pair<float, PID>> estimated;
    std::vector<std::pair<float, PID>> exact;
    std::unordered_set<PID> seen;
    for (size_t slot : reader.valid_slots(anchor_row)) {
      const PID candidate = candidate_ids[slot];
      if (candidate >= live.size() || candidate >= base.n || live[candidate] == 0 ||
          candidate == query_id || !seen.insert(candidate).second) {
        continue;
      }
      const float estimate = approximate[slot];
      if (finite_float(estimate)) estimated.emplace_back(estimate, candidate);
      exact.emplace_back(exact_distance(query, base.row(candidate), base.dim), candidate);
    }
    const auto pair_less = [](const auto &lhs, const auto &rhs) {
      return lhs.first != rhs.first ? lhs.first < rhs.first : lhs.second < rhs.second;
    };
    std::sort(estimated.begin(), estimated.end(), pair_less);
    std::sort(exact.begin(), exact.end(), pair_less);
    AnchorExpansion expansion;
    for (const auto &[distance, id] : estimated) {
      (void)distance;
      expansion.estimated.push_back(id);
    }
    for (const auto &[distance, id] : exact) {
      (void)distance;
      expansion.exact.push_back(id);
    }
    expansions.push_back(std::move(expansion));
  }
  return expansions;
}

std::unordered_set<PID> merge_expansions(const std::vector<AnchorExpansion> &expansions,
                                         size_t anchor_count,
                                         size_t top_t,
                                         bool exact) {
  std::unordered_set<PID> result;
  const size_t count = std::min(anchor_count, expansions.size());
  result.reserve(count * top_t);
  for (size_t anchor = 0; anchor < count; ++anchor) {
    const std::vector<PID> &order = exact ? expansions[anchor].exact : expansions[anchor].estimated;
    for (size_t rank = 0; rank < std::min(top_t, order.size()); ++rank) result.insert(order[rank]);
  }
  return result;
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(quantile * static_cast<double>(values.size() - 1));
  return values[index];
}

struct CoverageStats {
  size_t samples = 0;
  uint64_t anchors_used = 0;
  uint64_t beam_size = 0;
  uint64_t estimated_size = 0;
  uint64_t exact_size = 0;
  uint64_t estimated_intersection = 0;
  uint64_t exact_intersection = 0;
  uint64_t estimated_exact_intersection = 0;
  uint64_t beam_edges = 0;
  uint64_t estimated_edges = 0;
  uint64_t exact_edges = 0;
  double estimated_coverage = 0;
  double exact_coverage = 0;
  double estimated_edge_recall = 0;
  double exact_edge_recall = 0;
  double estimated_edge_jaccard = 0;
  std::vector<double> coverage_samples;

  void add(const std::unordered_set<PID> &beam,
           const std::unordered_set<PID> &estimated,
           const std::unordered_set<PID> &exact,
           const std::vector<PID> &beam_selected,
           const std::vector<PID> &estimated_selected,
           const std::vector<PID> &exact_selected,
           size_t used_anchors) {
    ++samples;
    anchors_used += used_anchors;
    const size_t estimated_hits = intersection_size(beam, estimated);
    const size_t exact_hits = intersection_size(beam, exact);
    const size_t rank_agreement = intersection_size(estimated, exact);
    beam_size += beam.size();
    estimated_size += estimated.size();
    exact_size += exact.size();
    estimated_intersection += estimated_hits;
    exact_intersection += exact_hits;
    estimated_exact_intersection += rank_agreement;
    const double estimated_ratio =
        beam.empty() ? 0 : static_cast<double>(estimated_hits) / beam.size();
    const double exact_ratio = beam.empty() ? 0 : static_cast<double>(exact_hits) / beam.size();
    estimated_coverage += estimated_ratio;
    exact_coverage += exact_ratio;
    coverage_samples.push_back(estimated_ratio);

    const std::unordered_set<PID> beam_edge_set = as_set(beam_selected);
    const std::unordered_set<PID> estimated_edge_set = as_set(estimated_selected);
    const std::unordered_set<PID> exact_edge_set = as_set(exact_selected);
    const size_t estimated_edge_hits = intersection_size(beam_edge_set, estimated_edge_set);
    const size_t exact_edge_hits = intersection_size(beam_edge_set, exact_edge_set);
    beam_edges += beam_edge_set.size();
    estimated_edges += estimated_edge_set.size();
    exact_edges += exact_edge_set.size();
    estimated_edge_recall +=
        beam_edge_set.empty() ? 0 : static_cast<double>(estimated_edge_hits) / beam_edge_set.size();
    exact_edge_recall +=
        beam_edge_set.empty() ? 0 : static_cast<double>(exact_edge_hits) / beam_edge_set.size();
    const size_t edge_union =
        beam_edge_set.size() + estimated_edge_set.size() - estimated_edge_hits;
    estimated_edge_jaccard +=
        edge_union == 0 ? 0 : static_cast<double>(estimated_edge_hits) / edge_union;
  }

  void print(size_t anchors, size_t top_t) const {
    const double n = static_cast<double>(samples);
    const double rank_recall =
        exact_size == 0 ? 0 : static_cast<double>(estimated_exact_intersection) / exact_size;
    std::cout << "oracle3," << anchors << ',' << top_t << ',' << samples << ','
              << (samples == 0 ? 0 : static_cast<double>(anchors_used) / n) << ','
              << (samples == 0 ? 0 : static_cast<double>(beam_size) / n) << ','
              << (samples == 0 ? 0 : static_cast<double>(estimated_size) / n) << ','
              << (samples == 0 ? 0 : static_cast<double>(exact_size) / n) << ','
              << (samples == 0 ? 0 : estimated_coverage / n) << ','
              << percentile(coverage_samples, 0.50) << ','
              << (beam_size == 0 ? 0 : static_cast<double>(estimated_intersection) / beam_size)
              << ',' << (samples == 0 ? 0 : exact_coverage / n) << ','
              << (beam_size == 0 ? 0 : static_cast<double>(exact_intersection) / beam_size) << ','
              << rank_recall << ',' << (samples == 0 ? 0 : static_cast<double>(beam_edges) / n)
              << ',' << (samples == 0 ? 0 : static_cast<double>(estimated_edges) / n) << ','
              << (samples == 0 ? 0 : estimated_edge_recall / n) << ','
              << (samples == 0 ? 0 : estimated_edge_jaccard / n) << ','
              << (samples == 0 ? 0 : exact_edge_recall / n) << '\n';
  }
};

}  // namespace

int run_fastscan_oracle(const OracleConfig &config) {
  if (config.prefix.empty() || config.base.empty() || config.samples == 0) {
    throw std::invalid_argument("fastscan_oracle requires --prefix, --base, and --samples > 0");
  }
  MappedFloatMatrix base(config.base);
  const uint32_t main_dim = config.main_dim == 0 ? base.dim : config.main_dim;
  DiskQGReader reader(config.prefix, config.degree, main_dim, base.dim);
  if (reader.num_points() == 0 || reader.num_points() > base.n) {
    throw std::runtime_error("fastscan_oracle: index/base count mismatch");
  }
  std::cout << "[fastscan_oracle] points=" << reader.num_points() << " R=" << reader.degree()
            << " format=" << (reader.is_v2() ? "v2" : "v1") << " samples=" << config.samples
            << " seed=" << config.seed << '\n';
  const auto start = std::chrono::steady_clock::now();
  const RankStats one_hop = sample_relation(reader, base, config.samples, false, config.seed);
  const RankStats two_hop = sample_relation(reader, base, config.samples, true, config.seed);
  std::cout << "kind,relation,samples,mean_candidates,top1,top4,top8,spearman,mean_rel_error,"
               "median_rel_error,normalized_mae,near_zero_exact_rate,invalid_estimate_rate,"
               "mean_anchor_l2\n";
  one_hop.print("1hop");
  two_hop.print("2hop");
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "[fastscan_oracle] elapsed_s=" << seconds << '\n';
  return 0;
}

int run_twohop_oracle(const OracleConfig &config) {
  if (config.prefix.empty() || config.base.empty() || config.samples == 0 ||
      config.ef_maintenance == 0) {
    throw std::invalid_argument(
        "twohop_oracle requires --prefix, --base, --samples, and --ef_maintenance");
  }
  MappedFloatMatrix base(config.base);
  const uint32_t main_dim = config.main_dim == 0 ? base.dim : config.main_dim;
  DiskQGReader reader(config.prefix, config.degree, main_dim, base.dim);
  if (reader.num_points() > base.n) {
    throw std::runtime_error("twohop_oracle: PID/source mapping exceeds base file");
  }
  std::vector<uint8_t> live;
  std::vector<uint32_t> indegree;
  const auto scan_start = std::chrono::steady_clock::now();
  reader.load_live_and_indegree(live, indegree);
  std::vector<uint32_t> live_indegrees;
  live_indegrees.reserve(reader.num_points());
  for (size_t id = 0; id < reader.num_points(); ++id) {
    if (live[id] != 0) live_indegrees.push_back(indegree[id]);
  }
  if (live_indegrees.empty()) throw std::runtime_error("twohop_oracle: no live rows");
  std::sort(live_indegrees.begin(), live_indegrees.end());
  const uint32_t p10 =
      live_indegrees[static_cast<size_t>(0.10 * static_cast<double>(live_indegrees.size() - 1))];
  std::vector<PID> low_indegree;
  for (size_t id = 0; id < reader.num_points(); ++id) {
    if (live[id] != 0 && indegree[id] < p10) low_indegree.push_back(static_cast<PID>(id));
  }
  if (low_indegree.empty()) {
    throw std::runtime_error("twohop_oracle: indegree < p10 selected no rows");
  }
  std::mt19937_64 random(config.seed);
  std::shuffle(low_indegree.begin(), low_indegree.end(), random);
  std::cout << "[twohop_oracle] points=" << reader.num_points() << " live=" << live_indegrees.size()
            << " p10=" << p10 << " below_p10=" << low_indegree.size()
            << " ef=" << config.ef_maintenance << " requested_samples=" << config.samples
            << " scan_s="
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - scan_start).count()
            << '\n';

  const std::array<size_t, 3> anchor_counts = {4, 8, config.degree};
  const std::array<size_t, 2> top_counts = {4, 8};
  std::array<std::array<CoverageStats, 2>, 3> metrics;
  size_t accepted = 0;
  const size_t target = config.r_target == 0 ? config.degree : config.r_target;
  const size_t beam_cap = std::max(config.prune_pool_cap, config.ef_maintenance);
  const auto oracle_start = std::chrono::steady_clock::now();
  for (PID query_id : low_indegree) {
    if (accepted >= config.samples) break;
    std::vector<PID> beam_vector =
        beam_candidates(reader, base, query_id, config.ef_maintenance, beam_cap, live);
    if (beam_vector.empty()) continue;
    std::unordered_set<PID> beam = as_set(beam_vector);
    std::vector<PID> old_neighbors;
    const std::vector<AnchorExpansion> expansions =
        expand_anchors(reader, base, query_id, live, old_neighbors);
    if (expansions.empty()) continue;
    std::unordered_set<PID> old_set = as_set(old_neighbors);
    std::unordered_set<PID> beam_prune_pool = beam;
    beam_prune_pool.insert(old_set.begin(), old_set.end());
    const std::vector<PID> beam_selected =
        robust_prune(base, query_id, beam_prune_pool, target, config.alpha);

    for (size_t ai = 0; ai < anchor_counts.size(); ++ai) {
      for (size_t ti = 0; ti < top_counts.size(); ++ti) {
        const size_t anchor_count = anchor_counts[ai];
        const size_t top_t = top_counts[ti];
        std::unordered_set<PID> estimated =
            merge_expansions(expansions, anchor_count, top_t, false);
        std::unordered_set<PID> exact = merge_expansions(expansions, anchor_count, top_t, true);
        std::unordered_set<PID> estimated_prune_pool = estimated;
        std::unordered_set<PID> exact_prune_pool = exact;
        estimated_prune_pool.insert(old_set.begin(), old_set.end());
        exact_prune_pool.insert(old_set.begin(), old_set.end());
        const std::vector<PID> estimated_selected =
            robust_prune(base, query_id, estimated_prune_pool, target, config.alpha);
        const std::vector<PID> exact_selected =
            robust_prune(base, query_id, exact_prune_pool, target, config.alpha);
        metrics[ai][ti].add(beam,
                            estimated,
                            exact,
                            beam_selected,
                            estimated_selected,
                            exact_selected,
                            std::min(anchor_count, expansions.size()));
      }
    }
    ++accepted;
    if (accepted % 100 == 0) {
      std::cout << "[twohop_oracle] samples=" << accepted << '/' << config.samples << '\n';
    }
  }
  if (accepted < config.samples) {
    std::cerr << "[twohop_oracle] accepted only " << accepted << '/' << config.samples << '\n';
  }
  std::cout << "kind,A,T,samples,mean_anchors_used,mean_beam_size,mean_fastscan_2hop_size,"
               "mean_exact_2hop_size,"
               "fastscan_candidate_coverage_macro,fastscan_candidate_coverage_p50,"
               "fastscan_candidate_coverage_micro,exact_candidate_coverage_macro,"
               "exact_candidate_coverage_micro,fastscan_set_recall_vs_exact,mean_beam_edges,"
               "mean_fastscan_edges,fastscan_edge_recall,fastscan_edge_jaccard,"
               "exact_edge_recall\n";
  for (size_t ai = 0; ai < anchor_counts.size(); ++ai) {
    for (size_t ti = 0; ti < top_counts.size(); ++ti) {
      metrics[ai][ti].print(anchor_counts[ai], top_counts[ti]);
    }
  }
  std::cout
      << "[twohop_oracle] elapsed_s="
      << std::chrono::duration<double>(std::chrono::steady_clock::now() - oracle_start).count()
      << '\n';
  return 0;
}

}  // namespace alaya::laser::bench

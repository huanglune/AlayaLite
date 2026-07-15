// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include "../utils/prefetch.hpp"
#include "core/log.hpp"
#include "core/value_types.hpp"
#include "platform/detect.hpp"
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "space_concepts.hpp"
#include "storage/container/sequential_storage.hpp"
#include "utils/math.hpp"
#include "utils/prefetch.hpp"

namespace alaya {

/**
 * @brief Vector storage and distance computation backend for memory graph segments.
 *
 * @tparam DataType The data type for storing data points, with the default being float.
 * @tparam DistanceType The data type for storing distances, with the default being float.
 * @tparam IDType The data type for storing IDs, with the default being uint32_t.
 */
template <typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<DataType, IDType>>
class RawSpace {
 public:
  using DistDataType = DataType;
  using DataTypeAlias = DataType;
  using IDTypeAlias = IDType;
  using DistanceTypeAlias = DistanceType;
  using DistanceFunction = DistanceType (*)(const DataType *, const DataType *, std::size_t);

  DistanceFunction distance_calc_func_;

  IDType capacity_{0};
  uint32_t dim_{0};
  core::Metric metric_{core::Metric::l2};
  uint32_t data_size_{0};
  IDType item_cnt_{0};
  IDType delete_cnt_{0};
  DataStorage data_storage_;

 public:
  RawSpace() = default;

  RawSpace(IDType capacity, size_t dim, core::Metric metric)
      : capacity_(capacity), dim_(dim), metric_(metric) {
    data_size_ = dim * sizeof(DataType);
    distance_calc_func_ = simd::l2_sqr<DataType, DistanceType>;

    data_storage_.init(data_size_, capacity);

    if constexpr (!(std::is_same_v<DataType, float> || std::is_same_v<DataType, double>)) {
      if (metric_ == core::Metric::cosine) {
        LOG_ERROR("COS metric only support float or double");
        exit(-1);
      }
    }

    set_metric_function();
  }

  RawSpace(RawSpace &&other) = delete;
  RawSpace(const RawSpace &other) = delete;

  ~RawSpace() = default;

  void set_metric_function() {
    switch (metric_) {
      case core::Metric::l2:
        distance_calc_func_ = simd::l2_sqr<DataType, DistanceType>;
        break;
      case core::Metric::inner_product:
      case core::Metric::cosine:
        distance_calc_func_ = simd::ip_sqr<DataType, DistanceType>;
        break;
      default:
        break;
    }
  }

  void fit(const DataType *data, IDType item_cnt) {
    if (data == nullptr) {
      throw std::invalid_argument("Invalid or null vector data pointer.");
    }
    if (item_cnt > capacity_) {
      throw std::length_error("The number of data points exceeds the capacity of the space");
    }
    item_cnt_ = item_cnt;
    for (IDType i = 0; i < item_cnt_; ++i) {
      data_storage_.insert(data + (i * dim_));
    }
  }

  auto insert(const DataType *data) -> IDType {
    auto id = data_storage_.insert(data);
    item_cnt_++;
    return id;
  }

  auto remove(IDType id) -> IDType {
    delete_cnt_++;
    return data_storage_.remove(id);
  }

  auto get_data_by_id(IDType id) const -> DataType * { return data_storage_[id]; }

  auto get_distance(IDType i, IDType j) const -> DistanceType {
    return distance_calc_func_(get_data_by_id(i), get_data_by_id(j), dim_);
  }

  auto get_data_num() const -> IDType { return item_cnt_; }

  auto get_avl_data_num() -> IDType { return item_cnt_ - delete_cnt_; }

  auto get_capacity() const -> IDType { return capacity_; }

  auto get_data_size() const -> size_t { return data_size_; }

  auto get_dist_func() const -> DistanceFunction { return distance_calc_func_; }

  auto get_dim() const -> uint32_t { return dim_; }

  auto metric() const -> core::Metric {
    return metric_ == core::Metric::l2
               ? core::Metric::l2
               : (metric_ == core::Metric::inner_product ? core::Metric::inner_product
                                                         : core::Metric::cosine);
  }

  auto load(std::string_view filename) -> void {
    std::ifstream reader(std::string(filename), std::ios::binary);
    if (!reader.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    reader.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    reader.read(reinterpret_cast<char *>(&data_size_), sizeof(data_size_));
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    reader.read(reinterpret_cast<char *>(&delete_cnt_), sizeof(delete_cnt_));
    reader.read(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));

    data_storage_.load(reader);
    LOG_INFO("RawSpace is loaded from {}", filename);
  }

  auto save(std::string_view filename) -> void {
    std::ofstream writer(std::string(filename), std::ios::binary);
    if (!writer.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    writer.write(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    writer.write(reinterpret_cast<char *>(&data_size_), sizeof(data_size_));
    writer.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    writer.write(reinterpret_cast<char *>(&delete_cnt_), sizeof(delete_cnt_));
    writer.write(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));

    data_storage_.save(writer);
    LOG_INFO("RawSpace is saved to {}", filename);
  }

  struct QueryComputer {
    const RawSpace &distance_space_;
    DataType *query_ = nullptr;

    QueryComputer(const RawSpace &distance_space, const DataType *query)
        : distance_space_(distance_space) {
      auto aligned_size = math::round_up_pow2(distance_space_.data_size_, kAlignment);
      query_ = static_cast<DataType *>(alaya_aligned_alloc_impl(aligned_size, kAlignment));
      std::memcpy(query_, query, distance_space.data_size_);
    }

    QueryComputer(const RawSpace &distance_space, const IDType id)
        : distance_space_(distance_space) {
      size_t aligned_size = math::round_up_pow2(distance_space_.data_size_, kAlignment);
      query_ = static_cast<DataType *>(alaya_aligned_alloc_impl(aligned_size, kAlignment));
      std::memcpy(query_, distance_space.get_data_by_id(id), distance_space.data_size_);
    }

    ~QueryComputer() {
      if (query_ != nullptr) {
        alaya_aligned_free_impl(query_);
      }
    }

    auto operator()(IDType u) const -> DistanceType {
      if (!distance_space_.data_storage_.is_valid(u)) {
        return std::numeric_limits<float>::max();
      }
      return distance_space_.distance_calc_func_(query_,
                                                 distance_space_.get_data_by_id(u),
                                                 distance_space_.dim_);
    }
  };

  auto prefetch_by_id(IDType id) -> void { mem_prefetch_l1(get_data_by_id(id), data_size_ / 64); }

  auto prefetch_by_address(DataType *address) -> void { mem_prefetch_l1(address, data_size_ / 64); }

  auto get_query_computer(const DataType *query) const { return QueryComputer(*this, query); }

  auto get_query_computer(IDType id) const { return QueryComputer(*this, id); }
};

static_assert(Space<RawSpace<uint32_t, float, uint32_t>>);
static_assert(Space<RawSpace<uint32_t, float, uint64_t>>);
static_assert(Space<RawSpace<uint64_t, float, uint32_t>>);
static_assert(Space<RawSpace<uint64_t, float, uint64_t>>);
static_assert(Space<RawSpace<float, float, uint32_t>>);
static_assert(Space<RawSpace<float, float, uint64_t>>);
static_assert(Space<RawSpace<double, float, uint32_t>>);
static_assert(Space<RawSpace<double, float, uint64_t>>);

}  // namespace alaya

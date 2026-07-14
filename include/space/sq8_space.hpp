// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "space/quant/sq8.hpp"
#include "space_concepts.hpp"
#include "storage/sequential_storage.hpp"
#include "utils/log.hpp"
#include "utils/math.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform.hpp"
#include "utils/prefetch.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

template <typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<uint8_t, IDType>,
          typename ScalarDataType = EmptyScalarData>
class SQ8Space {
  static_assert(std::is_same_v<ScalarDataType, EmptyScalarData>,
                "ScalarDataType is deprecated; scalar data is managed by Collection");

 public:
  static constexpr bool has_scalar_data = false;

  using DataTypeAlias = DataType;
  using IDTypeAlias = IDType;
  using DistanceTypeAlias = DistanceType;
  using DistanceFunction = DistanceType (*)(const std::uint8_t *,
                                            const std::uint8_t *,
                                            std::size_t,
                                            const DataType *,
                                            const DataType *);

  using DistDataType = DataType;

  SQ8Space() = default;

  SQ8Space(IDType capacity, size_t dim, MetricType metric)
      : capacity_(capacity), dim_(dim), metric_(metric), quantizer_(dim) {
    data_size_ = dim_ * sizeof(uint8_t);
    data_storage_.init(data_size_, capacity);
    set_metric_function();
  }

  ~SQ8Space() = default;

  SQ8Space(SQ8Space &&other) = delete;
  SQ8Space(const SQ8Space &other) = delete;
  auto operator=(const SQ8Space &) -> SQ8Space & = delete;
  auto operator=(SQ8Space &&) -> SQ8Space & = delete;

  void set_metric_function() {
    switch (metric_) {
      case MetricType::L2:
        distance_calu_func_ = simd::l2_sqr_sq8<DataType, DistanceType>;
        break;
      case MetricType::COS:
      case MetricType::IP:
        distance_calu_func_ = simd::ip_sqr_sq8<DataType, DistanceType>;
        break;
      default:
        break;
    }
  }

  auto get_capacity() const -> IDType { return capacity_; }

  void fit(const DataType *data, IDType item_cnt) {
    if (data == nullptr) {
      throw std::invalid_argument("Invalid or null vector data pointer.");
    }

    if (item_cnt > capacity_) {
      throw std::length_error("The number of data points exceeds the capacity of the space");
    }
    item_cnt_ = item_cnt;

    quantizer_.fit(data, item_cnt);
    for (IDType i = 0; i < item_cnt; i++) {
      auto id = data_storage_.reserve();
      quantizer_.encode(data + (i * dim_), data_storage_[id]);
    }
  }

  auto get_data_by_id(IDType id) const -> uint8_t * { return data_storage_[id]; }

  auto get_distance(IDType i, IDType j) const -> DistanceType {
    return distance_calu_func_(get_data_by_id(i),
                               get_data_by_id(j),
                               dim_,
                               quantizer_.get_min(),
                               quantizer_.get_max());
  }

  auto get_data_num() const -> IDType { return item_cnt_; }

  auto get_data_size() const -> size_t { return data_size_; }

  auto get_dist_func() const -> DistanceFunction { return distance_calu_func_; }

  auto get_dim() const -> uint32_t { return dim_; }

  auto metric() const -> core::Metric {
    return metric_ == MetricType::L2
               ? core::Metric::l2
               : (metric_ == MetricType::IP ? core::Metric::inner_product : core::Metric::cosine);
  }

  auto get_quantizer() const -> SQ8Quantizer<DataType> { return quantizer_; }

  auto insert(DataType *data) -> IDType {
    auto id = data_storage_.reserve();
    if (id == static_cast<IDType>(-1)) {
      return static_cast<IDType>(-1);
    }
    item_cnt_++;
    quantizer_.encode(data, data_storage_[id]);
    return id;
  }

  auto remove(IDType id) -> IDType {
    delete_cnt_++;
    return data_storage_.remove(id);
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
    quantizer_.load(reader);
    LOG_INFO("SQ8Space is loaded from {}", filename);
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
    quantizer_.save(writer);
    LOG_INFO("SQ8Space is saved to {}", filename);
  }

  struct QueryComputer {
    const SQ8Space &distance_space_;
    uint8_t *query_ = nullptr;

    QueryComputer(const SQ8Space &distance_space, const DataType *query)
        : distance_space_(distance_space) {
      size_t aligned_size = math::round_up_pow2(distance_space_.get_data_size(), 64);
      query_ = static_cast<uint8_t *>(alaya_aligned_alloc_impl(aligned_size, 64));
      distance_space.get_quantizer().encode(query, query_);
    }

    QueryComputer(const SQ8Space &distance_space, const IDType id)
        : distance_space_(distance_space) {
      size_t aligned_size = math::round_up_pow2(distance_space_.get_data_size(), 64);
      query_ = static_cast<uint8_t *>(alaya_aligned_alloc_impl(aligned_size, 64));
      std::memcpy(query_, distance_space_.get_data_by_id(id), distance_space_.get_data_size());
    }

    ~QueryComputer() {
      if (query_ != nullptr) {
        alaya_aligned_free_impl(query_);
      }
    }

    auto operator()(IDType u) const -> DistanceType {
      return distance_space_.distance_calu_func_(query_,
                                                 distance_space_.get_data_by_id(u),
                                                 distance_space_.get_dim(),
                                                 distance_space_.get_quantizer().get_min(),
                                                 distance_space_.get_quantizer().get_max());
    }
  };

  auto prefetch_by_id(IDType id) -> void { mem_prefetch_l1(get_data_by_id(id), data_size_ / 64); }

  auto prefetch_by_address(DataType *address) -> void { mem_prefetch_l1(address, data_size_ / 64); }

  auto get_query_computer(const DataType *query) const { return QueryComputer(*this, query); }

  auto get_query_computer(const IDType id) const { return QueryComputer(*this, id); }

 private:
  IDType capacity_{0};
  uint32_t dim_{0};
  MetricType metric_{MetricType::L2};

  DistanceFunction distance_calu_func_;
  uint32_t data_size_{0};
  IDType item_cnt_{0};
  IDType delete_cnt_{0};
  DataStorage data_storage_;
  SQ8Quantizer<DataType> quantizer_;
};
}  // namespace alaya

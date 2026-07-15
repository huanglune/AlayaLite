// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include "core/log.hpp"
#include "core/value_types.hpp"
#include "platform/detect.hpp"
#include "space_concepts.hpp"
#include "storage/container/sequential_storage.hpp"
#include "utils/math.hpp"
#include "utils/prefetch.hpp"

namespace alaya {

template <typename Traits,
          typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<uint8_t, IDType>>
class ScalarQuantizedSpace {
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
  using QuantizerType = typename Traits::template Quantizer<DataType>;

  ScalarQuantizedSpace() = default;

  ScalarQuantizedSpace(IDType capacity, size_t dim, core::Metric metric)
      : capacity_(capacity), dim_(dim), metric_(metric), quantizer_(dim) {
    data_size_ = Traits::data_size(dim);
    data_storage_.init(data_size_, capacity);
    set_metric_function();
  }

  ~ScalarQuantizedSpace() = default;

  ScalarQuantizedSpace(ScalarQuantizedSpace &&) = delete;
  ScalarQuantizedSpace(const ScalarQuantizedSpace &) = delete;
  auto operator=(const ScalarQuantizedSpace &) -> ScalarQuantizedSpace & = delete;
  auto operator=(ScalarQuantizedSpace &&) -> ScalarQuantizedSpace & = delete;

  void set_metric_function() {
    switch (metric_) {
      case core::Metric::l2:
        distance_calc_func_ = Traits::template l2_func<DataType, DistanceType>;
        break;
      case core::Metric::cosine:
      case core::Metric::inner_product:
        distance_calc_func_ = Traits::template ip_func<DataType, DistanceType>;
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
    return distance_calc_func_(
        get_data_by_id(i), get_data_by_id(j), dim_, quantizer_.get_min(), quantizer_.get_max());
  }

  auto get_data_num() const -> IDType { return item_cnt_; }
  auto get_data_size() const -> size_t { return data_size_; }
  auto get_dist_func() const -> DistanceFunction { return distance_calc_func_; }
  auto get_dim() const -> uint32_t { return dim_; }

  auto metric() const -> core::Metric {
    return metric_ == core::Metric::l2
               ? core::Metric::l2
               : (metric_ == core::Metric::inner_product ? core::Metric::inner_product
                                                          : core::Metric::cosine);
  }

  auto get_quantizer() const -> QuantizerType { return quantizer_; }

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
    LOG_INFO("{} is loaded from {}", Traits::name, filename);
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
    LOG_INFO("{} is saved to {}", Traits::name, filename);
  }

  struct QueryComputer {
    const ScalarQuantizedSpace &distance_space_;
    uint8_t *query_ = nullptr;

    QueryComputer(const ScalarQuantizedSpace &distance_space, const DataType *query)
        : distance_space_(distance_space) {
      size_t aligned_size = math::round_up_pow2(distance_space_.get_data_size(), 64);
      query_ = static_cast<uint8_t *>(alaya_aligned_alloc_impl(aligned_size, 64));
      distance_space.get_quantizer().encode(query, query_);
    }

    QueryComputer(const ScalarQuantizedSpace &distance_space, const IDType id)
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
      return distance_space_.distance_calc_func_(query_,
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
  core::Metric metric_{core::Metric::l2};
  DistanceFunction distance_calc_func_;
  uint32_t data_size_{0};
  IDType item_cnt_{0};
  IDType delete_cnt_{0};
  DataStorage data_storage_;
  QuantizerType quantizer_;
};

}  // namespace alaya

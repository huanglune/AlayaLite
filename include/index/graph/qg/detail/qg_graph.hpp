// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "index/neighbor.hpp"
#include "space/rabitq_space.hpp"
#include "space/space_concepts.hpp"

namespace alaya::detail {

// Segment-owned QG graph authority. Memory QG format v1 interleaves neighbor
// IDs with vectors, packed codes, and factors in RaBitQSpace. This view keeps
// those codec bytes in place while moving the graph API and entry point into
// QgSegment. RaBitQSpace retains graph codec hooks plus its legacy public API;
// Segment build/search reach them only through this graph authority.
template <typename SpaceType>
  requires Space<SpaceType> && is_rabitq_space_v<SpaceType>
class QgGraph {
 public:
  using DistanceType = typename SpaceType::DistanceTypeAlias;
  using IDType = typename SpaceType::IDTypeAlias;

  static constexpr std::size_t kDegreeBound = SpaceType::kDegreeBound;

  explicit QgGraph(std::shared_ptr<SpaceType> codec_space, IDType entry_point = IDType{})
      : codec_space_(std::move(codec_space)), entry_point_(entry_point) {
    if (codec_space_ == nullptr) {
      throw std::invalid_argument("QG graph requires a RaBitQ codec space");
    }
  }

  void set_ep(IDType entry_point) {
    entry_point_ = entry_point;
    // Format v1 persists this codec mirror in RaBitQSpace. Segment search never
    // reads it; legacy search continues to do so unchanged.
    codec_space_->set_ep(entry_point);
  }

  [[nodiscard]] auto get_ep() const noexcept -> IDType { return entry_point_; }

  [[nodiscard]] auto get_edges(IDType id) const -> const IDType * {
    return codec_space_->get_edges(id);
  }

  [[nodiscard]] auto get_edges(IDType id) -> IDType * { return codec_space_->get_edges(id); }

  void update_nei(IDType id, const std::vector<Neighbor<IDType, DistanceType>> &new_neighbors) {
    // Keep the retained quantization implementation and interleaved v1 layout
    // byte-for-byte: it writes neighbor IDs, codes, and factors together.
    codec_space_->update_nei(id, new_neighbors);
  }

 private:
  std::shared_ptr<SpaceType> codec_space_;
  IDType entry_point_{};
};

// Compatibility facade used only by QgSegment to inject its graph authority
// into the retained builder/search kernels. The underlying RaBitQSpace remains
// the vector/quantization owner; graph operations are routed to QgGraph.
template <typename SpaceType>
  requires Space<SpaceType> && is_rabitq_space_v<SpaceType>
class QgSegmentSpaceView {
 public:
  using DataTypeAlias = typename SpaceType::DataTypeAlias;
  using DistanceTypeAlias = typename SpaceType::DistanceTypeAlias;
  using IDTypeAlias = typename SpaceType::IDTypeAlias;
  using DistDataType = typename SpaceType::DistDataType;

  static constexpr std::size_t kDegreeBound = SpaceType::kDegreeBound;

  QgSegmentSpaceView(SpaceType &space, QgGraph<SpaceType> &graph)
      : space_(std::addressof(space)), graph_(std::addressof(graph)) {}

  [[nodiscard]] auto get_dim() const -> std::uint32_t { return space_->get_dim(); }
  [[nodiscard]] auto metric() const -> core::Metric { return space_->metric(); }
  [[nodiscard]] auto get_distance(IDTypeAlias lhs, IDTypeAlias rhs) const -> DistanceTypeAlias {
    return space_->get_distance(lhs, rhs);
  }

  [[nodiscard]] auto get_data_size() const -> std::size_t { return space_->get_data_size(); }
  [[nodiscard]] auto get_capacity() const -> std::size_t { return space_->get_capacity(); }
  [[nodiscard]] auto get_data_num() const -> IDTypeAlias { return space_->get_data_num(); }
  [[nodiscard]] auto get_data_by_id(IDTypeAlias id) const -> const DataTypeAlias * {
    return space_->get_data_by_id(id);
  }
  [[nodiscard]] auto get_data_by_id(IDTypeAlias id) -> DataTypeAlias * {
    return space_->get_data_by_id(id);
  }
  void fit(const DataTypeAlias *data, core::RowCount rows) {
    space_->fit(data, static_cast<IDTypeAlias>(rows));
  }

  [[nodiscard]] auto get_query_computer(const DataTypeAlias *query) const {
    return space_->get_query_computer(query);
  }
  [[nodiscard]] auto get_query_computer(IDTypeAlias id) const {
    return space_->get_query_computer(id);
  }
  [[nodiscard]] auto get_dist_func() const { return space_->get_dist_func(); }

  void set_ep(IDTypeAlias entry_point) { graph_->set_ep(entry_point); }
  [[nodiscard]] auto get_ep() const noexcept -> IDTypeAlias { return graph_->get_ep(); }
  [[nodiscard]] auto get_edges(IDTypeAlias id) const -> const IDTypeAlias * {
    return graph_->get_edges(id);
  }
  [[nodiscard]] auto get_edges(IDTypeAlias id) -> IDTypeAlias * { return graph_->get_edges(id); }
  void update_nei(IDTypeAlias id,
                  const std::vector<Neighbor<IDTypeAlias, DistanceTypeAlias>> &new_neighbors) {
    graph_->update_nei(id, new_neighbors);
  }

 private:
  SpaceType *space_;
  QgGraph<SpaceType> *graph_;
};

}  // namespace alaya::detail

namespace alaya {

template <typename SpaceType>
struct is_rabitq_space<detail::QgSegmentSpaceView<SpaceType>> : std::true_type {};

}  // namespace alaya

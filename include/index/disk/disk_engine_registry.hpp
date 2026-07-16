// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <string_view>

namespace alaya::internal::disk {

enum class EngineFeature {
  none,
  disk_flat_segment,
  flat = disk_flat_segment,
  disk_laser_segment,
  laser = disk_laser_segment,
  diskann_segment,
  diskann = diskann_segment,
  // Gate 8-B writer surface.  The ordinary DiskANN Segment gate deliberately
  // remains independent and enabled by default; mutable creation is internal
  // and opt-in until the crash/concurrency battery is complete.
  diskann_mutable_segment,
};

struct DiskEngineFeatureFlags {
  bool disk_flat_segment{true};
  bool disk_laser_segment{true};
  bool diskann_segment{true};
  bool diskann_mutable_segment{};

  [[nodiscard]] constexpr auto enabled(EngineFeature feature) const noexcept -> bool {
    switch (feature) {
      case EngineFeature::none:
        return true;
      case EngineFeature::disk_flat_segment:
        return disk_flat_segment;
      case EngineFeature::disk_laser_segment:
        return disk_laser_segment;
      case EngineFeature::diskann_segment:
        return diskann_segment;
      case EngineFeature::diskann_mutable_segment:
        return diskann_mutable_segment;
    }
    return false;
  }
};

struct FactoryIdentity {
  std::string_view declared_index_type{};
  std::string_view implementation_key{};
  std::string_view engine_factory_key{};
};

// The legacy factory remains an explicit compatibility entry. Disabling the
// new feature never silently selects it: construction through the new factory
// returns not_supported while DiskCollection v1 continues to call its own
// legacy factory unchanged.
struct FactoryRegistration {
  FactoryIdentity current{};
  FactoryIdentity legacy{};
  EngineFeature feature{EngineFeature::none};
  bool has_legacy_factory{};

  [[nodiscard]] constexpr auto current_enabled(const DiskEngineFeatureFlags &flags) const noexcept
      -> bool {
    return flags.enabled(feature);
  }
};

inline constexpr FactoryRegistration kDiskFlatRegistration{
    {"disk_flat", "disk_flat_segment", "flat"},
    {"disk_flat", "disk_flat_legacy", "disk_flat"},
    EngineFeature::disk_flat_segment,
    true,
};

inline constexpr FactoryRegistration kDiskLaserRegistration{
    {"disk_laser", "disk_laser_segment", "laser"},
    {"disk_laser", "disk_laser_legacy", "disk_laser"},
    EngineFeature::disk_laser_segment,
    true,
};

inline constexpr FactoryRegistration kDiskAnnRegistration{
    {"diskann", "diskann_segment", "diskann"},
    {"diskann", "diskann_index", "diskann"},
    EngineFeature::diskann_segment,
    true,
};

inline constexpr FactoryRegistration kDiskAnnMutableRegistration{
    {"diskann_mutable_internal", "diskann_mutable_segment", "diskann"},
    {},
    EngineFeature::diskann_mutable_segment,
    false,
};

}  // namespace alaya::internal::disk

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
  disk_vamana_segment,
  vamana = disk_vamana_segment,
  disk_laser_segment,
  laser = disk_laser_segment,
};

struct DiskEngineFeatureFlags {
  bool disk_flat_segment{true};
  bool disk_vamana_segment{true};
  bool disk_laser_segment{true};

  [[nodiscard]] constexpr auto enabled(EngineFeature feature) const noexcept -> bool {
    switch (feature) {
      case EngineFeature::none:
        return true;
      case EngineFeature::disk_flat_segment:
        return disk_flat_segment;
      case EngineFeature::disk_vamana_segment:
        return disk_vamana_segment;
      case EngineFeature::disk_laser_segment:
        return disk_laser_segment;
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

inline constexpr FactoryRegistration kDiskVamanaRegistration{
    {"disk_vamana", "disk_vamana_segment", "vamana"},
    {"disk_vamana", "disk_vamana_legacy", "disk_vamana"},
    EngineFeature::disk_vamana_segment,
    true,
};

inline constexpr FactoryRegistration kDiskLaserRegistration{
    {"disk_laser", "disk_laser_segment", "laser"},
    {"disk_laser", "disk_laser_legacy", "disk_laser"},
    EngineFeature::disk_laser_segment,
    true,
};

}  // namespace alaya::internal::disk

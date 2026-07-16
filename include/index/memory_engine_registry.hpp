// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <string_view>

namespace alaya::internal::memory {

// Standalone memory-engine switches are explicit instance configuration.
enum class EngineFeature {
  none,
  vamana_memory_segment,
  vamana_memory = vamana_memory_segment,
};

struct MemoryEngineFeatureFlags {
  bool vamana_memory_segment{true};

  [[nodiscard]] constexpr auto enabled(EngineFeature feature) const noexcept -> bool {
    switch (feature) {
      case EngineFeature::none:
        return true;
      case EngineFeature::vamana_memory_segment:
        return vamana_memory_segment;
    }
    return false;
  }
};

struct DispatchIdentity {
  std::string_view declared_index_type;
  std::string_view implementation_key;
  std::string_view engine_factory_key;
};

enum class EngineRole {
  searchable_segment,
  build_kernel,
};

// Non-dispatch Gate 5 registrations.  These entries deliberately describe
// roles instead of inferring runtime capabilities from an engine name.
struct StandaloneEngineRegistration {
  DispatchIdentity current;
  DispatchIdentity legacy;
  EngineFeature feature{EngineFeature::none};
  EngineRole role{EngineRole::searchable_segment};
  bool has_legacy_factory{};
  bool feature_switches_behavior{};
};

inline constexpr DispatchIdentity kNoLegacyIdentity{"n/a", "none", "none"};

// Vamana-memory has no prior user-facing memory factory.  Disabling its bit
// therefore means not_supported, not fallback to the disk adapter or another
// graph implementation.
inline constexpr StandaloneEngineRegistration kVamanaMemoryRegistration{
    {"vamana_memory", "vamana_mem_segment", "vamana"},
    kNoLegacyIdentity,
    EngineFeature::vamana_memory,
    EngineRole::searchable_segment,
    false,
    true,
};

}  // namespace alaya::internal::memory

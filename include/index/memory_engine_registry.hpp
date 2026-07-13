// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <string_view>

namespace alaya::internal::memory {

// Standalone memory-engine switches are explicit instance configuration.
enum class EngineFeature {
  none,
  knng_segment,
  knng = knng_segment,
  vamana_memory_segment,
  vamana_memory = vamana_memory_segment,
};

struct MemoryEngineFeatureFlags {
  bool knng_segment{true};
  bool vamana_memory_segment{true};

  [[nodiscard]] constexpr auto enabled(EngineFeature feature) const noexcept -> bool {
    switch (feature) {
      case EngineFeature::none:
        return true;
      case EngineFeature::knng_segment:
        return knng_segment;
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

// Design §7.1: NN-Descent is a BuildFactory/kernel.  A future user-facing
// index must independently prove searchability; this registration must not be
// used to manufacture a Searchable capability.  Its feature bit records
// kernel ownership only and therefore never switches runtime behavior.
inline constexpr StandaloneEngineRegistration kKnngKernelRegistration{
    {"knng", "nndescent_kernel", "knng"},
    kNoLegacyIdentity,
    EngineFeature::knng,
    EngineRole::build_kernel,
    false,
    false,
};

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

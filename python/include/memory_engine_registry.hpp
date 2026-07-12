// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <string_view>

namespace alaya::internal::memory {

// These switches follow CollectionFeatureFlags: configuration is explicit and
// each engine can be disabled without mutating process-global state.  A switch
// is consulted only after a Gate 5 row registers distinct current and legacy
// factories.  HNSW is intentionally absent: its public legacy builder was
// removed before Gate 5, so its rollback unit remains a source/git revert.
enum class EngineFeature {
  none,
  knng_segment,
  nsg_segment,
  fusion_segment,
  qg_segment,
  vamana_memory_segment,
};

struct MemoryEngineFeatureFlags {
  bool knng_segment{true};
  bool nsg_segment{true};
  bool fusion_segment{true};
  bool qg_segment{true};
  bool vamana_memory_segment{true};

  [[nodiscard]] constexpr auto enabled(EngineFeature feature) const noexcept -> bool {
    switch (feature) {
      case EngineFeature::none:
        return true;
      case EngineFeature::knng_segment:
        return knng_segment;
      case EngineFeature::nsg_segment:
        return nsg_segment;
      case EngineFeature::fusion_segment:
        return fusion_segment;
      case EngineFeature::qg_segment:
        return qg_segment;
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

// A generated dispatch row carries both identities even while they are equal.
// Migrating that row changes current, retains legacy, and assigns the engine's
// independent feature bit.  That creates a per-row new/legacy selection point
// without a global factory cutover.
struct FactoryRegistration {
  DispatchIdentity current;
  DispatchIdentity legacy;
  EngineFeature feature{EngineFeature::none};
  bool source_revert_only{};

  [[nodiscard]] constexpr auto runtime_rollback_supported() const noexcept -> bool {
    return !source_revert_only && feature != EngineFeature::none &&
           (current.implementation_key != legacy.implementation_key ||
            current.engine_factory_key != legacy.engine_factory_key);
  }

  [[nodiscard]] constexpr auto use_legacy(const MemoryEngineFeatureFlags &flags) const noexcept
      -> bool {
    return runtime_rollback_supported() && !flags.enabled(feature);
  }

  [[nodiscard]] constexpr auto selected(const MemoryEngineFeatureFlags &flags) const noexcept
      -> DispatchIdentity {
    return use_legacy(flags) ? legacy : current;
  }
};

}  // namespace alaya::internal::memory

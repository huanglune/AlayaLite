// SPDX-License-Identifier: AGPL-3.0-only
#include "recovery/recovery_manager.hpp"

#include <type_traits>

static_assert(std::is_class_v<alaya::recovery::RecoveryManager>);
static_assert(std::is_default_constructible_v<alaya::recovery::SnapshotManifest>);

auto golden_recovery_compile() -> int { return 0; }

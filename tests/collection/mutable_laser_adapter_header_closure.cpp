// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// Compile-time capability-surface proof (codex B-10): the active LASER adapter must
// satisfy every concept the Collection routes through it, not just the mutation
// bundle -- descriptor / search / batch_search / checkpoint / stats / close too.

#include "core/capabilities.hpp"
#include "index/collection/detail/mutable_laser_collection_adapter.hpp"

namespace {
using Adapter = alaya::internal::collection::detail::MutableLaserCollectionAdapter;
static_assert(alaya::core::DescriptorProvider<Adapter>, "adapter must provide a descriptor");
static_assert(alaya::core::Searchable<Adapter>, "adapter must be searchable");
static_assert(alaya::core::BatchSearchable<Adapter>, "adapter must support batch_search (B-10)");
static_assert(alaya::core::Mutable<Adapter>, "adapter must implement the five mutation methods");
static_assert(alaya::core::Checkpointable<Adapter>, "adapter must be checkpointable (B-10)");
static_assert(alaya::core::StatsProvider<Adapter>, "adapter must provide stats");
static_assert(alaya::core::Closable<Adapter>, "adapter must be closable/drainable");
}  // namespace

auto main() -> int { return 0; }

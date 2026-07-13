// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/disk/disk_vamana_segment.hpp"

static_assert(alaya::core::Searchable<alaya::disk::DiskVamanaSegment>);
static_assert(alaya::core::BatchSearchable<alaya::disk::DiskVamanaSegment>);
static_assert(alaya::core::Saveable<alaya::disk::DiskVamanaSegment>);
static_assert(!alaya::core::Exportable<alaya::disk::DiskVamanaSegment>);
static_assert(!alaya::core::Mutable<alaya::disk::DiskVamanaSegment>);

auto main() -> int { return 0; }

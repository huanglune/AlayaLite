// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <type_traits>

#include "alaya/collection.hpp"

static_assert(std::is_class_v<alaya::Collection>);
static_assert(alaya::kCollectionPublicVersion == "1.1.0");
static_assert(alaya::kCollectionLegacyRemovalVersion == "1.1.0");

int main() { return 0; }

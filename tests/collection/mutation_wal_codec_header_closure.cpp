// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/mutation_wal_codec.hpp"

int main() {
  return alaya::internal::collection::mutation_wal_codec_detail::kPayloadVersion == 1 ? 0 : 1;
}

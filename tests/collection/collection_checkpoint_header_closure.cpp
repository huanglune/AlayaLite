// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/collection_checkpoint.hpp"

int main() {
  alaya::internal::collection::CheckpointReceipt checkpoint;
  alaya::internal::collection::ArtifactManifestV2 manifest;
  alaya::internal::collection::CollectionCheckpointStore::apply_to_manifest(checkpoint, manifest);
  return 0;
}

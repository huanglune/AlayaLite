# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  recovery_test
  GTEST
  SRCS recovery_test.cpp
)

alaya_add_test(
  NAME recovery_test_wal
  TARGET recovery_test
  FILTER WalTest.*
  LABELS recovery
)
alaya_add_test(
  NAME recovery_test_snapshot_manifest
  TARGET recovery_test
  FILTER SnapshotManifestTest.*
  LABELS recovery
)
alaya_add_test(
  NAME recovery_test_recovery_manager
  TARGET recovery_test
  FILTER RecoveryManagerTest.*
  LABELS recovery
)

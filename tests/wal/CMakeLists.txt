# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# Framework-layer WAL envelope (wal/frame.hpp): golden byte format, structural
# scan, one-envelope/two-family scanning, and the collection cross-family guard.
alaya_cc_target(
  wal_frame_test
  SRCS wal_frame_test.cpp
  GTEST
)
alaya_add_test(
  NAME wal_frame_test
  TARGET wal_frame_test
  LABELS unit wal persistence
  TIMEOUT 60
)

// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Intentionally empty. Shared PCH owner targets link GTest::gtest_main, which
// supplies main(), while this translation unit gives CMake a matching C++
// compile profile on which to build the reusable precompiled header.

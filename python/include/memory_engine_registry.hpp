// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// The registry is a C++ engine boundary, not a Python-dispatch definition.
// Keep this forwarding header so existing binding includes remain stable.
#include "index/memory_engine_registry.hpp"

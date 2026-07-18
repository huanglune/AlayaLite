// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Stable, expensive dependencies shared by the mutable LASER and Collection
// tests. Keep qg_updater.hpp and collection.hpp out of this layer: they are the
// actively edited headers, and putting either one in a shared PCH serializes
// every dependent target behind regeneration of a very large .gch file.
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_query.hpp"
#include "index/graph/laser/qg/segment_op_wal.hpp"
#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "index/graph/laser/space/l2.hpp"
#include "platform/fs.hpp"
#include "simd/fastscan.hpp"
#include "wal/frame.hpp"

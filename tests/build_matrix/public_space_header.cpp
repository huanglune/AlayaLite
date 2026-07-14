// SPDX-License-Identifier: AGPL-3.0-only
#include "space/raw_space.hpp"
#include "space/rabitq_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"

#include <cstdint>
#include <type_traits>

#include "storage/sequential_storage.hpp"

using GoldenSpace = alaya::RawSpace<float,
                                    float,
                                    uint32_t,
                                    alaya::SequentialStorage<float, uint32_t>>;
static_assert(std::is_class_v<GoldenSpace>);
static_assert(alaya::DistanceSpace<GoldenSpace>);
static_assert(alaya::Quantizer<GoldenSpace>);
static_assert(alaya::VectorStore<GoldenSpace>);
static_assert(alaya::Space<GoldenSpace>);

using GoldenSQ4Space = alaya::SQ4Space<float, float, uint32_t>;
using GoldenSQ8Space = alaya::SQ8Space<float, float, uint32_t>;
using GoldenRaBitQSpace = alaya::RaBitQSpace<float, float, uint32_t>;
static_assert(alaya::Space<GoldenSQ4Space>);
static_assert(alaya::Space<GoldenSQ8Space>);
static_assert(alaya::Space<GoldenRaBitQSpace>);

auto golden_space_compile(GoldenSpace *space) -> size_t { return space->get_dim(); }

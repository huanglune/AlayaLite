// SPDX-License-Identifier: AGPL-3.0-only
#include "space/raw_space.hpp"

#include <cstdint>
#include <type_traits>

#include "storage/sequential_storage.hpp"
#include "utils/scalar_data.hpp"

using GoldenSpace = alaya::RawSpace<float,
                                    float,
                                    uint32_t,
                                    alaya::SequentialStorage<float, uint32_t>,
                                    alaya::EmptyScalarData>;
static_assert(std::is_class_v<GoldenSpace>);

auto golden_space_compile(GoldenSpace *space) -> size_t { return space->get_dim(); }

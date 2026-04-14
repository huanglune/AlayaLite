/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

#pragma once

#include <cstdint>

#include "utils/visited_set.hpp"

namespace alaya::diskann {

using VisitedVersionType = ::alaya::DenseVisitedSet::VersionType;
using VisitedList = ::alaya::DenseVisitedSet;
using GlobalVisitedList = ::alaya::GlobalDenseVisitedSet;

}  // namespace alaya::diskann

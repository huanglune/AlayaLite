// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Modifications Copyright 2025 AlayaDB.AI.

#pragma once

#include <memory>

#include "index/graph/laser/utils/aligned_file_reader.hpp"

#if defined(ALAYA_LASER_USE_THREADPOOL)
  #include "index/graph/laser/utils/threadpool_file_reader.hpp"
#endif

inline std::unique_ptr<AlignedFileReader> make_aligned_file_reader() {
#if defined(ALAYA_LASER_USE_THREADPOOL)
  return std::make_unique<ThreadPoolFileReader>();
#else
  return std::make_unique<LinuxAlignedFileReader>();
#endif
}

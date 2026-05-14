// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <memory>

#include "base_py_index.hpp"
#include "params.hpp"

namespace alaya {

class IndexFactory {
 public:
  static auto create(const IndexParams &params) -> std::unique_ptr<BasePyIndex>;
};

}  // namespace alaya

/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "executor/scheduler.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace alaya {

class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpus_ = {0};
    scheduler_ = std::make_unique<Scheduler>(cpus_, nullptr);
  }

  void TearDown() override { scheduler_->join(); }

  std::vector<CpuID> cpus_;
  std::unique_ptr<Scheduler> scheduler_;
};

}  // namespace alaya

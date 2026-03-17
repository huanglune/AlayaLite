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

#include "utils/types.hpp"
#include <gtest/gtest.h>
#include <string_view>

namespace alaya {

// Test operator[]
TEST(MetricTypeTest, NormalCase) {
  EXPECT_EQ(kMetricMap["L2"], MetricType::L2);
  EXPECT_EQ(kMetricMap["IP"], MetricType::IP);
  EXPECT_EQ(kMetricMap["COS"], MetricType::COS);
}

// Test string view copy
TEST(MetricTypeTest, StringViewCopyBehavior) {
  std::string test_key = "L2";
  MetricType metric = kMetricMap[test_key];
  EXPECT_EQ(metric, MetricType::L2);
  EXPECT_EQ(test_key, "L2");
}

// Test metric name retrieval
TEST(MetricTypeTest, GetMetricName) {
  MetricType metric = MetricType::IP;
  auto metric_name = kMetricMap[metric];
  EXPECT_EQ(metric_name, "IP");
}

}  // namespace alaya

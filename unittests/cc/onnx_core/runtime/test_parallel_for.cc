// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/parallel_for.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

TEST(ParallelFor, GrainIsMinimumBlockSize) {
  std::mutex mutex;
  std::vector<std::pair<int64_t, int64_t>> ranges;

  ParallelFor(100, 30, [&](int64_t begin, int64_t end) {
    std::lock_guard lock(mutex);
    ranges.emplace_back(begin, end);
  });

  std::sort(ranges.begin(), ranges.end());
  ASSERT_FALSE(ranges.empty());
  EXPECT_EQ(ranges.front().first, 0);
  EXPECT_EQ(ranges.back().second, 100);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    EXPECT_GE(ranges[i].second - ranges[i].first, 30);
    if (i != 0) {
      EXPECT_EQ(ranges[i - 1].second, ranges[i].first);
    }
  }
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

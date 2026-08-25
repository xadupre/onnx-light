// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/parallel_for.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
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

TEST(ThreadPool, WorkerStartupFailureRejectsPool) {
  ThreadPoolOptions options;
  options.worker_start = [](void *, int64_t, std::string &error) {
    error = "synthetic worker startup failure";
    return false;
  };
  EXPECT_THROW(ThreadPool(1, options), std::runtime_error);
}

TEST(ThreadPool, ParkImmediatelyRepeatedDispatchesStillCompleteWork) {
  ThreadPoolOptions options;
  options.spin_iterations = 0;
  options.spin_duration_ns = 0;
  ThreadPool pool(1, options);
  std::atomic<int> completed{0};

  constexpr int iterations = 100000;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    pool.Run(2, [&completed](int64_t) { completed.fetch_add(1, std::memory_order_relaxed); });
  }

  EXPECT_EQ(completed.load(std::memory_order_relaxed), 2 * iterations);
}

TEST(ThreadPool, SpinningLimitedWakeupsCompleteVaryingBlockCounts) {
  ThreadPoolOptions options;
  options.spin_iterations = 10000;
  options.spin_duration_ns = 0;
  ThreadPool pool(8, options);

  constexpr int iterations = 10000;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int64_t num_blocks = 2 + iteration % 8;
    std::vector<std::atomic<int>> visits(static_cast<std::size_t>(num_blocks));
    pool.Run(num_blocks, [&visits](int64_t block) {
      visits[static_cast<std::size_t>(block)].fetch_add(1, std::memory_order_relaxed);
    });
    for (const std::atomic<int> &visit : visits) {
      EXPECT_EQ(visit.load(std::memory_order_relaxed), 1);
    }
  }
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

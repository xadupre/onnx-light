// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/parallel_region_collector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

std::shared_ptr<CpuExecutor> MakeExecutor(CpuExecutorRegistry &registry, int32_t threads) {
  CpuExecutionPolicy policy;
  policy.num_threads = threads;
  policy.affinity_policy = CpuAffinityPolicy::kNone;
  return registry.Acquire(policy);
}

TEST(ParallelRegionCollector, RecordsSerialLimitedAndParallelRegions) {
  CpuExecutorRegistry registry(3);
  const auto serial_executor = MakeExecutor(registry, 1);
  const auto parallel_executor = MakeExecutor(registry, 2);
  ParallelRegionCollector collector(3);
  ParallelRegionCollectorScope collector_scope(&collector);

  {
    CpuExecutorScope executor_scope(serial_executor.get());
    ParallelFor(8, 1, [](int64_t, int64_t) {}, "serial");
  }
  {
    CpuExecutorScope executor_scope(parallel_executor.get());
    ParallelFor(2, 4, [](int64_t, int64_t) {}, "limited");
  }
  {
    CpuExecutorScope executor_scope(parallel_executor.get());
    ParallelFor(8, 1, [](int64_t, int64_t) {}, "parallel");
  }

  ASSERT_EQ(collector.events().size(), 3u);
  const auto &serial = collector.events()[0];
  EXPECT_EQ(serial.label, "serial");
  EXPECT_EQ(serial.requested_threads, 1);
  EXPECT_EQ(serial.admitted_threads, 1);
  EXPECT_EQ(serial.observed_threads, 1);
  EXPECT_FALSE(serial.nested_inline);

  const auto &limited = collector.events()[1];
  EXPECT_EQ(limited.requested_threads, 2);
  EXPECT_EQ(limited.admitted_threads, 1);
  EXPECT_EQ(limited.observed_threads, 1);
  EXPECT_FALSE(limited.nested_inline);

  const auto &parallel = collector.events()[2];
  EXPECT_EQ(parallel.total_iterations, 8);
  EXPECT_EQ(parallel.grain_size, 1);
  EXPECT_EQ(parallel.requested_threads, 2);
  EXPECT_EQ(parallel.admitted_threads, 2);
  EXPECT_EQ(parallel.observed_threads, 2);
  EXPECT_EQ(parallel.executor_instance_id, parallel_executor->instance_id());
  ASSERT_TRUE(parallel.wall_time_ns.has_value());
  EXPECT_GT(*parallel.wall_time_ns, 0u);
  ASSERT_TRUE(parallel.process_cpu_time_ns.has_value());
  ASSERT_TRUE(parallel.cpu_utilization.has_value());
  EXPECT_DOUBLE_EQ(*parallel.cpu_utilization, static_cast<double>(*parallel.process_cpu_time_ns) /
                                                  static_cast<double>(*parallel.wall_time_ns) /
                                                  parallel.admitted_threads);
  EXPECT_NE(parallel.region_id, 0u);
  EXPECT_NE(parallel.run_id, 0u);
  EXPECT_EQ(parallel.calling_thread_id, std::this_thread::get_id());
}

TEST(ParallelRegionCollector, PropagatesToWorkersAndMarksNestedInline) {
  CpuExecutorRegistry registry(1);
  const auto executor = MakeExecutor(registry, 2);
  ParallelRegionCollector collector(2);
  ParallelRegionCollectorScope collector_scope(&collector);
  CpuExecutorScope executor_scope(executor.get());
  std::atomic<int> nested_calls{0};

  ParallelFor(
      8, 1,
      [&nested_calls](int64_t begin, int64_t) {
        if (begin != 0) {
          ParallelFor(
              4, 1,
              [&nested_calls](int64_t, int64_t) {
                nested_calls.fetch_add(1, std::memory_order_relaxed);
              },
              "nested");
        }
      },
      "outer");

  EXPECT_EQ(nested_calls.load(std::memory_order_relaxed), 1);
  ASSERT_EQ(collector.events().size(), 2u);
  EXPECT_EQ(collector.events()[0].label, "nested");
  EXPECT_TRUE(collector.events()[0].nested_inline);
  EXPECT_EQ(collector.events()[0].admitted_threads, 1);
  EXPECT_EQ(collector.events()[0].observed_threads, 1);
  EXPECT_EQ(collector.events()[0].parent_region_id, collector.events()[1].region_id);
  EXPECT_EQ(collector.events()[0].run_id, collector.events()[1].run_id);
  EXPECT_NE(collector.events()[0].calling_thread_id, collector.events()[1].calling_thread_id);
  EXPECT_EQ(collector.events()[1].label, "outer");
  EXPECT_FALSE(collector.events()[1].nested_inline);
  EXPECT_EQ(collector.events()[1].admitted_threads, 2);
}

TEST(ParallelRegionCollector, ComputesNormalizedCpuUtilization) {
  const std::optional<double> utilization = ComputeCpuUtilization(uint64_t{600}, uint64_t{100}, 3);
  ASSERT_TRUE(utilization.has_value());
  EXPECT_DOUBLE_EQ(*utilization, 2.0);

  EXPECT_FALSE(ComputeCpuUtilization(std::nullopt, uint64_t{100}, 2).has_value());
  EXPECT_FALSE(ComputeCpuUtilization(uint64_t{100}, std::nullopt, 2).has_value());
  EXPECT_FALSE(ComputeCpuUtilization(uint64_t{100}, uint64_t{0}, 2).has_value());
  EXPECT_FALSE(ComputeCpuUtilization(uint64_t{100}, uint64_t{100}, 0).has_value());
}

TEST(ParallelRegionCollector, MarksLimitedStandaloneWorkerCallsNestedInline) {
  ThreadPool pool(1);
  ParallelRegionCollector collector(1);

  pool.Run(2, [&collector](int64_t block) {
    if (block == 1) {
      ParallelRegionCollectorScope collector_scope(&collector);
      ParallelFor(1, 2, [](int64_t, int64_t) {}, "limited-nested");
    }
  });

  ASSERT_EQ(collector.events().size(), 1u);
  EXPECT_EQ(collector.events()[0].label, "limited-nested");
  EXPECT_TRUE(collector.events()[0].nested_inline);
  EXPECT_EQ(collector.events()[0].admitted_threads, 1);
}

TEST(ParallelRegionCollector, PreservesCallerSourceLocation) {
  CpuExecutorRegistry registry(1);
  const auto executor = MakeExecutor(registry, 1);
  ParallelRegionCollector collector(1);
  ParallelRegionCollectorScope collector_scope(&collector);
  CpuExecutorScope executor_scope(executor.get());

  const uint_least32_t expected_line = __LINE__ + 1;
  ParallelFor(1, [](int64_t, int64_t) {}, "located");

  ASSERT_EQ(collector.events().size(), 1u);
  EXPECT_EQ(collector.events()[0].location.line(), expected_line);
  EXPECT_NE(std::string_view(collector.events()[0].location.file_name())
                .find("test_parallel_region_collector.cc"),
            std::string_view::npos);
}

TEST(ParallelRegionCollector, ReportsEventsDroppedBeyondCapacity) {
  CpuExecutorRegistry registry(1);
  const auto executor = MakeExecutor(registry, 1);
  ParallelRegionCollector collector(1);
  ParallelRegionCollectorScope collector_scope(&collector);
  CpuExecutorScope executor_scope(executor.get());

  for (int iteration = 0; iteration < 3; ++iteration) {
    ParallelFor(1, [](int64_t, int64_t) {});
  }

  EXPECT_EQ(collector.events().size(), 1u);
  EXPECT_EQ(collector.dropped_events(), 2u);
}

TEST(ParallelRegionCollector, DisabledPathDoesNotCreateEvents) {
  EXPECT_EQ(CurrentParallelRegionCollector(), nullptr);
  EXPECT_EQ(CurrentParallelRegionRunId(), 0u);
  EXPECT_EQ(CurrentParallelRegionId(), 0u);
  ParallelFor(8, 1, [](int64_t, int64_t) {}, "disabled");
  EXPECT_EQ(CurrentParallelRegionCollector(), nullptr);
  EXPECT_EQ(CurrentParallelRegionRunId(), 0u);
  EXPECT_EQ(CurrentParallelRegionId(), 0u);
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

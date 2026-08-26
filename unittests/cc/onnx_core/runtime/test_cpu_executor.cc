// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_executor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

ResolvedCpuExecutionPolicy NoAffinityPolicy(uint32_t participants) {
  CpuExecutionPolicy request;
  request.num_threads = static_cast<int32_t>(participants);
  request.affinity_policy = CpuAffinityPolicy::kNone;
  return ResolveCpuExecutionPolicy(request);
}

ResolvedCpuExecutionPolicy ParkImmediatelyPolicy(uint32_t participants) {
  CpuExecutionPolicy request;
  request.num_threads = static_cast<int32_t>(participants);
  request.affinity_policy = CpuAffinityPolicy::kNone;
  request.spin_policy = CpuSpinPolicy::kParkImmediately;
  return ResolveCpuExecutionPolicy(request);
}

struct RangeObservation {
  explicit RangeObservation(size_t size) : visits(size, 0) {}

  std::vector<int> visits;
  std::mutex mutex;
  std::set<std::thread::id> threads;
};

void ObserveRange(void *context, int64_t begin, int64_t end) {
  auto &observation = *static_cast<RangeObservation *>(context);
  {
    std::lock_guard<std::mutex> lock(observation.mutex);
    observation.threads.insert(std::this_thread::get_id());
  }
  for (int64_t index = begin; index < end; ++index) {
    ++observation.visits[static_cast<size_t>(index)];
  }
}

struct ExternalDispatchObservation {
  int64_t dispatched_blocks = 0;
  int64_t dispatches = 0;
};

void DispatchInline(void *context, int64_t num_blocks, void *block_context,
                    CpuParallelBlockFn block_function) {
  auto &observation = *static_cast<ExternalDispatchObservation *>(context);
  observation.dispatched_blocks += num_blocks;
  ++observation.dispatches;
  for (int64_t block = 0; block < num_blocks; ++block) {
    block_function(block_context, block);
  }
}

TEST(CpuExecutorRegistry, CompatibleResolvedPoliciesShareExecutor) {
  CpuExecutorRegistry registry(2);
  ResolvedCpuExecutionPolicy first_policy = NoAffinityPolicy(2);
  ResolvedCpuExecutionPolicy equivalent_policy = first_policy;
  equivalent_policy.request.num_threads = 0;
  equivalent_policy.diagnostics.emplace_back("not part of executor identity");

  std::shared_ptr<CpuExecutor> first = registry.Acquire(first_policy);
  std::shared_ptr<CpuExecutor> equivalent = registry.Acquire(equivalent_policy);

  EXPECT_EQ(first, equivalent);
  EXPECT_EQ(first->instance_id(), equivalent->instance_id());
  EXPECT_NE(first->instance_id(), 0u);
  EXPECT_EQ(registry.live_pool_count(), 1u);
}

TEST(CpuExecutorRegistry, IncompatiblePoliciesUseDistinctExecutors) {
  CpuExecutorRegistry registry(2);
  std::shared_ptr<CpuExecutor> serial = registry.Acquire(NoAffinityPolicy(1));
  std::shared_ptr<CpuExecutor> parallel = registry.Acquire(NoAffinityPolicy(2));

  EXPECT_NE(serial, parallel);
  EXPECT_NE(serial->instance_id(), parallel->instance_id());
  EXPECT_EQ(registry.live_pool_count(), 2u);
}

TEST(CpuExecutorRegistry, SpinBehaviorIsPartOfCompatibilityKey) {
  CpuExecutorRegistry registry(2);
  ResolvedCpuExecutionPolicy adaptive = NoAffinityPolicy(2);
  ResolvedCpuExecutionPolicy park = adaptive;
  park.request.spin_policy = CpuSpinPolicy::kParkImmediately;
  park.spin = ResolvedSpinPolicy{CpuSpinPolicy::kParkImmediately, 0, 0};

  EXPECT_NE(registry.Acquire(adaptive), registry.Acquire(park));
}

TEST(CpuExecutorRegistry, CapacityRejectsAdditionalLivePolicy) {
  CpuExecutorRegistry registry(2);
  std::shared_ptr<CpuExecutor> one = registry.Acquire(NoAffinityPolicy(1));
  std::shared_ptr<CpuExecutor> two = registry.Acquire(NoAffinityPolicy(2));
  EXPECT_THROW(registry.Acquire(NoAffinityPolicy(3)), std::runtime_error);

  one.reset();
  EXPECT_NO_THROW(one = registry.Acquire(NoAffinityPolicy(3)));
  EXPECT_EQ(registry.live_pool_count(), 2u);
}

TEST(CpuExecutorRegistry, BoundedSpinPoolSurvivesBetweenLeases) {
  CpuExecutorRegistry registry(1);
  std::weak_ptr<CpuExecutor> observer;
  uint64_t instance_id = 0;
  {
    std::shared_ptr<CpuExecutor> lease = registry.Acquire(NoAffinityPolicy(2));
    observer = lease;
    instance_id = lease->instance_id();
  }

  EXPECT_FALSE(observer.expired());
  EXPECT_EQ(registry.live_pool_count(), 1u);
  EXPECT_EQ(registry.Acquire(NoAffinityPolicy(2))->instance_id(), instance_id);
}

TEST(CpuExecutorRegistry, ParkImmediatelyPoolSurvivesBetweenLeases) {
  CpuExecutorRegistry registry(1);
  std::weak_ptr<CpuExecutor> observer;
  uint64_t instance_id = 0;
  {
    std::shared_ptr<CpuExecutor> lease = registry.Acquire(ParkImmediatelyPolicy(2));
    observer = lease;
    instance_id = lease->instance_id();
  }

  EXPECT_FALSE(observer.expired());
  EXPECT_EQ(registry.live_pool_count(), 1u);
  EXPECT_EQ(registry.Acquire(ParkImmediatelyPolicy(2))->instance_id(), instance_id);
}

TEST(CpuExecutorRegistry, CapacityEvictsIdleBoundedSpinPool) {
  CpuExecutorRegistry registry(1);
  std::weak_ptr<CpuExecutor> first;
  {
    std::shared_ptr<CpuExecutor> lease = registry.Acquire(NoAffinityPolicy(2));
    first = lease;
  }

  std::shared_ptr<CpuExecutor> replacement = registry.Acquire(NoAffinityPolicy(3));

  EXPECT_TRUE(first.expired());
  EXPECT_EQ(replacement->effective_threads(), 3u);
  EXPECT_EQ(registry.live_pool_count(), 1u);
}

TEST(CpuExecutorRegistry, CapacityDoesNotEvictLeasedParkImmediatelyPool) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> active = registry.Acquire(ParkImmediatelyPolicy(2));

  EXPECT_THROW(registry.Acquire(ParkImmediatelyPolicy(3)), std::runtime_error);
  EXPECT_EQ(registry.live_pool_count(), 1u);
}

TEST(CpuExecutorRegistry, SerialExecutorIsNotRetained) {
  CpuExecutorRegistry registry(1);
  std::weak_ptr<CpuExecutor> observer;
  {
    std::shared_ptr<CpuExecutor> lease = registry.Acquire(NoAffinityPolicy(1));
    observer = lease;
  }

  EXPECT_TRUE(observer.expired());
  EXPECT_EQ(registry.live_pool_count(), 0u);
}

TEST(CpuExecutorRegistry, RejectsZeroCapacity) {
  EXPECT_THROW(CpuExecutorRegistry(0), std::invalid_argument);
}

TEST(CpuExecutorRegistry, RequestOverloadResolvesPolicy) {
  CpuExecutorRegistry registry(1);
  CpuExecutionPolicy request;
  request.num_threads = 1;
  request.affinity_policy = CpuAffinityPolicy::kNone;

  std::shared_ptr<CpuExecutor> executor = registry.Acquire(request);

  EXPECT_EQ(executor->effective_threads(), 1u);
  EXPECT_EQ(executor->policy().request, request);
}

TEST(CpuExecutorRegistry, ResolvedWorkerAffinityIsAppliedAtAcquisition) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.size() < 2) {
    GTEST_SKIP() << "fewer than two stable process-visible processors";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {visible[0], visible[1]};
  CpuExecutorRegistry registry(1);

  EXPECT_NO_THROW(registry.Acquire(request));
}

TEST(CpuExecutor, ParallelForCoversRangeWithAllParticipants) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  RangeObservation observation(400);

  executor->ParallelFor(400, 100, &observation, &ObserveRange);

  EXPECT_EQ(observation.threads.size(), 4u);
  EXPECT_TRUE(std::all_of(observation.visits.begin(), observation.visits.end(),
                          [](int visits) { return visits == 1; }));
}

TEST(CpuExecutor, ExternalDispatcherCoversRangeWithoutCreatingAPool) {
  ExternalDispatchObservation dispatch;
  std::unique_ptr<CpuExecutor> executor =
      CpuExecutor::CreateExternal(4, &dispatch, &DispatchInline);
  RangeObservation observation(400);

  executor->ParallelFor(400, 100, &observation, &ObserveRange);

  EXPECT_EQ(dispatch.dispatches, 1);
  EXPECT_EQ(dispatch.dispatched_blocks, 4);
  EXPECT_TRUE(std::all_of(observation.visits.begin(), observation.visits.end(),
                          [](int visits) { return visits == 1; }));
}

TEST(CpuExecutor, ExternalDispatcherValidatesConfiguration) {
  ExternalDispatchObservation dispatch;
  EXPECT_THROW(CpuExecutor::CreateExternal(0, &dispatch, &DispatchInline), std::invalid_argument);
  EXPECT_THROW(CpuExecutor::CreateExternal(1, &dispatch, nullptr), std::invalid_argument);
}

TEST(CpuExecutor, MaximumParticipantsLowersSessionLimit) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  RangeObservation observation(400);

  executor->ParallelFor(400, 100, &observation, &ObserveRange, 2);

  EXPECT_EQ(observation.threads.size(), 2u);
  EXPECT_TRUE(std::all_of(observation.visits.begin(), observation.visits.end(),
                          [](int visits) { return visits == 1; }));
}

TEST(CpuExecutor, CostModelScalesParticipantsWithWork) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(8));
  const CpuLoopCost abs_float32{4.0, 4.0, 1.0};
  const CpuLoopCost log_float32{4.0, 4.0, 15.0};

  EXPECT_EQ(executor->PlanParallelFor(1024, abs_float32).participants, 1u);
  EXPECT_GT(executor->PlanParallelFor(1 << 20, abs_float32).participants, 1u);
  EXPECT_GT(executor->PlanParallelFor(1 << 18, log_float32).participants,
            executor->PlanParallelFor(1 << 18, abs_float32).participants);
  EXPECT_LE(executor->PlanParallelFor(1 << 20, log_float32, 3).participants, 3u);
  EXPECT_EQ(
      executor->PlanParallelFor(1 << 20, log_float32, CpuParallelConstraints{8, 3}).participants,
      3u);
  EXPECT_EQ(executor->PlanParallelFor(1024, log_float32, CpuParallelConstraints{8, 3}).participants,
            1u);
  EXPECT_EQ(
      executor->PlanParallelFor(1 << 20, log_float32, CpuParallelConstraints{2, 3}).participants,
      2u);
}

TEST(CpuExecutor, CostBasedParallelForCoversRange) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  RangeObservation observation(200000);

  executor->ParallelFor(200000, CpuLoopCost{4.0, 4.0, 15.0}, &observation, &ObserveRange);

  EXPECT_GT(observation.threads.size(), 1u);
  EXPECT_TRUE(std::all_of(observation.visits.begin(), observation.visits.end(),
                          [](int visits) { return visits == 1; }));
}

TEST(CpuExecutor, SerialPolicyDoesNotCreateParallelParticipants) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(1));
  RangeObservation observation(400);

  executor->ParallelFor(400, 100, &observation, &ObserveRange);

  EXPECT_EQ(executor->effective_threads(), 1u);
  EXPECT_EQ(observation.threads.size(), 1u);
}

struct NestedObservation {
  CpuExecutor *executor = nullptr;
  std::atomic<bool> changed_thread{false};
};

struct InnerObservation {
  std::thread::id expected_thread;
  std::atomic<bool> *changed_thread = nullptr;
};

void ObserveNestedRange(void *context, int64_t, int64_t) {
  auto &observation = *static_cast<NestedObservation *>(context);
  InnerObservation inner{std::this_thread::get_id(), &observation.changed_thread};
  observation.executor->ParallelFor(8, 1, &inner, [](void *inner_context, int64_t, int64_t) {
    auto &inner_observation = *static_cast<InnerObservation *>(inner_context);
    if (std::this_thread::get_id() != inner_observation.expected_thread) {
      inner_observation.changed_thread->store(true, std::memory_order_relaxed);
    }
  });
}

TEST(CpuExecutor, NestedParallelForRunsInline) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  NestedObservation observation{executor.get()};

  executor->ParallelFor(4, 1, &observation, &ObserveNestedRange);

  EXPECT_FALSE(observation.changed_thread.load(std::memory_order_relaxed));
}

TEST(CpuExecutor, DisabledCountersRemainEmpty) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(1));
  RangeObservation observation(8);

  executor->ParallelFor(8, 1, &observation, &ObserveRange);

  EXPECT_FALSE(executor->counters_enabled());
  EXPECT_EQ(executor->counters(), CpuExecutorCounters{});
}

TEST(CpuExecutor, EnabledCountersReportDispatchesAndNestedInlineCalls) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  executor->EnableCounters();
  NestedObservation observation{executor.get()};

  executor->ParallelFor(4, 1, &observation, &ObserveNestedRange);

  EXPECT_TRUE(executor->counters_enabled());
  const CpuExecutorCounters counters = executor->counters();
  EXPECT_EQ(counters.dispatches, 5u);
  EXPECT_EQ(counters.nested_inline_dispatches, 4u);
  EXPECT_EQ(counters.limited_inline_dispatches, 0u);
}

TEST(CpuExecutor, UnevenRangesRemainValidAndComplete) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  RangeObservation observation(5);

  executor->ParallelFor(5, 1, &observation, &ObserveRange);

  EXPECT_EQ(observation.threads.size(), 4u);
  EXPECT_TRUE(std::all_of(observation.visits.begin(), observation.visits.end(),
                          [](int visits) { return visits == 1; }));
}

TEST(CpuExecutor, ConcurrentRegionsRemainCorrect) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  RangeObservation first(1000);
  RangeObservation second(1000);

  std::thread first_call([&]() { executor->ParallelFor(1000, 100, &first, &ObserveRange); });
  std::thread second_call([&]() { executor->ParallelFor(1000, 100, &second, &ObserveRange); });
  first_call.join();
  second_call.join();

  EXPECT_TRUE(std::all_of(first.visits.begin(), first.visits.end(),
                          [](int visits) { return visits == 1; }));
  EXPECT_TRUE(std::all_of(second.visits.begin(), second.visits.end(),
                          [](int visits) { return visits == 1; }));
}

TEST(CpuExecutor, ValidatesParallelForArguments) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(1));
  EXPECT_THROW(executor->ParallelFor(1, 0, nullptr, &ObserveRange), std::invalid_argument);
  EXPECT_THROW(executor->ParallelFor(1, 1, nullptr, nullptr), std::invalid_argument);
}

TEST(CpuExecutor, RejectsMalformedResolvedPolicy) {
  CpuExecutorRegistry registry(1);
  ResolvedCpuExecutionPolicy malformed = NoAffinityPolicy(2);
  malformed.worker_processors.push_back(CpuLogicalProcessor{0});
  EXPECT_THROW(registry.Acquire(malformed), std::invalid_argument);
}

#if defined(__linux__)
TEST(CpuExecutor, ExecutorInheritedAcrossForkIsRejected) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(2));
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    bool rejected = false;
    try {
      executor->ParallelFor(2, 1, nullptr, [](void *, int64_t, int64_t) {});
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    _exit(rejected ? 0 : 1);
  }

  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
#endif

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

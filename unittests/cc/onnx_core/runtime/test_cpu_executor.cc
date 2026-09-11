// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_executor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
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
  std::unique_ptr<CpuExecutor> executor = CpuExecutor::CreateExternal(4, &DispatchInline);
  CpuExecutorDispatchScope dispatch_scope(executor.get(), &dispatch);
  RangeObservation observation(400);

  executor->ParallelFor(400, 100, &observation, &ObserveRange);

  EXPECT_EQ(dispatch.dispatches, 1);
  EXPECT_EQ(dispatch.dispatched_blocks, 4);
  EXPECT_TRUE(std::all_of(observation.visits.begin(), observation.visits.end(),
                          [](int visits) { return visits == 1; }));
}

TEST(CpuExecutor, MinimumElementsIsParallelCrossover) {
  std::unique_ptr<CpuExecutor> executor = CpuExecutor::CreateExternal(4, &DispatchInline);
  constexpr int64_t minimum_elements = 8;
  constexpr std::array<std::pair<int64_t, bool>, 4> cases{{
      {minimum_elements - 1, false},
      {minimum_elements, true},
      {2 * minimum_elements - 1, true},
      {2 * minimum_elements, true},
  }};

  for (const auto &[total, expect_parallel] : cases) {
    SCOPED_TRACE(total);
    ExternalDispatchObservation dispatch;
    CpuExecutorDispatchScope dispatch_scope(executor.get(), &dispatch);
    RangeObservation observation(static_cast<size_t>(total));
    executor->ParallelFor(total, minimum_elements, &observation, &ObserveRange);
    EXPECT_EQ(dispatch.dispatches != 0, expect_parallel);
    EXPECT_EQ(dispatch.dispatched_blocks, expect_parallel ? 4 : 0);
    EXPECT_TRUE(std::all_of(observation.visits.begin(), observation.visits.end(),
                            [](int visits) { return visits == 1; }));
  }

  ExternalDispatchObservation single_dispatch;
  CpuExecutorDispatchScope dispatch_scope(executor.get(), &single_dispatch);
  RangeObservation single_observation(1);
  executor->ParallelFor(1, 1, &single_observation, &ObserveRange);
  EXPECT_EQ(single_dispatch.dispatches, 0);
  EXPECT_EQ(single_observation.visits, std::vector<int>{1});
}

TEST(CpuExecutor, ExternalDispatcherValidatesConfiguration) {
  EXPECT_THROW(CpuExecutor::CreateExternal(0, &DispatchInline), std::invalid_argument);
  EXPECT_THROW(CpuExecutor::CreateExternal(1, nullptr), std::invalid_argument);
}

TEST(CpuExecutor, ExternalDispatcherRequiresInvocationScope) {
  std::unique_ptr<CpuExecutor> executor = CpuExecutor::CreateExternal(4, &DispatchInline);
  RangeObservation observation(400);

  EXPECT_THROW(executor->ParallelFor(400, 100, &observation, &ObserveRange), std::runtime_error);
}

TEST(CpuExecutor, ExternalDispatcherUsesPerInvocationContextConcurrently) {
  std::unique_ptr<CpuExecutor> executor = CpuExecutor::CreateExternal(4, &DispatchInline);
  ExternalDispatchObservation first_dispatch;
  ExternalDispatchObservation second_dispatch;
  RangeObservation first(400);
  RangeObservation second(400);
  std::thread first_call([&]() {
    CpuExecutorDispatchScope scope(executor.get(), &first_dispatch);
    executor->ParallelFor(400, 100, &first, &ObserveRange);
  });
  std::thread second_call([&]() {
    CpuExecutorDispatchScope scope(executor.get(), &second_dispatch);
    executor->ParallelFor(400, 100, &second, &ObserveRange);
  });
  first_call.join();
  second_call.join();

  EXPECT_EQ(first_dispatch.dispatches, 1);
  EXPECT_EQ(second_dispatch.dispatches, 1);
  EXPECT_TRUE(std::all_of(first.visits.begin(), first.visits.end(),
                          [](int visits) { return visits == 1; }));
  EXPECT_TRUE(std::all_of(second.visits.begin(), second.visits.end(),
                          [](int visits) { return visits == 1; }));
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

TEST(CpuExecutor, CostModelAmortizesRecurringDispatch) {
  CpuExecutorRegistry registry(1);
  std::shared_ptr<CpuExecutor> executor = registry.Acquire(NoAffinityPolicy(4));
  const CpuLoopCost exp_float32_avx2{4.0, 4.0, 1.5};

  EXPECT_EQ(executor->PlanParallelFor(32768, exp_float32_avx2).participants, 1u);
  EXPECT_GT(executor->PlanParallelFor(65536, exp_float32_avx2).participants, 1u);
  EXPECT_EQ(executor->PlanParallelFor(131072, exp_float32_avx2).participants, 4u);
}

TEST(CpuExecutor, CostModelDoesNotFillHighCoreExecutorForModestWork) {
  std::unique_ptr<CpuExecutor> four_threads = CpuExecutor::CreateExternal(4, &DispatchInline);
  std::unique_ptr<CpuExecutor> high_core = CpuExecutor::CreateExternal(64, &DispatchInline);
  const CpuLoopCost exp_float32_avx2{4.0, 4.0, 1.5};
  const CpuParallelPlan four_thread_plan = four_threads->PlanParallelFor(131072, exp_float32_avx2);
  const CpuParallelPlan high_core_plan = high_core->PlanParallelFor(131072, exp_float32_avx2);

  EXPECT_EQ(high_core_plan.participants, four_thread_plan.participants);
  EXPECT_EQ(high_core_plan.participants, 4u);
  EXPECT_LT(high_core_plan.participants, high_core->effective_threads());
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

TEST(CpuExecutorRegistry, NestedPolicyControlsSharingAndRetainsCompatiblePool) {
  CpuExecutorRegistry registry(2);
  std::array<uint64_t, 2> identities{};
  for (bool allow_nested : {false, true}) {
    CpuExecutionPolicy request;
    request.num_threads = 4;
    request.allow_nested_parallelism = allow_nested;
    auto executor = registry.Acquire(request);
    EXPECT_EQ(executor, registry.Acquire(request));
    EXPECT_EQ(executor->policy().allow_nested_parallelism, allow_nested);
    identities[allow_nested] = executor->instance_id();
    executor.reset();
    EXPECT_EQ(registry.Acquire(request)->instance_id(), identities[allow_nested]);
  }
  EXPECT_NE(identities[0], identities[1]);
  EXPECT_EQ(registry.live_pool_count(), 2u);
}

TEST(CpuExecutor, NestedAdmissionUsesOnlyIdleParticipants) {
  for (bool allow_nested : {false, true}) {
    for (CpuSpinPolicy spin : {CpuSpinPolicy::kAdaptive, CpuSpinPolicy::kParkImmediately}) {
      CpuExecutorRegistry registry(1);
      CpuExecutionPolicy request;
      request.num_threads = 4;
      request.spin_policy = spin;
      request.allow_nested_parallelism = allow_nested;
      auto executor = registry.Acquire(request);
      for (int64_t outer_participants : {1, 2, 4}) {
        for (int64_t nesting_block : {int64_t{0}, outer_participants - 1}) {
          for (uint32_t inner_limit : {0u, 2u}) {
            SCOPED_TRACE(::testing::Message()
                         << allow_nested << ", " << static_cast<int>(spin) << ", "
                         << outer_participants << ", " << nesting_block << ", " << inner_limit);
            struct Observation {
              CpuExecutor *executor;
              int64_t nesting_block;
              uint32_t inner_limit;
              std::barrier<> barrier;
              RangeObservation inner{17};
              RangeObservation outer{4};
            } observation{executor.get(), nesting_block, inner_limit,
                          std::barrier<>(outer_participants)};
            executor->ParallelFor(
                outer_participants, 1, &observation, [](void *context, int64_t begin, int64_t end) {
                  auto &call = *static_cast<Observation *>(context);
                  ObserveRange(&call.outer, begin, end);
                  call.barrier.arrive_and_wait();
                  if (begin == call.nesting_block) {
                    call.executor->ParallelFor(17, 1, &call.inner, &ObserveRange, call.inner_limit);
                  }
                  // Keeps every outer participant reserved until the nested region finishes.
                  call.barrier.arrive_and_wait();
                });
            const size_t expected =
                allow_nested
                    ? std::min<int64_t>(5 - outer_participants, inner_limit == 0 ? 4 : inner_limit)
                    : 1;
            EXPECT_EQ(observation.inner.threads.size(), expected);
            EXPECT_TRUE(std::all_of(observation.inner.visits.begin(),
                                    observation.inner.visits.end(),
                                    [](int visits) { return visits == 1; }));
            observation.outer.threads.insert(observation.inner.threads.begin(),
                                             observation.inner.threads.end());
            EXPECT_EQ(observation.outer.threads.size(), outer_participants + expected - 1);
            EXPECT_LE(observation.outer.threads.size(), 4u);
          }
        }
      }
    }
  }
}

TEST(CpuExecutor, NestedCostBasedDispatchAndProfilingHonorPolicy) {
  for (bool allow_nested : {false, true}) {
    for (int32_t participants : {1, 4}) {
      CpuExecutorRegistry registry(1);
      CpuExecutionPolicy request;
      request.num_threads = participants;
      request.allow_nested_parallelism = allow_nested;
      auto executor = registry.Acquire(request);
      executor->EnableCounters();
      ParallelRegionCollector collector(8);
      struct Observation {
        CpuExecutor *executor;
        ParallelRegionCollector *collector;
        RangeObservation inner{200000};
      } observation{executor.get(), &collector};
      executor->ParallelFor(
          1, 2, &observation,
          [](void *context, int64_t, int64_t) {
            auto &call = *static_cast<Observation *>(context);
            call.executor->ParallelFor(200000, CpuLoopCost{4.0, 4.0, 15.0}, &call.inner,
                                       &ObserveRange, 0, call.collector, "inner");
          },
          0, &collector, "outer");
      const size_t expected = allow_nested ? participants : 1;
      EXPECT_EQ(observation.inner.threads.size(), expected);
      EXPECT_TRUE(std::all_of(observation.inner.visits.begin(), observation.inner.visits.end(),
                              [](int visits) { return visits == 1; }));
      const auto events = collector.events();
      ASSERT_EQ(events.size(), 2u);
      EXPECT_EQ(events[0].label, "inner");
      EXPECT_EQ(events[0].admitted_threads, expected);
      EXPECT_EQ(events[0].observed_threads, expected);
      EXPECT_EQ(events[0].nested_inline, expected == 1);
      EXPECT_EQ(events[0].parent_region_id, events[1].region_id);
      EXPECT_EQ(events[0].executor_instance_id, executor->instance_id());
      EXPECT_EQ(executor->counters().nested_inline_dispatches, expected == 1 ? 1u : 0u);
    }
  }
}

TEST(CpuExecutor, ConcurrentRecursiveRegionsRespectParticipantBudget) {
  CpuExecutorRegistry registry(1);
  CpuExecutionPolicy request;
  request.num_threads = 4;
  request.allow_nested_parallelism = true;
  auto executor = registry.Acquire(request);
  struct Observation {
    CpuExecutor *executor;
    std::atomic<int> active{0};
    std::atomic<int> peak{0};
    std::atomic<int> visits{0};

    void Run(int depth) {
      struct Context {
        Observation *observation;
        int depth;
      } context{this, depth};
      executor->ParallelFor(
          3, 1, &context,
          [](void *opaque, int64_t begin, int64_t end) {
            auto &call = *static_cast<Context *>(opaque);
            auto &observation = *call.observation;
            for (int64_t i = begin; i < end; ++i) {
              if (call.depth != 0) {
                observation.Run(call.depth - 1);
              } else {
                const int active = observation.active.fetch_add(1) + 1;
                int peak = observation.peak.load();
                while (peak < active && !observation.peak.compare_exchange_weak(peak, active)) {
                }
                std::this_thread::yield();
                observation.visits.fetch_add(1);
                observation.active.fetch_sub(1);
              }
            }
          },
          depth == 3 ? 1 : 0);
    }
  } observation{executor.get()};
  std::thread first([&]() { observation.Run(3); });
  std::thread second([&]() { observation.Run(3); });
  first.join();
  second.join();
  EXPECT_EQ(observation.visits.load(), 162);
  EXPECT_LE(observation.peak.load(), 4);
  EXPECT_EQ(observation.active.load(), 0);
  RangeObservation reuse(17);
  executor->ParallelFor(17, 1, &reuse, &ObserveRange);
  EXPECT_EQ(reuse.threads.size(), 4u);
}

TEST(CpuExecutor, CrossPoolNestingDoesNotDeadlockOrCreateAnotherTeam) {
  CpuExecutionPolicy request;
  request.num_threads = 4;
  request.allow_nested_parallelism = true;
  CpuExecutorRegistry first_registry(1);
  CpuExecutorRegistry second_registry(1);
  auto first = first_registry.Acquire(request);
  auto second = second_registry.Acquire(request);
  struct Observation {
    CpuExecutor *first;
    CpuExecutor *second;
    RangeObservation inner{17};
  } observation{first.get(), second.get()};
  first->ParallelFor(1, 1, &observation, [](void *context, int64_t, int64_t) {
    auto &call = *static_cast<Observation *>(context);
    const std::thread::id caller = std::this_thread::get_id();
    call.second->ParallelFor(4, 1, context, [](void *context, int64_t begin, int64_t end) {
      auto &call = *static_cast<Observation *>(context);
      EXPECT_EQ(begin, 0);
      EXPECT_EQ(end, 4);
      call.first->ParallelFor(17, 1, &call.inner, &ObserveRange);
    });
    EXPECT_TRUE(call.inner.threads.contains(caller));
  });
  EXPECT_EQ(observation.inner.threads.size(), 4u);
  EXPECT_TRUE(std::all_of(observation.inner.visits.begin(), observation.inner.visits.end(),
                          [](int visits) { return visits == 1; }));
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
TEST(CpuExecutor, NestedDispatchPreservesWorkerAffinity) {
  const auto visible = ProcessVisibleLogicalProcessors();
  if (visible.size() < 2) {
    GTEST_SKIP() << "fewer than two stable process-visible processors";
  }
  for (bool allow_nested : {false, true}) {
    CpuExecutionPolicy request;
    request.affinity_policy = CpuAffinityPolicy::kExplicit;
    request.cpu_set = {visible[0], visible[1]};
    request.allow_nested_parallelism = allow_nested;
    CpuExecutorRegistry registry(1);
    auto executor = registry.Acquire(request);
    // Keeps explicit caller pinning off the test runner's thread.
    std::thread caller([&]() {
      executor->ParallelFor(2, 1, executor.get(), [](void *context, int64_t, int64_t) {
        auto &executor = *static_cast<CpuExecutor *>(context);
        auto before = ProcessVisibleLogicalProcessors();
        executor.ParallelFor(1, 1, &before, [](void *context, int64_t, int64_t) {
          const auto &expected = *static_cast<std::vector<CpuLogicalProcessor> *>(context);
          EXPECT_EQ(ProcessVisibleLogicalProcessors(), expected);
        });
        EXPECT_EQ(ProcessVisibleLogicalProcessors(), before);
      });
    });
    caller.join();
  }
}

TEST(CpuExecutor, CrossExecutorDispatchPreservesWorkerAffinity) {
  const auto visible = ProcessVisibleLogicalProcessors();
  if (visible.size() < 2) {
    GTEST_SKIP() << "fewer than two stable process-visible processors";
  }
  CpuExecutionPolicy outer_request;
  outer_request.num_threads = 2;
  outer_request.affinity_policy = CpuAffinityPolicy::kExplicit;
  outer_request.cpu_set = {visible[0], visible[1]};
  CpuExecutionPolicy inner_request;
  inner_request.num_threads = 1;
  inner_request.affinity_policy = CpuAffinityPolicy::kExplicit;
  inner_request.cpu_set = {visible[0]};
  CpuExecutorRegistry registry(2);
  auto outer = registry.Acquire(outer_request);
  auto inner = registry.Acquire(inner_request);

  std::thread caller([&]() {
    outer->ParallelFor(2, 1, inner.get(), [](void *context, int64_t, int64_t) {
      auto &inner = *static_cast<CpuExecutor *>(context);
      const auto before = ProcessVisibleLogicalProcessors();
      inner.ParallelFor(1, 1, nullptr, [](void *, int64_t, int64_t) {});
      EXPECT_EQ(ProcessVisibleLogicalProcessors(), before);
    });
  });
  caller.join();
}

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

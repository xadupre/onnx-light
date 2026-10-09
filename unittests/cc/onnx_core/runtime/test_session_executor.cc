// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/parallel_region_collector.h"
#include "onnx_core/runtime/tuning/runtime_parameters.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::CpuAffinityPolicy;
using core::runtime::CpuExecutionPolicy;
using core::runtime::CpuExecutor;
using core::runtime::CurrentCpuExecutor;
using core::runtime::CurrentParallelRegionCollector;
using core::runtime::ExecuteActionKind;
using core::runtime::ExecutionPlan;
using core::runtime::KernelContext;
using core::runtime::ParallelFor;
using core::runtime::ParallelForThreadCount;
using core::runtime::ParallelRegionCollector;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeEventAction;
using core::runtime::RuntimeParameters;
using core::runtime::RuntimeSession;
using core::runtime::RuntimeSessionOptions;

namespace {

/// What a kernel observed about the executor installed while it ran.
struct ExecutorObservation {
  CpuExecutor *current = nullptr;
  CpuExecutor *from_context = nullptr;
  int64_t reported_threads = 0;
  uint64_t event_executor_instance_id = 0;
  uint32_t event_effective_threads = 0;
  std::set<std::thread::id> block_threads;
};

bool session_execution_scope_active = false;
const NodeProto *session_kernel_node = nullptr;

class TestSessionExecutionScope {
public:
  explicit TestSessionExecutionScope(RuntimeContext &) { session_execution_scope_active = true; }

  ~TestSessionExecutionScope() { session_execution_scope_active = false; }
};

class TestSessionKernel : public core::runtime::KernelBase {
public:
  explicit TestSessionKernel(const KernelContext &ctx) : KernelBase(ctx) {}

  void Run(RuntimeContext &) override {
    EXPECT_TRUE(session_execution_scope_active);
    session_kernel_node = node_;
  }
};

/// Builds a one-node graph dispatched to a custom operator so a test kernel can
/// inspect the executor the session installed. A private domain keeps the
/// registration away from the built-in kernels.
GraphProto MakeObserverGraph() {
  GraphProto graph;
  NodeProto *node = graph.add_node();
  node->set_domain("test.execution_pools");
  node->set_op_type("ObserveExecutor");
  node->add_input("x");
  node->add_output("y");
  return graph;
}

/// Registers the observer kernel on ``rt`` and runs ``session`` once.
void RunObserver(RuntimeSession &session, RuntimeContext &rt, ExecutorObservation &observation,
                 int64_t total) {
  rt.RegisterCustomKernel("test.execution_pools", "ObserveExecutor",
                          [&observation, total](const NodeProto &node, RuntimeContext &ctx) {
                            observation.current = CurrentCpuExecutor();
                            observation.from_context = ctx.cpu_executor();
                            observation.reported_threads = ParallelForThreadCount();
                            std::mutex mutex;
                            ParallelFor(total, 1, [&](int64_t begin, int64_t end) {
                              EXPECT_LT(begin, end);
                              const std::lock_guard<std::mutex> guard(mutex);
                              observation.block_threads.insert(std::this_thread::get_id());
                            });
                            ctx.Set(std::string(node.output(0)), ctx.Get(node.input(0)));
                          });
  rt.Set("x", core::runtime::Tensor::FromFloat("x", {1}, {1.0f}));
  session.Run(rt);
}

} // namespace

TEST(ExecutionPlan, DerivesMissingReleasesFromPartialMetadata) {
  GraphProto graph;
  graph.add_input()->set_name("X");
  graph.add_input()->set_name("W");
  graph.add_output()->set_name("Y");
  NodeProto *add = graph.add_node();
  add->set_op_type("Add");
  add->add_input("X");
  add->add_input("W");
  add->add_output("A");
  add->add_metadata(core::compute::kNotUsedAfterMetadataKey, "X;W");
  NodeProto *relu = graph.add_node();
  relu->set_op_type("Relu");
  relu->add_input("A");
  relu->add_output("B");
  relu->add_metadata(core::compute::kReleaseAfterMetadataKey, "A");
  NodeProto *identity = graph.add_node();
  identity->set_op_type("Identity");
  identity->add_input("B");
  identity->add_output("Y");

  const ExecutionPlan plan(graph);
  const auto &actions = plan.actions();
  const auto releases = [&](const std::string &name, size_t node_index) {
    return std::count_if(actions.begin(), actions.end(), [&](const auto &action) {
      return action.kind() == ExecuteActionKind::kDeleteBuffer && action.name() == name &&
             action.node_index() == node_index;
    });
  };
  EXPECT_EQ(releases("A", 1), 1);
  EXPECT_EQ(releases("B", 2), 1);
  EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                          [](const auto &action) {
                            return action.kind() == ExecuteActionKind::kDeleteBuffer;
                          }),
            2);
}

TEST(ExecutionPlan, PreservesExplicitReleaseAndDerivesMissingRelease) {
  GraphProto graph;
  graph.add_input()->set_name("X");
  graph.add_output()->set_name("Y");
  NodeProto *first = graph.add_node();
  first->set_op_type("Identity");
  first->add_input("X");
  first->add_output("A");
  first->add_metadata(core::compute::kNotUsedAfterMetadataKey, "X");
  NodeProto *second = graph.add_node();
  second->set_op_type("Identity");
  second->add_input("A");
  second->add_output("B");
  NodeProto *third = graph.add_node();
  third->set_op_type("Identity");
  third->add_input("B");
  third->add_output("Y");
  third->add_metadata(core::compute::kReleaseAfterMetadataKey, "A");

  const ExecutionPlan plan(graph);
  const auto &actions = plan.actions();
  const auto explicit_release =
      std::find_if(actions.begin(), actions.end(), [](const auto &action) {
        return action.kind() == ExecuteActionKind::kDeleteBuffer && action.name() == "A";
      });
  ASSERT_NE(explicit_release, actions.end());
  EXPECT_EQ(explicit_release->node_index(), 2);
  EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                          [](const auto &action) {
                            return action.kind() == ExecuteActionKind::kDeleteBuffer &&
                                   action.name() == "A";
                          }),
            1);
  EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                          [](const auto &action) {
                            return action.kind() == ExecuteActionKind::kDeleteBuffer &&
                                   action.name() == "B" && action.node_index() == 2;
                          }),
            1);
}

TEST(ExecutionPlan, DerivesMissingUnlockFromPartialMetadata) {
  GraphProto graph;
  graph.add_input()->set_name("X");
  graph.add_input()->set_name("Z");
  graph.add_initializer()->set_name("W");
  graph.add_output()->set_name("Y");
  NodeProto *first = graph.add_node();
  first->set_op_type("If");
  first->add_input("X");
  first->add_output("A");
  first->add_metadata(core::compute::kNotUsedAfterMetadataKey, "X");
  AttributeProto *body_attribute = first->add_attribute();
  body_attribute->set_name("then_branch");
  body_attribute->set_type(AttributeProto::GRAPH);
  NodeProto *capturing_node = body_attribute->mutable_g()->add_node();
  capturing_node->set_op_type("Identity");
  capturing_node->add_input("W");
  capturing_node->add_output("inner");
  NodeProto *second = graph.add_node();
  second->set_op_type("Identity");
  second->add_input("Z");
  second->add_output("Y");
  second->add_metadata(core::compute::kNotUsedAfterMetadataKey, "Z");

  const ExecutionPlan plan(graph);
  const auto &actions = plan.actions();
  EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                          [](const auto &action) {
                            return action.kind() == ExecuteActionKind::kUnlockInput &&
                                   action.name() == "X";
                          }),
            1);
  const auto lock_w = std::find_if(actions.begin(), actions.end(), [](const auto &action) {
    return action.kind() == ExecuteActionKind::kLockInitializer && action.name() == "W";
  });
  const auto first_execute = std::find_if(actions.begin(), actions.end(), [](const auto &action) {
    return action.kind() == ExecuteActionKind::kExecuteNode && action.node_index() == 0;
  });
  const auto unlock_w = std::find_if(actions.begin(), actions.end(), [](const auto &action) {
    return action.kind() == ExecuteActionKind::kUnlockInitializer && action.name() == "W";
  });
  const auto second_execute = std::find_if(actions.begin(), actions.end(), [](const auto &action) {
    return action.kind() == ExecuteActionKind::kExecuteNode && action.node_index() == 1;
  });
  ASSERT_NE(lock_w, actions.end());
  ASSERT_NE(first_execute, actions.end());
  ASSERT_NE(unlock_w, actions.end());
  ASSERT_NE(second_execute, actions.end());
  EXPECT_LT(lock_w, first_execute);
  EXPECT_LT(first_execute, unlock_w);
  EXPECT_LT(unlock_w, second_execute);
}

TEST(ExecutionPlan, DefersStaleUnlockUntilLastUse) {
  GraphProto graph;
  graph.add_input()->set_name("X");
  graph.add_output()->set_name("Y");
  NodeProto *first = graph.add_node();
  first->set_op_type("Identity");
  first->add_input("X");
  first->add_output("A");
  first->add_metadata(core::compute::kNotUsedAfterMetadataKey, "X");
  NodeProto *second = graph.add_node();
  second->set_op_type("Identity");
  second->add_input("X");
  second->add_output("Y");

  const ExecutionPlan plan(graph);
  const auto &actions = plan.actions();
  EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                          [](const auto &action) {
                            return action.kind() == ExecuteActionKind::kLockInput &&
                                   action.name() == "X";
                          }),
            1);
  EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                          [](const auto &action) {
                            return action.kind() == ExecuteActionKind::kUnlockInput &&
                                   action.name() == "X";
                          }),
            1);
  const auto second_execute = std::find_if(actions.begin(), actions.end(), [](const auto &action) {
    return action.kind() == ExecuteActionKind::kExecuteNode && action.node_index() == 1;
  });
  const auto unlock_x = std::find_if(actions.begin(), actions.end(), [](const auto &action) {
    return action.kind() == ExecuteActionKind::kUnlockInput && action.name() == "X";
  });
  ASSERT_NE(second_execute, actions.end());
  ASSERT_NE(unlock_x, actions.end());
  EXPECT_LT(second_execute, unlock_x);
}

TEST(SessionExecutor, MakeSessionKernelInstallsBackendExecutionScope) {
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
  NodeProto node;
  node.set_op_type("TestSessionKernel");
  session_execution_scope_active = false;
  session_kernel_node = nullptr;

  std::unique_ptr<core::runtime::KernelBase> kernel =
      core::runtime::MakeSessionKernel<TestSessionKernel, TestSessionExecutionScope>(
          node, rt.kernel_ctx());

  EXPECT_FALSE(session_execution_scope_active);
  kernel->Run(rt);
  EXPECT_FALSE(session_execution_scope_active);
  EXPECT_EQ(session_kernel_node, &node);
}

TEST(SessionExecutor, DefaultPolicyDerivesFromParameters) {
  ExecutionPlan plan;
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(3)});
  EXPECT_EQ(session.cpu_execution_policy().num_threads, 3);
  EXPECT_EQ(session.cpu_execution_policy().affinity_policy, CpuAffinityPolicy::kNone);
  EXPECT_EQ(session.cpu_executor()->effective_threads(), 3u);
}

TEST(SessionExecutor, NegativeThreadCountRequestsTheTopologyDefault) {
  ExecutionPlan plan;
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(-4)});
  EXPECT_EQ(session.cpu_execution_policy().num_threads, 0);
  EXPECT_GE(session.cpu_executor()->effective_threads(), 1u);
}

TEST(SessionExecutor, SerialPolicyCreatesNoWorkers) {
  ExecutionPlan plan;
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(1)});
  EXPECT_EQ(session.cpu_executor()->effective_threads(), 1u);
}

TEST(SessionExecutor, ExplicitPolicyOverridesParameters) {
  CpuExecutionPolicy request;
  request.num_threads = 2;
  request.affinity_policy = CpuAffinityPolicy::kNone;
  ExecutionPlan plan;
  RuntimeSession session(plan, RuntimeSessionOptions{
                                   .parameters = RuntimeParameters(5),
                                   .cpu_execution = request,
                               });
  EXPECT_EQ(session.cpu_execution_policy().num_threads, 2);
  EXPECT_EQ(session.cpu_executor()->effective_threads(), 2u);
}

TEST(SessionExecutor, CompatibleSessionsShareOneExecutor) {
  CpuExecutionPolicy request;
  request.num_threads = 2;
  request.affinity_policy = CpuAffinityPolicy::kNone;
  ExecutionPlan plan;
  RuntimeSession first(plan, RuntimeSessionOptions{.cpu_execution = request});
  RuntimeSession second(plan, RuntimeSessionOptions{.cpu_execution = request});
  EXPECT_EQ(first.cpu_executor().get(), second.cpu_executor().get());
}

TEST(SessionExecutor, IncompatibleSessionsGetDistinctExecutors) {
  ExecutionPlan plan;
  RuntimeSession first(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});
  RuntimeSession second(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(3)});
  EXPECT_NE(first.cpu_executor().get(), second.cpu_executor().get());
}

TEST(SessionExecutor, LeaseIsStableAcrossCalls) {
  ExecutionPlan plan;
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});
  EXPECT_EQ(session.cpu_executor().get(), session.cpu_executor().get());
}

TEST(SessionExecutor, RunInstallsTheLeasedExecutor) {
  const GraphProto graph = MakeObserverGraph();
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});
  ExecutorObservation observation;
  RunObserver(session, rt, observation, 1);

  EXPECT_EQ(observation.current, session.cpu_executor().get());
  EXPECT_EQ(observation.from_context, session.cpu_executor().get());
  EXPECT_EQ(observation.reported_threads, 2);
  // The view is detached once the run is over so a context reused outside a
  // session never points at a released lease.
  EXPECT_EQ(rt.cpu_executor(), nullptr);
  EXPECT_EQ(CurrentCpuExecutor(), nullptr);
}

TEST(SessionExecutor, RunEventsIdentifyTheResolvedExecutor) {
  const GraphProto graph = MakeObserverGraph();
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});
  ExecutorObservation observation;

  RunObserver(session, rt, observation, 8);

  const auto event =
      std::find_if(rt.events().begin(), rt.events().end(), [](const auto &candidate) {
        return candidate.action == RuntimeEventAction::kRunNode;
      });
  ASSERT_NE(event, rt.events().end());
  EXPECT_EQ(event->cpu_executor_instance_id, session.cpu_executor()->instance_id());
  EXPECT_EQ(event->cpu_effective_threads, session.cpu_executor()->effective_threads());
  EXPECT_NE(event->cpu_executor_instance_id, 0u);
}

TEST(SessionExecutor, ObservedParticipantsMatchTheRequestedThreadCount) {
  const GraphProto graph = MakeObserverGraph();
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});
  ExecutorObservation observation;
  RunObserver(session, rt, observation, 8);
  EXPECT_EQ(observation.block_threads.size(), 2u);
}

TEST(SessionExecutor, RunInstallsAndRetainsParallelRegionCollector) {
  const GraphProto graph = MakeObserverGraph();
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  auto collector = std::make_shared<ParallelRegionCollector>(1);
  const std::weak_ptr<ParallelRegionCollector> retained = collector;
  RuntimeSession session(plan, RuntimeSessionOptions{
                                   .parameters = RuntimeParameters(2),
                                   .parallel_region_collector = collector,
                               });
  collector.reset();
  ExecutorObservation observation;

  RunObserver(session, rt, observation, 8);

  ASSERT_FALSE(retained.expired());
  ASSERT_EQ(session.parallel_region_collector()->events().size(), 1u);
  const auto &event = session.parallel_region_collector()->events()[0];
  EXPECT_EQ(event.requested_threads, 2);
  EXPECT_EQ(event.admitted_threads, 2);
  EXPECT_EQ(event.observed_threads, 2);
  EXPECT_EQ(event.executor_instance_id, session.cpu_executor()->instance_id());
  EXPECT_NE(std::string_view(event.location.file_name()).find("test_session_executor.cc"),
            std::string_view::npos);
  EXPECT_EQ(CurrentParallelRegionCollector(), nullptr);
}

TEST(SessionExecutor, ReportsAnEmptySnapshotWhenProfilingIsDisabled) {
  const GraphProto graph = MakeObserverGraph();
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(1)});

  const auto report = session.parallel_region_report();

  EXPECT_TRUE(report.events().empty());
  EXPECT_EQ(report.dropped_events(), 0u);
}

TEST(SessionExecutor, ConcurrentRunsUseIndependentCollectorsAndRunIds) {
  auto first_collector = std::make_shared<ParallelRegionCollector>(1);
  auto second_collector = std::make_shared<ParallelRegionCollector>(1);
  const auto run = [](const std::shared_ptr<ParallelRegionCollector> &collector) {
    const GraphProto graph = MakeObserverGraph();
    RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
    const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
    RuntimeSession session(plan, RuntimeSessionOptions{
                                     .parameters = RuntimeParameters(1),
                                     .parallel_region_collector = collector,
                                 });
    ExecutorObservation observation;
    RunObserver(session, rt, observation, 8);
  };

  std::thread first(run, first_collector);
  std::thread second(run, second_collector);
  first.join();
  second.join();

  ASSERT_EQ(first_collector->events().size(), 1u);
  ASSERT_EQ(second_collector->events().size(), 1u);
  EXPECT_NE(first_collector->events()[0].run_id, 0u);
  EXPECT_NE(second_collector->events()[0].run_id, 0u);
  EXPECT_NE(first_collector->events()[0].run_id, second_collector->events()[0].run_id);
  EXPECT_NE(first_collector->events()[0].calling_thread_id,
            second_collector->events()[0].calling_thread_id);
}

TEST(SessionExecutor, SerialSessionRunsEveryRangeInline) {
  const GraphProto graph = MakeObserverGraph();
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(1)});
  ExecutorObservation observation;
  RunObserver(session, rt, observation, 8);
  EXPECT_EQ(observation.reported_threads, 1);
  EXPECT_EQ(observation.block_threads.size(), 1u);
  EXPECT_EQ(*observation.block_threads.begin(), std::this_thread::get_id());
}

TEST(SessionExecutor, NestedSessionKeepsTheEnclosingExecutor) {
  // A nested session (a subgraph body, a model-local function, ...) is built
  // without an explicit policy; it must keep running on the executor the
  // enclosing session installed instead of leasing a second pool.
  GraphProto inner_graph;
  NodeProto *inner_node = inner_graph.add_node();
  inner_node->set_domain("test.execution_pools");
  inner_node->set_op_type("ObserveNested");
  inner_node->add_input("x");
  inner_node->add_output("z");

  const GraphProto outer_graph = MakeObserverGraph();
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  const ExecutionPlan &outer_plan = rt.GetExecutionPlan(outer_graph);
  RuntimeSession outer(outer_plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});

  ExecutorObservation nested;
  rt.RegisterCustomKernel(
      "test.execution_pools", "ObserveExecutor",
      [&inner_graph, &nested](const NodeProto &node, RuntimeContext &ctx) {
        RuntimeContext child = ctx.MakeSubgraphContext("body");
        child.RegisterCustomKernel("test.execution_pools", "ObserveNested",
                                   [&nested](const NodeProto &inner, RuntimeContext &inner_ctx) {
                                     nested.current = CurrentCpuExecutor();
                                     nested.from_context = inner_ctx.cpu_executor();
                                     nested.reported_threads = ParallelForThreadCount();
                                     inner_ctx.Set(std::string(inner.output(0)),
                                                   inner_ctx.Get(inner.input(0)));
                                   });
        const ExecutionPlan &inner_plan = child.GetExecutionPlan(inner_graph);
        RuntimeSession inner_session(inner_plan);
        inner_session.Run(child);
        const auto event =
            std::find_if(child.events().begin(), child.events().end(), [](const auto &candidate) {
              return candidate.action == RuntimeEventAction::kRunNode &&
                     candidate.op_type == "ObserveNested";
            });
        ASSERT_NE(event, child.events().end());
        nested.event_executor_instance_id = event->cpu_executor_instance_id;
        nested.event_effective_threads = event->cpu_effective_threads;
        ctx.Set(std::string(node.output(0)), ctx.Get(node.input(0)));
      });
  rt.Set("x", core::runtime::Tensor::FromFloat("x", {1}, {1.0f}));
  outer.Run(rt);

  EXPECT_EQ(nested.current, outer.cpu_executor().get());
  EXPECT_EQ(nested.from_context, outer.cpu_executor().get());
  EXPECT_EQ(nested.reported_threads, 2);
  EXPECT_EQ(nested.event_executor_instance_id, outer.cpu_executor()->instance_id());
  EXPECT_EQ(nested.event_effective_threads, 2u);
}

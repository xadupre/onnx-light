// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/runtime_parameters.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::CpuAffinityPolicy;
using core::runtime::CpuExecutionPolicy;
using core::runtime::CpuExecutor;
using core::runtime::CurrentCpuExecutor;
using core::runtime::ExecutionPlan;
using core::runtime::KernelContext;
using core::runtime::ParallelFor;
using core::runtime::ParallelForThreadCount;
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

TEST(SessionExecutor, MakeSessionKernelInstallsBackendExecutionScope) {
  RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(18)));
  NodeProto node;
  node.set_op_type("TestSessionKernel");
  session_execution_scope_active = false;
  session_kernel_node = nullptr;

  std::unique_ptr<core::runtime::KernelBase> kernel =
      core::runtime::MakeSessionKernel<TestSessionKernel, TestSessionExecutionScope>(node, rt);

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
  ExecutionPlan plan;
  RuntimeSession first(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});
  RuntimeSession second(plan, RuntimeSessionOptions{.parameters = RuntimeParameters(2)});
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

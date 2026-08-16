// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/tuning/runtime_parameters.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::ExecutionPlan;
using core::runtime::KernelContext;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeContextOptions;
using core::runtime::RuntimeParameters;
using core::runtime::RuntimeSession;
using core::runtime::RuntimeSessionOptions;
using core::runtime::Tensor;

namespace {

// Resolves the number of CPU cores the same way RuntimeParameters does so the
// tests do not hard-code an expected value that depends on the test host.
int32_t ExpectedCores() {
  unsigned int cores = std::thread::hardware_concurrency();
  return cores == 0 ? 1 : static_cast<int32_t>(cores);
}

} // namespace

TEST(runtime_parameters, DefaultIsCores) {
  RuntimeParameters params;
  EXPECT_EQ(params.num_threads, 0);
  EXPECT_EQ(params.EffectiveNumThreads(), ExpectedCores());
}

TEST(runtime_parameters, ZeroResolvesToCores) {
  RuntimeParameters params(0);
  EXPECT_EQ(params.EffectiveNumThreads(), ExpectedCores());
}

TEST(runtime_parameters, NegativeResolvesToCores) {
  RuntimeParameters params(-4);
  EXPECT_EQ(params.EffectiveNumThreads(), ExpectedCores());
}

TEST(runtime_parameters, OneDisablesParallelism) {
  RuntimeParameters params(1);
  EXPECT_EQ(params.EffectiveNumThreads(), 1);
  EXPECT_FALSE(params.is_parallel());
}

TEST(runtime_parameters, GreaterThanOneUsesExactCount) {
  RuntimeParameters params(7);
  EXPECT_EQ(params.EffectiveNumThreads(), 7);
  EXPECT_TRUE(params.is_parallel());
}

TEST(runtime_parameters, RuntimeSessionDefaultsToNoParameters) {
  ExecutionPlan plan;
  RuntimeSession session(plan);
  EXPECT_EQ(session.parameters().num_threads, 0);
}

TEST(runtime_parameters, RuntimeSessionSetParameters) {
  ExecutionPlan plan;
  RuntimeSession session(plan, RuntimeSessionOptions{
                                   .parameters = RuntimeParameters(3),
                                   .verbose = 0,
                                   .check_shapes = false,
                               });
  EXPECT_EQ(session.parameters().num_threads, 3);
  EXPECT_EQ(session.parameters().EffectiveNumThreads(), 3);
}

TEST(runtime_parameters, RuntimeSessionFromModelBuildsPlan) {
  // Constructing a session from a ModelProto (no external plan supplied) builds
  // and owns the plan from the model's graph, so the session can execute the
  // graph directly. Here a single-node Add graph is run against a context that
  // supplies its two external inputs.
  ModelProto model;
  GraphProto &graph = model.ref_graph();
  NodeProto *node = graph.add_node();
  node->set_op_type("Add");
  node->add_input("x");
  node->add_input("y");
  node->add_output("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});

  RuntimeSession session(model);
  session.Run(rt);

  EXPECT_EQ(session.required_inputs(), std::vector<std::string>({"x", "y"}));
  ASSERT_NE(rt.tensors().find("z"), rt.tensors().end());
  const Tensor &z = rt.tensors()["z"];
  ASSERT_EQ(z.element_count(), 3);
  const float *got = z.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 11.0f);
  EXPECT_FLOAT_EQ(got[1], 22.0f);
  EXPECT_FLOAT_EQ(got[2], 33.0f);
}

TEST(runtime_parameters, RuntimeSessionDefaultsToNoVerbose) {
  ExecutionPlan plan;
  RuntimeSession session(plan);
  EXPECT_EQ(session.verbose(), 0);
}

TEST(runtime_parameters, RuntimeSessionSetVerbose) {
  ExecutionPlan plan;
  RuntimeSession session(plan, 2);
  EXPECT_EQ(session.verbose(), 2);
}

TEST(runtime_parameters, RuntimeSessionConstructorVerbose) {
  ExecutionPlan plan;
  RuntimeSession session(plan, 3);
  EXPECT_EQ(session.verbose(), 3);
}

TEST(runtime_parameters, RuntimeSessionRunEnablesVerboseOnContext) {
  // A non-zero session verbosity given at construction is used while the graph
  // runs, but it does not mutate the RuntimeContext's own verbosity.
  ModelProto model;
  GraphProto &graph = model.ref_graph();
  NodeProto *node = graph.add_node();
  node->set_op_type("Add");
  node->add_input("x");
  node->add_input("y");
  node->add_output("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});
  EXPECT_EQ(rt.verbose(), 0);

  RuntimeSession session(model, 5);
  session.Run(rt);
  EXPECT_EQ(rt.verbose(), 0);
}

TEST(runtime_parameters, RuntimeSessionRunLeavesContextVerboseWhenZero) {
  // A zero verbosity (the default) leaves the context's own verbosity untouched
  // so callers can drive verbosity through the RuntimeContext directly.
  ModelProto model;
  GraphProto &graph = model.ref_graph();
  NodeProto *node = graph.add_node();
  node->set_op_type("Add");
  node->add_input("x");
  node->add_input("y");
  node->add_output("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)), RuntimeContextOptions{
                                                         .allocator = nullptr,
                                                         .events_enabled = false,
                                                         .verbose = 4,
                                                         .release_intermediates = false,
                                                     });
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});

  RuntimeSession session(model);
  session.Run(rt);
  EXPECT_EQ(rt.verbose(), 4);
}

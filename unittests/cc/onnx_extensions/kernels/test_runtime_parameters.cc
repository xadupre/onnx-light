// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/runtime/runtime_parameters.h"
#include "onnx_core/runtime/runtime_session.h"

#include <gtest/gtest.h>

#include <thread>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::ExecutionPlan;
using core::runtime::RuntimeParameters;
using core::runtime::RuntimeSession;

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
  RuntimeSession session(plan);
  session.set_parameters(RuntimeParameters(3));
  EXPECT_EQ(session.parameters().num_threads, 3);
  EXPECT_EQ(session.parameters().EffectiveNumThreads(), 3);
}

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/backend_test/cases_runtime/local_function/include_local_function_cases.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DataSet;
using core::backend_test::DefaultOpset;
using core::backend_test::TestCase;
using core::runtime::ExecutionPlan;
using core::runtime::KernelContext;
using core::runtime::RegisterModelFunctions;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeSession;
using core::runtime::Tensor;
using core::runtime::TensorFromProto;
using onnx_backend_test::CollectLocalFunctionTestCases;

namespace {

constexpr int64_t kDefaultOpsetVersion = 18;

std::vector<TestCase> Collect(const std::string &op_type = "") {
  std::vector<TestCase> registry;
  CollectLocalFunctionTestCases(registry, op_type);
  return registry;
}

const TestCase *Find(const std::vector<TestCase> &cases, const std::string &name) {
  for (const auto &c : cases) {
    if (c.name == name) {
      return &c;
    }
  }
  return nullptr;
}

// Registers `model`'s local functions in `rt`, seeds `model.graph()`'s
// initializers into `rt`, and runs the graph by building its ExecutionPlan
// and driving it through a fresh RuntimeSession. This is what the removed
// `RunModel` used to do internally.
void RunModelViaSession(const ModelProto &model, RuntimeContext &rt) {
  RegisterModelFunctions(model, rt);
  const GraphProto &graph = model.ref_graph();
  const auto &inits = graph.initializer();
  for (size_t i = 0; i < inits.size(); ++i) {
    const TensorProto &tp = inits[i];
    const std::string init_name = tp.name();
    if (!rt.Has(init_name)) {
      rt.Set(init_name, TensorFromProto(tp), core::runtime::RuntimeEventKind::kInitializer);
    }
  }
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);
  session.Run(rt);
}

// Runs ``tc`` through ``RunModelViaSession`` and verifies that every data
// set's expected outputs are reproduced bit-for-bit by the runtime.
void ExpectModelMatchesDataSets(const TestCase &tc) {
  for (const DataSet &ds : tc.data_sets()) {
    RuntimeContext rt(KernelContext(DefaultOpset(kDefaultOpsetVersion)));
    for (const Tensor &in : ds.inputs) {
      rt.Set(in.name, in);
    }
    ASSERT_NO_THROW(RunModelViaSession(tc.model(), rt)) << "case: " << tc.name;
    for (const Tensor &expected : ds.outputs) {
      ASSERT_TRUE(rt.Has(expected.name))
          << "case: " << tc.name << " missing output '" << expected.name << "'";
      const Tensor &actual = rt.Get(expected.name);
      EXPECT_EQ(actual.data_type, expected.data_type);
      EXPECT_EQ(actual.shape, expected.shape);
      EXPECT_EQ(actual.data, expected.data);
    }
  }
}

} // namespace

namespace Test {

TEST(LocalFunctionRuntimeCases, CrossDomainModelHasFunctions) {
  const auto cases = Collect("local_function");
  const TestCase *tc = Find(cases, "test_cc_local_function_calls_function_across_domains");
  ASSERT_NE(tc, nullptr);
  // The model must carry both FunctionProto entries so running it can
  // resolve the cross-domain dispatch.
  ASSERT_EQ(tc->model().ref_functions().size(), 2u);
}

TEST(LocalFunctionRuntimeCases, CrossDomainRunModelMatchesExpected) {
  const auto cases = Collect("local_function");
  const TestCase *tc = Find(cases, "test_cc_local_function_calls_function_across_domains");
  ASSERT_NE(tc, nullptr);
  ExpectModelMatchesDataSets(*tc);
}

} // namespace Test

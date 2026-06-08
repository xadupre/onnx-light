// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_runtime/local_function/include_local_function_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectLocalFunctionTestCases;
using onnx_backend_test::DataSet;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::TestCase;
using onnx_kernels::RunModel;
using onnx_kernels::RuntimeContext;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;

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

// Runs ``tc`` through ``RunModel`` and verifies that every data set's
// expected outputs are reproduced bit-for-bit by the runtime.
void ExpectModelMatchesDataSets(const TestCase &tc) {
  for (const DataSet &ds : tc.data_sets) {
    RuntimeContext rt(KernelContext(DefaultOpset(kDefaultOpsetVersion)));
    for (const Tensor &in : ds.inputs) {
      rt.Set(in.name, in);
    }
    ASSERT_NO_THROW(RunModel(tc.model, rt)) << "case: " << tc.name;
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
  // The model must carry both FunctionProto entries so RunModel can
  // resolve the cross-domain dispatch.
  ASSERT_EQ(tc->model.ref_functions().size(), 2u);
}

TEST(LocalFunctionRuntimeCases, CrossDomainRunModelMatchesExpected) {
  const auto cases = Collect("local_function");
  const TestCase *tc = Find(cases, "test_cc_local_function_calls_function_across_domains");
  ASSERT_NE(tc, nullptr);
  ExpectModelMatchesDataSets(*tc);
}

} // namespace Test

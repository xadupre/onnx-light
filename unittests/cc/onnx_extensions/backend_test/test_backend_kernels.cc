// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/backend_test/test_case_registry.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"
#include "test_case_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::DefaultOpset;
using core::backend_test::GetRegisteredCollectors;
using core::backend_test::TestCase;
using core::backend_test::TestCaseUnloadGuard;
using core::backend_test::TestMode;
using core::runtime::KernelBase;
using core::runtime::Tensor;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Split;

namespace Test {

namespace {

constexpr int64_t kFallbackDefaultOpsetVersion = 18;

std::string UtilsStringToStdString(const std::string &s) { return s; }

int64_t GetDefaultOpsetVersion(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    if (opset.ref_domain().empty()) {
      return opset.version();
    }
  }
  return kFallbackDefaultOpsetVersion;
}

std::vector<int64_t> ToInt64Vector(const Tensor &t) {
  EXPECT_EQ(t.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  std::vector<int64_t> values;
  values.reserve(static_cast<size_t>(t.element_count()));
  const int64_t *p = t.AsInt64();
  for (int64_t i = 0; i < t.element_count(); ++i) {
    values.push_back(p[i]);
  }
  return values;
}

void ExpectTensorEqual(const Tensor &actual, const Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  EXPECT_EQ(actual.string_data, expected.string_data);
  EXPECT_EQ(actual.data, expected.data);
}

} // namespace

TEST(BackendKernels, CollectionDoesNotRetainReferenceKernels) {
  const int64_t initial_live = KernelBase::LiveInstanceCountForTesting();
  uint64_t construction_count = KernelBase::ConstructionCountForTesting();

  const auto &collectors = GetRegisteredCollectors();
  for (size_t collector_index = 0; collector_index < collectors.size(); ++collector_index) {
    SCOPED_TRACE("collector index: " + std::to_string(collector_index));
    const auto &collector = collectors[collector_index];
    std::vector<TestCase> test_cases;
    collector(test_cases, "", false, TestMode::TEST);
    EXPECT_EQ(KernelBase::ConstructionCountForTesting(), construction_count);
    EXPECT_EQ(KernelBase::LiveInstanceCountForTesting(), initial_live);
    construction_count = KernelBase::ConstructionCountForTesting();

    std::vector<TestCase> benchmark_cases;
    collector(benchmark_cases, "", false, TestMode::BENCHMARK);
    EXPECT_EQ(KernelBase::ConstructionCountForTesting(), construction_count);
    EXPECT_EQ(KernelBase::LiveInstanceCountForTesting(), initial_live);
    construction_count = KernelBase::ConstructionCountForTesting();
  }

  const uint64_t before_lazy_collection = KernelBase::ConstructionCountForTesting();
  std::vector<TestCase> abs_cases = CollectTestCases("Abs", false, TestMode::TEST);
  EXPECT_EQ(KernelBase::ConstructionCountForTesting(), before_lazy_collection);
  EXPECT_EQ(KernelBase::LiveInstanceCountForTesting(), initial_live);
  auto abs_case = std::find_if(abs_cases.begin(), abs_cases.end(), [](const TestCase &test_case) {
    return test_case.name == "test_cc_abs";
  });
  ASSERT_NE(abs_case, abs_cases.end());

  const uint64_t before_materialization = KernelBase::ConstructionCountForTesting();
  abs_case->Materialize();
  EXPECT_GT(KernelBase::ConstructionCountForTesting(), before_materialization);
  EXPECT_EQ(KernelBase::LiveInstanceCountForTesting(), initial_live);

  const uint64_t before_benchmark_collection = KernelBase::ConstructionCountForTesting();
  std::vector<TestCase> abs_benchmarks = CollectTestCases("Abs", false, TestMode::BENCHMARK, false);
  EXPECT_EQ(KernelBase::ConstructionCountForTesting(), before_benchmark_collection);
  EXPECT_EQ(KernelBase::LiveInstanceCountForTesting(), initial_live);
}

TEST(BackendKernels, SplitKernelRunsOnBackendTestCases) {
  std::vector<TestCase> cases = CollectTestCases("Split");
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    TestCaseUnloadGuard unload_guard(tc);
    SCOPED_TRACE(tc.name);
    ASSERT_EQ(tc.model().ref_graph().ref_node().size(), 1u);
    const NodeProto &node = tc.model().ref_graph().ref_node()[0];
    ASSERT_EQ(UtilsStringToStdString(node.ref_op_type()), "Split");

    const int64_t axis = GetAttributeOr<int64_t>(node, "axis", 0);
    int64_t num_outputs = GetAttributeOr<int64_t>(node, "num_outputs", 0);

    const KernelContext ctx{DefaultOpset(GetDefaultOpsetVersion(tc.model()))};
    const Split split_kernel{ctx};

    for (const DataSet &ds : tc.data_sets()) {
      ASSERT_FALSE(ds.inputs.empty());
      const Tensor &input = ds.inputs[0];
      const std::vector<int64_t> split =
          ds.inputs.size() > 1 ? ToInt64Vector(ds.inputs[1]) : std::vector<int64_t>();
      if (split.empty() && num_outputs <= 0) {
        num_outputs = static_cast<int64_t>(node.ref_output().size());
      }

      const std::vector<Tensor> outputs = split_kernel(input, axis, split, num_outputs);
      ASSERT_EQ(outputs.size(), ds.outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        ExpectTensorEqual(outputs[i], ds.outputs[i]);
      }
    }
  }
}

} // namespace Test

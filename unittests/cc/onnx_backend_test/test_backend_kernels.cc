// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::CollectTestCases;
using onnx_kernels::DataSet;
using onnx_kernels::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::TestCase;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Split;

namespace Test {

namespace {

constexpr int64_t kFallbackDefaultOpsetVersion = 18;

std::string UtilsStringToStdString(const utils::String &s) {
  return std::string(s.data(), s.size());
}

int64_t GetDefaultOpsetVersion(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    if (opset.ref_domain().empty()) {
      return opset.version();
    }
  }
  return kFallbackDefaultOpsetVersion;
}

std::vector<int64_t> ToInt64Vector(const Tensor &t) {
  EXPECT_EQ(t.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
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

TEST(BackendKernels, SplitKernelRunsOnBackendTestCases) {
  const std::vector<TestCase> cases = CollectTestCases("Split");
  ASSERT_FALSE(cases.empty());

  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_EQ(tc.model.ref_graph().ref_node().size(), 1u);
    const NodeProto &node = tc.model.ref_graph().ref_node()[0];
    ASSERT_EQ(UtilsStringToStdString(node.ref_op_type()), "Split");

    const int64_t axis = GetAttributeOr<int64_t>(node, "axis", 0);
    int64_t num_outputs = GetAttributeOr<int64_t>(node, "num_outputs", 0);

    const KernelContext ctx{DefaultOpset(GetDefaultOpsetVersion(tc.model))};
    const Split split_kernel{ctx};

    for (const DataSet &ds : tc.data_sets) {
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

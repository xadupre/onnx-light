// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterWhereNanInfCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::KernelContext ctx{opset};
  const kernel::Where where_kernel{ctx};

  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kPosInf = std::numeric_limits<float>::infinity();
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  // Element-wise case: condition selects between two ``float`` tensors that
  // each contain NaN, +Inf and -Inf. ``Where`` must forward the selected
  // value unchanged, including the non-finite specials.
  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    Tensor condition = Tensor::FromBool("condition", {6}, {1, 0, 1, 0, 1, 0});
    Tensor x = Tensor::FromFloat("x", {6}, {kNan, 1.0f, kPosInf, 2.0f, kNegInf, 3.0f});
    Tensor y = Tensor::FromFloat("y", {6}, {0.0f, kNan, 0.0f, kPosInf, 0.0f, kNegInf});
    Tensor output = where_kernel(condition, x, y);

    Expect(node, {condition, x, y}, {output}, "test_cc_where_nan_inf", {opset}, "backend-test",
           registry, "nan_inf");
  }

  // Broadcast case: the condition broadcasts a single column across all
  // rows; ``x`` carries +Inf values and ``y`` carries -Inf / NaN, so the
  // output mixes both branches with their non-finite values preserved.
  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    Tensor condition = Tensor::FromBool("condition", {2, 1}, {1, 0});
    Tensor x = Tensor::FromFloat("x", {2, 3}, {kPosInf, 1.0f, kPosInf, 2.0f, kPosInf, 3.0f});
    Tensor y = Tensor::FromFloat("y", {1, 3}, {kNegInf, kNan, 0.0f});
    Tensor output = where_kernel(condition, x, y);

    Expect(node, {condition, x, y}, {output}, "test_cc_where_nan_inf_bcast", {opset},
           "backend-test", registry, "nan_inf");
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

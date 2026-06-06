// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterDetCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::Det det_kernel{ctx};

  // 2-D input: output is a scalar (matches ONNX ``test_det_2d``).
  {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat("", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = det_kernel(x);
    Expect(node, {x}, {y}, "test_cc_det_2d", {opset}, "backend-test", registry);
  }

  // N-D input: batch of square matrices (matches ONNX ``test_det_nd``).
  {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat(
        "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 3.0f, 1.0f});
    Tensor y = det_kernel(x);
    Expect(node, {x}, {y}, "test_cc_det_nd", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

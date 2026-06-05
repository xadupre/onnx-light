// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterReluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Relu relu_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat("", {3, 4, 5}, std::vector<float>(60, -1.0f));
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_example", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

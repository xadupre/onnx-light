// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterSoftsignCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Softsign softsign_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Softsign");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
    Tensor y = softsign_kernel(x);
    Expect(node, {x}, {y}, "test_cc_softsign", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

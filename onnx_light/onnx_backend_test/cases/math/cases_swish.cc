// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterSwishCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(24);
  const kernel::KernelContext ctx{opset};
  const kernel::Swish swish_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Swish");
    node.add_input("X");
    node.add_output("Y");

    // No alpha attribute: defaults to 1.0 (standard Swish / SiLU).
    Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
    Tensor y = swish_kernel(x);
    Expect(node, {x}, {y}, "test_cc_swish", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("Swish");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
    Tensor y = swish_kernel(x, 2.0f);
    Expect(node, {x}, {y}, "test_cc_swish_alpha", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterLeakyReluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::KernelContext ctx{opset};
  const kernel::LeakyRelu leakyrelu_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("LeakyRelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.1f);

    Tensor x = Tensor::FromFloat("", {3, 4, 5}, std::vector<float>(60, -1.0f));
    Tensor y = leakyrelu_kernel(x, 0.1f);
    Expect(node, {x}, {y}, "test_cc_leakyrelu_example", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("LeakyRelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.1f);

    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = leakyrelu_kernel(x, 0.1f);
    Expect(node, {x}, {y}, "test_cc_leakyrelu", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("LeakyRelu");
    node.add_input("X");
    node.add_output("Y");

    // No alpha attribute: defaults to 0.01.
    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = leakyrelu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_leakyrelu_default", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

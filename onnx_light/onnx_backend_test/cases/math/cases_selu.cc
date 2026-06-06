// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterSeluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(6);
  const kernel::KernelContext ctx{opset};
  const kernel::Selu selu_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Selu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    AttributeProto *gamma = node.add_attribute();
    gamma->set_name("gamma");
    gamma->set_type(AttributeProto::FLOAT);
    gamma->set_f(3.0f);

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = selu_kernel(x, 2.0f, 3.0f);
    Expect(node, {x}, {y}, "test_cc_selu_example", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("Selu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    AttributeProto *gamma = node.add_attribute();
    gamma->set_name("gamma");
    gamma->set_type(AttributeProto::FLOAT);
    gamma->set_f(3.0f);

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
    Tensor y = selu_kernel(x, 2.0f, 3.0f);
    Expect(node, {x}, {y}, "test_cc_selu", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("Selu");
    node.add_input("X");
    node.add_output("Y");

    // No attributes: defaults to ONNX schema defaults
    // (alpha=1.67326319217681884765625, gamma=1.05070102214813232421875).
    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
    Tensor y = selu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_selu_default", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

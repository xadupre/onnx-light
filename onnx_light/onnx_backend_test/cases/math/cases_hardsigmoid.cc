// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterHardSigmoidCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::HardSigmoid hard_sigmoid_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("HardSigmoid");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.5f);

    AttributeProto *beta = node.add_attribute();
    beta->set_name("beta");
    beta->set_type(AttributeProto::FLOAT);
    beta->set_f(0.6f);

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = hard_sigmoid_kernel(x, 0.5f, 0.6f);
    Expect(node, {x}, {y}, "test_cc_hardsigmoid_example", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("HardSigmoid");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(0.5f);

    AttributeProto *beta = node.add_attribute();
    beta->set_name("beta");
    beta->set_type(AttributeProto::FLOAT);
    beta->set_f(0.6f);

    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, -0.5f, 0.5f, 1.0f, 3.0f});
    Tensor y = hard_sigmoid_kernel(x, 0.5f, 0.6f);
    Expect(node, {x}, {y}, "test_cc_hardsigmoid", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("HardSigmoid");
    node.add_input("X");
    node.add_output("Y");

    // No alpha/beta attributes: defaults to 0.2 and 0.5.
    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, -0.5f, 0.5f, 1.0f, 3.0f});
    Tensor y = hard_sigmoid_kernel(x, 0.2f, 0.5f);
    Expect(node, {x}, {y}, "test_cc_hardsigmoid_default", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

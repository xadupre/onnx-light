// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterShrinkCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(9);
  const kernel::KernelContext ctx{opset};
  const kernel::Shrink shrink_kernel{ctx};

  {
    // Mirrors the ONNX test_shrink_hard reference (bias=0.0, lambd=1.5).
    NodeProto node;
    node.set_op_type("Shrink");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *lambd = node.add_attribute();
    lambd->set_name("lambd");
    lambd->set_type(AttributeProto::FLOAT);
    lambd->set_f(1.5f);

    Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    Tensor y = shrink_kernel(x, /*bias=*/0.0f, /*lambd=*/1.5f);
    Expect(node, {x}, {y}, "test_cc_shrink_hard", {opset}, "backend-test", registry);
  }

  {
    // Mirrors the ONNX test_shrink_soft reference (bias=1.5, lambd=1.5).
    NodeProto node;
    node.set_op_type("Shrink");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *bias = node.add_attribute();
    bias->set_name("bias");
    bias->set_type(AttributeProto::FLOAT);
    bias->set_f(1.5f);

    AttributeProto *lambd = node.add_attribute();
    lambd->set_name("lambd");
    lambd->set_type(AttributeProto::FLOAT);
    lambd->set_f(1.5f);

    Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    Tensor y = shrink_kernel(x, /*bias=*/1.5f, /*lambd=*/1.5f);
    Expect(node, {x}, {y}, "test_cc_shrink_soft", {opset}, "backend-test", registry);
  }

  {
    // Default attributes: bias=0.0, lambd=0.5.
    NodeProto node;
    node.set_op_type("Shrink");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, -0.1f, 0.1f, 0.5f, 1.0f});
    Tensor y = shrink_kernel(x, /*bias=*/0.0f, /*lambd=*/0.5f);
    Expect(node, {x}, {y}, "test_cc_shrink_default", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterGeluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::KernelContext ctx{opset};
  const kernel::Gelu gelu_kernel{ctx};

  // Default approximate ("none"), 1-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = gelu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_gelu_default_1", {opset}, "backend-test", registry);
  }

  // Default approximate ("none"), 2-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *approximate = node.add_attribute();
    approximate->set_name("approximate");
    approximate->set_type(AttributeProto::STRING);
    approximate->set_s("none");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
    Tensor y = gelu_kernel(x, "none");
    Expect(node, {x}, {y}, "test_cc_gelu_default_2", {opset}, "backend-test", registry);
  }

  // approximate="tanh", 1-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *approximate = node.add_attribute();
    approximate->set_name("approximate");
    approximate->set_type(AttributeProto::STRING);
    approximate->set_s("tanh");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = gelu_kernel(x, "tanh");
    Expect(node, {x}, {y}, "test_cc_gelu_tanh_1", {opset}, "backend-test", registry);
  }

  // approximate="tanh", 2-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *approximate = node.add_attribute();
    approximate->set_name("approximate");
    approximate->set_type(AttributeProto::STRING);
    approximate->set_s("tanh");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
    Tensor y = gelu_kernel(x, "tanh");
    Expect(node, {x}, {y}, "test_cc_gelu_tanh_2", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterSoftmaxCases(std::vector<TestCase> &registry) {
  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::Softmax softmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("Softmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = softmax_kernel(x, 1);
    Expect(node, {x}, {y}, "test_cc_softmax", {opset}, "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::Softmax softmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("Softmax");
    node.add_input("input");
    node.add_output("output");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
    Tensor y = softmax_kernel(x, -1);
    Expect(node, {x}, {y}, "test_cc_softmax_default_axis", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterLogSoftmaxCases(std::vector<TestCase> &registry) {
  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = logsoftmax_kernel(x, 1);
    Expect(node, {x}, {y}, "test_cc_logsoftmax_example_1", {opset}, "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
    Tensor y = logsoftmax_kernel(x, -1);
    Expect(node, {x}, {y}, "test_cc_logsoftmax_default_axis", {opset}, "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(-1);

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
    Tensor y = logsoftmax_kernel(x, -1);
    Expect(node, {x}, {y}, "test_cc_logsoftmax_negative_axis", {opset}, "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(0);

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
    Tensor y = logsoftmax_kernel(x, 0);
    Expect(node, {x}, {y}, "test_cc_logsoftmax_axis_0", {opset}, "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
    Tensor y = logsoftmax_kernel(x, 1);
    Expect(node, {x}, {y}, "test_cc_logsoftmax_axis_1", {opset}, "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(2);

    Tensor x = Tensor::FromFloat(
        "", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 1.5f, 0.0f});
    Tensor y = logsoftmax_kernel(x, 2);
    Expect(node, {x}, {y}, "test_cc_logsoftmax_axis_2", {opset}, "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    // Numerical stability check: large inputs should not overflow.
    Tensor x =
        Tensor::FromFloat("", {2, 3}, {1000.0f, 1001.0f, 1002.0f, 1002.0f, 1001.0f, 1000.0f});
    Tensor y = logsoftmax_kernel(x, -1);
    Expect(node, {x}, {y}, "test_cc_logsoftmax_large_number", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

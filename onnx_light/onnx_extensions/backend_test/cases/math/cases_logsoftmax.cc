// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterLogSoftmaxCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};
    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);
    const std::vector<int64_t> shape = {2048, 2048};
    const int64_t count = 2048 * 2048;
    Expect(registry, std::move(node), "test_cc_logsoftmax_benchmark", {opset}, {count}, {count},
           [logsoftmax_kernel, shape]() -> IoData {
             Tensor x = Tensor::FromFloat("", shape, Randn<float>(shape, 432));
             Tensor y = logsoftmax_kernel(x, 1);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    Expect(registry, std::move(node), "test_cc_logsoftmax_example_1", {opset}, [=]() -> IoData {
      axis->set_i(1);

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = logsoftmax_kernel(x, 1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    Expect(registry, std::move(node), "test_cc_logsoftmax_default_axis", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    Expect(registry, std::move(node), "test_cc_logsoftmax_negative_axis", {opset}, [=]() -> IoData {
      axis->set_i(-1);

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    Expect(registry, std::move(node), "test_cc_logsoftmax_axis_0", {opset}, [=]() -> IoData {
      axis->set_i(0);

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, 0);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    Expect(registry, std::move(node), "test_cc_logsoftmax_axis_1", {opset}, [=]() -> IoData {
      axis->set_i(1);

      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = logsoftmax_kernel(x, 1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");

    AttributeProto *axis = node.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    Expect(registry, std::move(node), "test_cc_logsoftmax_axis_2", {opset}, [=]() -> IoData {
      axis->set_i(2);

      Tensor x = Tensor::FromFloat(
          "", {2, 2, 3},
          {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 1.5f, 0.0f});
      Tensor y = logsoftmax_kernel(x, 2);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    const OpsetId opset = DefaultOpset(13);
    const KernelContext ctx{opset};
    const onnx_kernels::kernel::LogSoftmax logsoftmax_kernel{ctx};

    NodeProto node;
    node.set_op_type("LogSoftmax");
    node.add_input("input");
    node.add_output("output");
    Expect(registry, std::move(node), "test_cc_logsoftmax_large_number", {opset}, [=]() -> IoData {
      // Numerical stability check: large inputs should not overflow.
      Tensor x =
          Tensor::FromFloat("", {2, 3}, {1000.0f, 1001.0f, 1002.0f, 1002.0f, 1001.0f, 1000.0f});
      Tensor y = logsoftmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

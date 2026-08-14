// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeSoftmaxNode(int64_t axis, bool include_axis = true) {
  NodeProto node;
  node.set_op_type("Softmax");
  node.add_input("input");
  node.add_output("output");
  if (include_axis) {
    AttributeProto *axis_attr = node.add_attribute();
    axis_attr->set_name("axis");
    axis_attr->set_type(AttributeProto::INT);
    axis_attr->set_i(axis);
  }
  return node;
}

} // namespace

void RegisterSoftmaxCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Softmax softmax_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeSoftmaxNode(/*axis=*/1);
    const std::vector<int64_t> shape = {2048, 2048};
    const int64_t count = 2048 * 2048;
    Expect(registry, std::move(node), "test_cc_softmax_benchmark", {opset}, {count}, {count},
           [softmax_kernel, shape]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, shape, 431);
             Tensor y = softmax_kernel(x, 1);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Two-dimensional input with an explicit axis attribute.
  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/1);
    Expect(registry, std::move(node), "test_cc_softmax", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = softmax_kernel(x, 1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Default axis (-1 in opset 13). Mirrors ONNX ``test_softmax_example``.
  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/0, /*include_axis=*/false);
    Expect(registry, std::move(node), "test_cc_softmax_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {1, 3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = softmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Large values along the default axis. Mirrors ONNX ``test_softmax_large_number``.
  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/0, /*include_axis=*/false);
    Expect(registry, std::move(node), "test_cc_softmax_large_number", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat(
          "", {2, 4}, {0.0f, 1.0f, 2.0f, 3.0f, 10000.0f, 10001.0f, 10002.0f, 10003.0f});
      Tensor y = softmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Rank-3 input reduced along each axis. Mirrors ONNX ``test_softmax_axis_*``
  // and ``test_softmax_negative_axis``.
  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/0);
    Expect(registry, std::move(node), "test_cc_softmax_axis_0", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, 431);
      Tensor y = softmax_kernel(x, 0);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/1);
    Expect(registry, std::move(node), "test_cc_softmax_axis_1", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, 432);
      Tensor y = softmax_kernel(x, 1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/2);
    Expect(registry, std::move(node), "test_cc_softmax_axis_2", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, 433);
      Tensor y = softmax_kernel(x, 2);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/-1);
    Expect(registry, std::move(node), "test_cc_softmax_negative_axis", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, 434);
      Tensor y = softmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Default axis (-1 in opset 13) on a small input.
  {
    NodeProto node = MakeSoftmaxNode(/*axis=*/0, /*include_axis=*/false);
    Expect(registry, std::move(node), "test_cc_softmax_default_axis", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = softmax_kernel(x, -1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

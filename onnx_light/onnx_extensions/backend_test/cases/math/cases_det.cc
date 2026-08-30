// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterDetCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);
  const auto det_kernel = MakeReferenceKernel<onnx_kernels::kernel::Det>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    const std::vector<int64_t> shape = {512, 64, 64};
    Expect(registry, std::move(node), "test_cc_det_benchmark", {opset}, {512 * 64 * 64}, {512},
           [det_kernel, shape]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, shape, 439);
             Tensor y = det_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // 2-D input: output is a scalar (matches ONNX ``test_det_2d``).
  {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_det_2d", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = det_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // N-D input: batch of square matrices (matches ONNX ``test_det_nd``).
  {
    NodeProto node;
    node.set_op_type("Det");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_det_nd", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat(
          "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 3.0f, 1.0f});
      Tensor y = det_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

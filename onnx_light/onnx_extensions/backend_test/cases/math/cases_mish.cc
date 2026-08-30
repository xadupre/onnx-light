// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterMishCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const auto mish_kernel = MakeReferenceKernel<onnx_kernels::kernel::Mish>(opset);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Mish>("Mish", "test_cc_mish_benchmark", opset,
                                                          registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Mish");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_mish", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
      Tensor y = mish_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

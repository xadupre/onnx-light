// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterHardSwishCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::HardSwish>(
        "HardSwish", "test_cc_hardswish_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("HardSwish");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_hardswish", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext hard_swish_kernel_ctx{opset};
      const onnx_kernels::kernel::HardSwish hard_swish_kernel{hard_swish_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
      Tensor y = hard_swish_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

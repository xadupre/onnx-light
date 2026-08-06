// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterSwishCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(24);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Swish swish_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Swish", swish_kernel, "test_cc_swish_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Swish");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_swish", {opset}, [=]() -> IoData {
      // No alpha attribute: defaults to 1.0 (standard Swish / SiLU).
      Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
      Tensor y = swish_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Swish");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_swish_alpha", {opset}, [=]() -> IoData {
      alpha->set_f(2.0f);

      Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
      Tensor y = swish_kernel(x, 2.0f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterCeluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(12);
  const auto celu_kernel = MakeReferenceKernel<onnx_kernels::kernel::Celu>(opset);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Celu>("Celu", "test_cc_celu_benchmark", opset,
                                                          registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_celu", {opset}, [=]() -> IoData {
      alpha->set_f(2.0f);

      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = celu_kernel.Invoke([&](const auto &kernel) { return kernel(x, 2.0f); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_celu_default", {opset}, [=]() -> IoData {
      // No alpha attribute: defaults to 1.0.
      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = celu_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Celu-28 opset: required for float16 and bfloat16 type constraints.
  const OpsetId opset28 = DefaultOpset(28);

  // FLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_celu_float16", {opset28}, [=]() -> IoData {
      alpha->set_f(2.0f);

      Tensor x = MakeFloat16Tensor("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = celu_kernel.Invoke([&](const auto &kernel) { return kernel(x, 2.0f); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_celu_bfloat16", {opset28}, [=]() -> IoData {
      alpha->set_f(1.0f);

      Tensor x = MakeBfloat16Tensor("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = celu_kernel.Invoke([&](const auto &kernel) { return kernel(x, 1.0f); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

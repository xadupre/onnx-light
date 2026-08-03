// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterLeakyReluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(16);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::LeakyRelu leakyrelu_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("LeakyRelu", leakyrelu_kernel, "test_cc_leakyrelu_benchmark", opset,
                              registry, false);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("LeakyRelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_leakyrelu_example", {opset}, [=]() -> IoData {
      alpha->set_f(0.1f);

      Tensor x = Tensor::FromFloat("", {3, 4, 5}, std::vector<float>(60, -1.0f));
      Tensor y = leakyrelu_kernel(x, 0.1f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("LeakyRelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_leakyrelu", {opset}, [=]() -> IoData {
      alpha->set_f(0.1f);

      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = leakyrelu_kernel(x, 0.1f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("LeakyRelu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_leakyrelu_default", {opset}, [=]() -> IoData {
      // No alpha attribute: defaults to 0.01.
      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = leakyrelu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

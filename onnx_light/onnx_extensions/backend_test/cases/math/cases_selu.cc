// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterSeluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(6);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Selu selu_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Selu", selu_kernel, "test_cc_selu_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Selu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    AttributeProto *gamma = node.add_attribute();
    gamma->set_name("gamma");
    gamma->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_selu_example", {opset}, [=]() -> IoData {
      gamma->set_f(3.0f);

      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = selu_kernel(x, 2.0f, 3.0f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Selu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    AttributeProto *gamma = node.add_attribute();
    gamma->set_name("gamma");
    gamma->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_selu", {opset}, [=]() -> IoData {
      gamma->set_f(3.0f);

      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = selu_kernel(x, 2.0f, 3.0f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Selu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_selu_default", {opset}, [=]() -> IoData {
      // No attributes: defaults to ONNX schema defaults
      // (alpha=1.67326319217681884765625, gamma=1.05070102214813232421875).
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = selu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

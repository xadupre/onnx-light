// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterEluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(6);
  const auto elu_kernel = MakeReferenceKernel<onnx_kernels::kernel::Elu>(opset);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Elu>("Elu", "test_cc_elu_benchmark", opset,
                                                         registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Elu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_elu_example", {opset}, [=]() -> IoData {
      alpha->set_f(2.0f);

      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = elu_kernel.Invoke([&](const auto &kernel) { return kernel(x, 2.0f); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Elu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_elu", {opset}, [=]() -> IoData {
      alpha->set_f(2.0f);

      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = elu_kernel.Invoke([&](const auto &kernel) { return kernel(x, 2.0f); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Elu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_elu_default", {opset}, [=]() -> IoData {
      // No alpha attribute: defaults to 1.0.
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = elu_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // FLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Elu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_elu_float16", {opset}, [=]() -> IoData {
      alpha->set_f(2.0f);

      Tensor x = MakeFloat16Tensor("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = elu_kernel.Invoke([&](const auto &kernel) { return kernel(x, 2.0f); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Elu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    Expect(registry, std::move(node), "test_cc_elu_bfloat16", {opset}, [=]() -> IoData {
      alpha->set_f(1.0f);

      std::vector<float> vals = {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i) {
        dst[i] = FloatToBfloat16Bits(vals[i]);
      }
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = elu_kernel.Invoke([&](const auto &kernel) { return kernel(x, 1.0f); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

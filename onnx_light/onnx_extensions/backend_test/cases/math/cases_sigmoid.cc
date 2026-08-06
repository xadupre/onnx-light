// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterSigmoidCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Sigmoid sigmoid_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Sigmoid", sigmoid_kernel, "test_cc_sigmoid_benchmark", opset,
                              registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Sigmoid");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_sigmoid", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
      Tensor y = sigmoid_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Sigmoid");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sigmoid_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
      Tensor y = sigmoid_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Sigmoid");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sigmoid_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = sigmoid_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // DOUBLE
  {
    NodeProto node;
    node.set_op_type("Sigmoid");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sigmoid_double", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {2, 3}, {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0});
      Tensor y = sigmoid_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

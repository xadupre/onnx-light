// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

Tensor NonNegativeRandFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  // Sqrt is only defined for non-negative inputs (negatives produce NaN).
  std::vector<float> values = Rand<float>(shape, seed);
  for (float &v : values) {
    if (v < 0.0f) {
      v = -v;
    }
  }
  return Tensor::FromFloat("", shape, values);
}

} // namespace

// ---------------------------------------------------------------------------
// Sqrt — y = sqrt(x) (latest opset: 13).
// Registers a small deterministic ``test_cc_sqrt`` case and upstream ONNX
// backend test cases (``test_sqrt_example`` and ``test_sqrt``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterSqrtCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto sqrt_kernel = MakeReferenceKernel<onnx_kernels::kernel::Sqrt>(opset);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Sqrt>("Sqrt", "test_cc_sqrt_benchmark", opset,
                                                          registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sqrt", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 0.25f, 1.0f, 2.0f, 4.0f, 9.0f});
      Tensor y = sqrt_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Sqrt.export(): ``test_sqrt_example`` uses x = [1, 4, 9].
  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_sqrt_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 9.0f});
      Tensor y = sqrt_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Sqrt.export(): ``test_sqrt`` uses np.abs(np.random.randn(3, 4, 5)).
  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_sqrt", {opset}, [=]() -> IoData {
      Tensor x = NonNegativeRandFloat({3, 4, 5}, /*seed=*/1);
      Tensor y = sqrt_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sqrt_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {0.0f, 0.25f, 1.0f, 2.25f, 4.0f, 9.0f});
      Tensor y = sqrt_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sqrt_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {0.0f, 0.25f, 1.0f, 2.25f, 4.0f, 9.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = sqrt_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

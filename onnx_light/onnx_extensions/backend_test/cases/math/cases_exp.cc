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

Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return RandnTensor(DataType::FLOAT, shape, seed);
}

} // namespace

// ---------------------------------------------------------------------------
// Exp — y = exp(x) (latest opset: 13).
// Registers both a small deterministic ``test_cc_exp`` case and upstream ONNX
// backend test cases (``test_exp_example`` and ``test_exp``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterExpCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto exp_kernel = MakeReferenceKernel<onnx_kernels::kernel::Exp>(opset);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Exp>("Exp", "test_cc_exp_benchmark", opset,
                                                         registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_exp", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
      Tensor y = exp_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Exp.export(): ``test_exp_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_exp_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = exp_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Exp.export(): ``test_exp`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_exp", {opset}, [=]() -> IoData {
      Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
      Tensor y = exp_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_exp_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
      Tensor y = exp_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_exp_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = exp_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

Tensor PositiveRandFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  std::vector<float> values = Rand<float>(shape, seed);
  for (float &v : values) {
    v += 1.0e-6f;
  }
  return Tensor::FromFloat("", shape, values);
}

} // namespace

// ---------------------------------------------------------------------------
// Log — y = log(x) (latest opset: 13).
// Registers both a small deterministic ``test_cc_log`` case and upstream ONNX
// backend test cases (``test_log_example`` and ``test_log``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterLogCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Log log_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Log", log_kernel, "test_cc_log_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_log", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {0.1f, 0.5f, 1.0f, 2.0f, 4.0f, 10.0f});
      Tensor y = log_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Log.export(): ``test_log_example`` uses positive inputs.
  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_log_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2}, {1.0f, 10.0f});
      Tensor y = log_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Log.export(): ``test_log`` uses random positive inputs.
  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_log", {opset}, [=]() -> IoData {
      Tensor x = PositiveRandFloat({3, 4, 5}, /*seed=*/1);
      Tensor y = log_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_log_float16", {opset}, [=]() -> IoData {
      Tensor x = kernel::MakeFloat16Tensor("", {2, 3}, {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f});
      Tensor y = log_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Log");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_log_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = kernel::FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = log_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

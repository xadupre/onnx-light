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

Tensor PositiveRandFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  // Match onnx.backend.test.case.node.reciprocal: np.random.rand(...) + 0.5
  // (strictly positive, avoids dividing by zero or values close to zero).
  std::vector<float> values = Rand<float>(shape, seed);
  for (float &v : values) {
    if (v < 0.0f) {
      v = -v;
    }
    v += 0.5f;
  }
  return Tensor::FromFloat("", shape, values);
}

} // namespace

// ---------------------------------------------------------------------------
// Reciprocal — y = 1 / x (latest opset: 13).
// Registers a small deterministic ``test_cc_reciprocal`` case and the upstream
// ONNX backend test cases (``test_reciprocal_example`` and
// ``test_reciprocal``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterReciprocalCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Reciprocal reciprocal_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Reciprocal", reciprocal_kernel, "test_cc_reciprocal_benchmark",
                              opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_reciprocal", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -0.5f, 0.25f, 1.0f, 2.0f, 4.0f});
      Tensor y = reciprocal_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Reciprocal.export(): ``test_reciprocal_example`` uses x = [-4, 2].
  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_reciprocal_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2}, {-4.0f, 2.0f});
      Tensor y = reciprocal_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Reciprocal.export(): ``test_reciprocal`` uses np.random.rand(3, 4, 5) + 0.5.
  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_reciprocal", {opset}, [=]() -> IoData {
      Tensor x = PositiveRandFloat({3, 4, 5}, /*seed=*/1);
      Tensor y = reciprocal_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_reciprocal_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {0.5f, 1.0f, 2.0f, 4.0f, 0.25f, 8.0f});
      Tensor y = reciprocal_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_reciprocal_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {0.5f, 1.0f, 2.0f, 4.0f, 0.25f, 8.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = reciprocal_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

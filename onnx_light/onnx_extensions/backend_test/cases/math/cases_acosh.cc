// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Deterministic FLOAT tensor with values uniformly drawn from ``[low, high)``
// (the valid domain of acosh is [1, +inf)). Built from the repository's
// SplitMix64-based ``Rand`` generator.
Tensor RandFloatInRange(const std::vector<int64_t> &shape, float low, float high, uint64_t seed) {
  std::vector<float> values = Rand<float>(shape, seed);
  for (float &v : values) {
    v = low + (high - low) * v;
  }
  return Tensor::FromFloat("", shape, values);
}

} // namespace

// ---------------------------------------------------------------------------
// Acosh — y = acosh(x) (since opset 9, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_acosh`` case and the
// upstream ONNX backend test cases (``test_acosh_example`` and
// ``test_acosh``) mirrored from ``onnx.backend.test.case.node.acosh.Acosh``
// for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterAcoshCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Acosh acosh_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Acosh", acosh_kernel, "test_cc_acosh_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Acosh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_acosh", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 1.25f, 1.5f, 2.0f, 3.5f, 10.0f});
      Tensor y = acosh_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Acosh`` operator (mirror the
  // ``onnx.backend.test.case.node.acosh.Acosh`` Python class).
  //
  // From Acosh.export(): ``test_acosh_example`` uses x = [10, e, 1].
  {
    NodeProto node;
    node.set_op_type("Acosh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_acosh_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {10.0f, static_cast<float>(std::exp(1.0)), 1.0f});
      Tensor y = acosh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Acosh.export(): ``test_acosh`` uses np.random.uniform(1, 10, (3,4,5)).
  {
    NodeProto node;
    node.set_op_type("Acosh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_acosh", {opset}, [=]() -> IoData {
      Tensor x = RandFloatInRange({3, 4, 5}, /*low=*/1.0f, /*high=*/10.0f, /*seed=*/1);
      Tensor y = acosh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

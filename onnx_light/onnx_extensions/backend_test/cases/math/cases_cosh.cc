// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Deterministic FLOAT tensor with approximately-normal values produced by
// the repository's SplitMix64-based ``Randn`` generator (Irwin-Hall
// approximation). cosh is defined for every real input so no clamping is
// required.
Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Cosh — y = cosh(x) (since opset 9, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_cosh`` case and the upstream
// ONNX backend test cases (``test_cosh_example`` and ``test_cosh``) mirrored
// from ``onnx.backend.test.case.node.cosh.Cosh`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterCoshCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Cosh cosh_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Cosh", cosh_kernel, "test_cc_cosh_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Cosh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_cosh", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f});
      Tensor y = cosh_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Cosh`` operator (mirror the
  // ``onnx.backend.test.case.node.cosh.Cosh`` Python class).
  //
  // From Cosh.export(): ``test_cosh_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Cosh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cosh_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = cosh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Cosh.export(): ``test_cosh`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Cosh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cosh", {opset}, [=]() -> IoData {
      Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
      Tensor y = cosh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

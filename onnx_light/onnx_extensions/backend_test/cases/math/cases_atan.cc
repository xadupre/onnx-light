// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Deterministic FLOAT tensor with approximately-normal values produced by
// the repository's SplitMix64-based ``Randn`` generator (Irwin-Hall
// approximation). atan is defined for every real input so no clamping is
// required.
Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return RandnTensor(DataType::FLOAT, shape, seed);
}

} // namespace

// ---------------------------------------------------------------------------
// Atan — y = atan(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_atan`` case and the upstream
// ONNX backend test cases (``test_atan_example`` and ``test_atan``) mirrored
// from ``onnx.backend.test.case.node.atan.Atan`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterAtanCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Atan atan_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Atan", atan_kernel, "test_cc_atan_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Atan");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_atan", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f});
      Tensor y = atan_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Atan`` operator (mirror the
  // ``onnx.backend.test.case.node.atan.Atan`` Python class).
  //
  // From Atan.export(): ``test_atan_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Atan");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_atan_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = atan_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Atan.export(): ``test_atan`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Atan");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_atan", {opset}, [=]() -> IoData {
      Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
      Tensor y = atan_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

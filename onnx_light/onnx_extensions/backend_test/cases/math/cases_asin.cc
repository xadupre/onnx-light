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

// ---------------------------------------------------------------------------
// Asin — y = asin(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_asin`` case and the upstream
// ONNX backend test cases (``test_asin_example`` and ``test_asin``) mirrored
// from ``onnx.backend.test.case.node.asin.Asin`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterAsinCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Asin asin_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Asin", asin_kernel, "test_cc_asin_benchmark", opset, registry,
                              false);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Asin");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_asin", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f});
      Tensor y = asin_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Asin`` operator (mirror the
  // ``onnx.backend.test.case.node.asin.Asin`` Python class).
  //
  // From Asin.export(): ``test_asin_example`` uses x = [-0.5, 0, 0.5].
  {
    NodeProto node;
    node.set_op_type("Asin");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_asin_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-0.5f, 0.0f, 0.5f});
      Tensor y = asin_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Asin.export(): ``test_asin`` uses x = np.random.rand(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Asin");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_asin", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = Tensor::FromFloat("", shape, Rand<float>(shape, /*seed=*/1));
      Tensor y = asin_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

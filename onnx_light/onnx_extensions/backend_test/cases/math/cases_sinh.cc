// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Sinh — y = sinh(x) (since opset 9, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_sinh`` case and the
// upstream ONNX backend test cases (``test_sinh_example`` and ``test_sinh``)
// mirrored from ``onnx.backend.test.case.node.sinh.Sinh`` for FLOAT.
// ---------------------------------------------------------------------------
void RegisterSinhCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Sinh sinh_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Sinh", sinh_kernel, "test_cc_sinh_benchmark", opset, registry,
                              false);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Sinh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sinh", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
      Tensor y = sinh_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Sinh`` operator (mirror the
  // ``onnx.backend.test.case.node.sinh.Sinh`` Python class).
  //
  // From Sinh.export(): ``test_sinh_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Sinh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_sinh_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = sinh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Sinh.export(): ``test_sinh`` uses x = np.random.rand(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Sinh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_sinh", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = Tensor::FromFloat("", shape, Rand<float>(shape, /*seed=*/1));
      Tensor y = sinh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

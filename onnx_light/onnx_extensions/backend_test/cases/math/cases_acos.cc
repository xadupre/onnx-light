// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Acos — y = acos(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_acos`` case and the upstream
// ONNX backend test cases (``test_acos_example`` and ``test_acos``) mirrored
// from ``onnx.backend.test.case.node.acos.Acos`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterAcosCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Acos acos_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Acos", acos_kernel, "test_cc_acos_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Acos");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_acos", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f});
      Tensor y = acos_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Acos`` operator (mirror the
  // ``onnx.backend.test.case.node.acos.Acos`` Python class).
  //
  // From Acos.export(): ``test_acos_example`` uses x = [-0.5, 0, 0.5].
  {
    NodeProto node;
    node.set_op_type("Acos");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_acos_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-0.5f, 0.0f, 0.5f});
      Tensor y = acos_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Acos.export(): ``test_acos`` uses x = np.random.rand(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Acos");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_acos", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = Tensor::FromFloat("", shape, Rand<float>(shape, /*seed=*/1));
      Tensor y = acos_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

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

// ---------------------------------------------------------------------------
// Atanh — y = atanh(x) (since opset 9, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_atanh`` case and the
// upstream ONNX backend test cases (``test_atanh_example`` and
// ``test_atanh``) mirrored from ``onnx.backend.test.case.node.atanh.Atanh``
// for the FLOAT variant. atanh is defined on the open interval (-1, 1).
// ---------------------------------------------------------------------------
void RegisterAtanhCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Atanh atanh_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Atanh", atanh_kernel, "test_cc_atanh_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Atanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_atanh", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-0.9f, -0.5f, 0.0f, 0.25f, 0.5f, 0.9f});
      Tensor y = atanh_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Atanh`` operator (mirror the
  // ``onnx.backend.test.case.node.atanh.Atanh`` Python class).
  //
  // From Atanh.export(): ``test_atanh_example`` uses x = [-0.5, 0, 0.5].
  {
    NodeProto node;
    node.set_op_type("Atanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_atanh_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-0.5f, 0.0f, 0.5f});
      Tensor y = atanh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Atanh.export(): ``test_atanh`` uses np.random.uniform(0.0, 1.0, (3,4,5)).
  {
    NodeProto node;
    node.set_op_type("Atanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_atanh", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = Tensor::FromFloat("", shape, Rand<float>(shape, /*seed=*/1));
      Tensor y = atanh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// BlackmanWindow — generates a Blackman window of length ``size`` (since opset
// 17). Registers both the default periodic case and the symmetric variant
// (``periodic = 0``).
// ---------------------------------------------------------------------------
void RegisterBlackmanWindowCases(std::vector<TestCase> &registry, TestMode mode) {
  constexpr int32_t kSize = 10;
  const OpsetId opset = DefaultOpset(17);
  const auto blackman_kernel = MakeReferenceKernel<onnx_kernels::kernel::BlackmanWindow>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("BlackmanWindow");
    node.add_input("x");
    node.add_output("y");
    constexpr int32_t size = 1 << 22;
    Expect(registry, std::move(node), "test_cc_blackmanwindow_benchmark", {opset}, {1}, {size},
           [blackman_kernel, size]() -> IoData {
             Tensor x = Tensor::FromInt32("", {}, {size});
             Tensor y = blackman_kernel.Invoke(
                 [&](const auto &kernel) { return kernel(x, /*periodic=*/true); });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Default periodic variant.
  {
    NodeProto node;
    node.set_op_type("BlackmanWindow");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_blackmanwindow", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {}, {kSize});
      Tensor y =
          blackman_kernel.Invoke([&](const auto &kernel) { return kernel(x, /*periodic=*/true); });

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Symmetric variant (periodic = 0).
  {
    NodeProto node;
    node.set_op_type("BlackmanWindow");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("periodic");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(0);
    Expect(registry, std::move(node), "test_cc_blackmanwindow_symmetric", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {}, {kSize});
      Tensor y =
          blackman_kernel.Invoke([&](const auto &kernel) { return kernel(x, /*periodic=*/false); });

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

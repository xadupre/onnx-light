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

// ---------------------------------------------------------------------------
// Tanh — y = tanh(x) (since opset 1, type widening at opset 6 and 13).
// Registers both a small deterministic ``test_cc_tanh`` case and the upstream
// ONNX backend test cases (``test_tanh_example`` and ``test_tanh``) mirrored
// from ``onnx.backend.test.case.node.tanh.Tanh`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterTanhCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Tanh>("Tanh", "test_cc_tanh_benchmark", opset,
                                                          registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("input");
    node.add_output("output");
    Expect(registry, std::move(node), "test_cc_tanh", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext tanh_kernel_ctx{opset};
      const onnx_kernels::kernel::Tanh tanh_kernel{tanh_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
      Tensor y = tanh_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Tanh`` operator (mirror the
  // ``onnx.backend.test.case.node.tanh.Tanh`` Python class).
  //
  // From Tanh.export(): ``test_tanh_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_tanh_example", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext tanh_kernel_ctx{opset};
      const onnx_kernels::kernel::Tanh tanh_kernel{tanh_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = tanh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Tanh.export(): ``test_tanh`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_tanh", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext tanh_kernel_ctx{opset};
      const onnx_kernels::kernel::Tanh tanh_kernel{tanh_kernel_ctx};

      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/1);
      Tensor y = tanh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_tanh_float16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext tanh_kernel_ctx{opset};
      const onnx_kernels::kernel::Tanh tanh_kernel{tanh_kernel_ctx};

      Tensor x = MakeFloat16Tensor("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
      Tensor y = tanh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_tanh_bfloat16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext tanh_kernel_ctx{opset};
      const onnx_kernels::kernel::Tanh tanh_kernel{tanh_kernel_ctx};

      std::vector<float> vals = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = tanh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // DOUBLE
  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_tanh_double", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext tanh_kernel_ctx{opset};
      const onnx_kernels::kernel::Tanh tanh_kernel{tanh_kernel_ctx};

      Tensor x = Tensor::FromDouble("", {2, 3}, {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0});
      Tensor y = tanh_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

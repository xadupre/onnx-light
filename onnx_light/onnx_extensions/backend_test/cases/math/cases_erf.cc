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
// Erf — y = erf(x) (since opset 9, widened at opset 13).
// Registers both a small deterministic ``test_cc_erf`` case and upstream ONNX
// backend test cases (``test_erf_example`` and ``test_erf``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterErfCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Erf>("Erf", "test_cc_erf_benchmark", opset,
                                                         registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Erf");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_erf", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext erf_kernel_ctx{opset};
      const onnx_kernels::kernel::Erf erf_kernel{erf_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
      Tensor y = erf_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Erf.export(): ``test_erf_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Erf");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_erf_example", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext erf_kernel_ctx{opset};
      const onnx_kernels::kernel::Erf erf_kernel{erf_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = erf_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Erf.export(): ``test_erf`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Erf");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_erf", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext erf_kernel_ctx{opset};
      const onnx_kernels::kernel::Erf erf_kernel{erf_kernel_ctx};

      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/1);
      Tensor y = erf_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Erf");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_erf_float16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext erf_kernel_ctx{opset};
      const onnx_kernels::kernel::Erf erf_kernel{erf_kernel_ctx};

      Tensor x = MakeFloat16Tensor("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
      Tensor y = erf_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Erf");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_erf_bfloat16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext erf_kernel_ctx{opset};
      const onnx_kernels::kernel::Erf erf_kernel{erf_kernel_ctx};

      std::vector<float> vals = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = erf_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

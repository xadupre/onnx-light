// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// IsNaN — y = isnan(x), element-wise on a floating-point tensor. Output is
// BOOL. Registers a deterministic ``test_cc_isnan`` case and the upstream
// ONNX backend test cases ``test_isnan`` / ``test_isnan_float16`` mirrored
// from ``onnx.backend.test.case.node.isnan.IsNaN.export*``. Also registers
// a deterministic ``test_cc_isnan_bfloat16`` case exercising the BFLOAT16
// branch of the kernel.
// ---------------------------------------------------------------------------
void RegisterIsNaNCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(20);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::IsNaN isnan_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_isnan_benchmark", {opset}, {count}, {count},
           [isnan_kernel, count]() -> IoData {
             Tensor x = Tensor::FromFloat("", {count}, Randn<float>({count}, /*seed=*/9301));
             Tensor y = isnan_kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const float inf_v = std::numeric_limits<float>::infinity();

  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_cc_isnan", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, nan_v, 2.0f});
      Tensor y = isnan_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ``onnx.backend.test.case.node.isnan.IsNaN.export``:
  //   x = [-1.2, NaN, +inf, 2.8, -inf, +inf] as float32
  //   y = np.isnan(x)
  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_isnan", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});
      Tensor y = isnan_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ``onnx.backend.test.case.node.isnan.IsNaN.export_float16``:
  //   x = [-1.2, NaN, +inf, 2.8, -inf, +inf] as float16
  //   y = np.isnan(x)
  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_isnan_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});
      Tensor y = isnan_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Deterministic BFLOAT16 case exercising the BFLOAT16 branch of the
  // kernel (no upstream equivalent).
  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_cc_isnan_bfloat16", {opset}, [=]() -> IoData {
      const std::vector<float> vals = {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {6}, std::move(raw));
      Tensor y = isnan_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // DOUBLE
  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_cc_isnan_double", {opset}, [=]() -> IoData {
      const double nan_d = std::numeric_limits<double>::quiet_NaN();
      const double inf_d = std::numeric_limits<double>::infinity();
      Tensor x = Tensor::FromDouble("", {6}, {-1.2, nan_d, inf_d, 2.8, -inf_d, nan_d});
      Tensor y = isnan_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

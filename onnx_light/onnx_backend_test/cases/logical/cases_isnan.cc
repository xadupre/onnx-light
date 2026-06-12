// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// IsNaN — y = isnan(x), element-wise on a floating-point tensor. Output is
// BOOL. Registers a deterministic ``test_cc_isnan`` case and the upstream
// ONNX backend test cases ``test_isnan`` / ``test_isnan_float16`` mirrored
// from ``onnx.backend.test.case.node.isnan.IsNaN.export*``. Also registers
// a deterministic ``test_cc_isnan_bfloat16`` case exercising the BFLOAT16
// branch of the kernel.
// ---------------------------------------------------------------------------
void RegisterIsNaNCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::KernelContext ctx{opset};
  const kernel::IsNaN isnan_kernel{ctx};
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const float inf_v = std::numeric_limits<float>::infinity();

  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, nan_v, 2.0f});
    Tensor y = isnan_kernel(x);
    Expect(node, {x}, {y}, "test_cc_isnan", {opset}, "backend-test", registry);
  }

  // Upstream ``onnx.backend.test.case.node.isnan.IsNaN.export``:
  //   x = [-1.2, NaN, +inf, 2.8, -inf, +inf] as float32
  //   y = np.isnan(x)
  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Tensor x = Tensor::FromFloat("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});
    Tensor y = isnan_kernel(x);
    Expect(node, {x}, {y}, "test_isnan", {opset}, "backend-test", registry);
  }

  // Upstream ``onnx.backend.test.case.node.isnan.IsNaN.export_float16``:
  //   x = [-1.2, NaN, +inf, 2.8, -inf, +inf] as float16
  //   y = np.isnan(x)
  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    Tensor x = kernel::MakeFloat16Tensor("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});
    Tensor y = isnan_kernel(x);
    Expect(node, {x}, {y}, "test_isnan_float16", {opset}, "backend-test", registry);
  }

  // Deterministic BFLOAT16 case exercising the BFLOAT16 branch of the
  // kernel (no upstream equivalent).
  {
    NodeProto node = MakeNode("IsNaN", {"x"}, {"y"});
    const std::vector<float> vals = {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v};
    std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
    auto *dst = reinterpret_cast<uint16_t *>(raw.data());
    for (size_t i = 0; i < vals.size(); ++i)
      dst[i] = kernel::FloatToBfloat16Bits(vals[i]);
    Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {6}, std::move(raw));
    Tensor y = isnan_kernel(x);
    Expect(node, {x}, {y}, "test_cc_isnan_bfloat16", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

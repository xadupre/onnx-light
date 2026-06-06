// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// IsInf — y = isinf(x), element-wise on a FLOAT tensor with optional
// ``detect_positive`` / ``detect_negative`` boolean attributes (default
// to 1). Output is BOOL.
//
// Registers a deterministic ``test_cc_isinf`` case and the three upstream
// ONNX backend test cases (``test_isinf``, ``test_isinf_positive``,
// ``test_isinf_negative``) mirrored from
// ``onnx.backend.test.case.node.isinf.IsInf``.
// ---------------------------------------------------------------------------
void RegisterIsInfCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::KernelContext ctx{opset};
  const kernel::IsInf isinf_kernel{ctx};
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const float inf_v = std::numeric_limits<float>::infinity();

  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, inf_v, -inf_v});
    Tensor y = isinf_kernel(x);
    Expect(node, {x}, {y}, "test_cc_isinf", {opset}, "backend-test", registry);
  }

  // Upstream ``onnx.backend.test.case.node.isinf.IsInf.export_infinity``:
  //   x = [-1.2, NaN, +inf, 2.8, -inf, +inf] as float32
  //   y = np.isinf(x)
  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    Tensor x = Tensor::FromFloat("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});
    Tensor y = isinf_kernel(x);
    Expect(node, {x}, {y}, "test_isinf", {opset}, "backend-test", registry);
  }

  // Upstream ``IsInf.export_positive_infinity_only`` (detect_negative=0):
  //   x = [-1.7, NaN, +inf, 3.6, -inf, +inf] as float32
  //   y = np.isposinf(x)
  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    AddAttribute<int64_t>(node, "detect_negative", 0);
    Tensor x = Tensor::FromFloat("", {6}, {-1.7f, nan_v, inf_v, 3.6f, -inf_v, inf_v});
    Tensor y = isinf_kernel(x, /*detect_positive=*/1, /*detect_negative=*/0);
    Expect(node, {x}, {y}, "test_isinf_positive", {opset}, "backend-test", registry);
  }

  // Upstream ``IsInf.export_negative_infinity_only`` (detect_positive=0):
  //   x = [-1.7, NaN, +inf, -3.6, -inf, +inf] as float32
  //   y = np.isneginf(x)
  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    AddAttribute<int64_t>(node, "detect_positive", 0);
    Tensor x = Tensor::FromFloat("", {6}, {-1.7f, nan_v, inf_v, -3.6f, -inf_v, inf_v});
    Tensor y = isinf_kernel(x, /*detect_positive=*/0, /*detect_negative=*/1);
    Expect(node, {x}, {y}, "test_isinf_negative", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

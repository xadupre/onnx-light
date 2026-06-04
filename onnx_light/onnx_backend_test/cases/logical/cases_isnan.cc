// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// IsNaN — y = isnan(x), element-wise on a FLOAT tensor. Output is BOOL.
// Registers a deterministic ``test_cc_isnan`` case and the upstream ONNX
// backend test case ``test_isnan`` mirrored from
// ``onnx.backend.test.case.node.isnan.IsNaN.export``.
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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

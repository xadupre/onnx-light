// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
void RegisterIsInfCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(20);
  const auto isinf_kernel = MakeReferenceKernel<onnx_kernels::kernel::IsInf>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_isinf_benchmark", {opset}, {count}, {count},
           [isinf_kernel, count]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {count}, /*seed=*/9301);
             Tensor y = isinf_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const float inf_v = std::numeric_limits<float>::infinity();

  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_cc_isinf", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, inf_v, -inf_v});
      Tensor y = isinf_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ``onnx.backend.test.case.node.isinf.IsInf.export_infinity``:
  //   x = [-1.2, NaN, +inf, 2.8, -inf, +inf] as float32
  //   y = np.isinf(x)
  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_isinf", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});
      Tensor y = isinf_kernel.Invoke([&](const auto &kernel) { return kernel(x); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ``IsInf.export_positive_infinity_only`` (detect_negative=0):
  //   x = [-1.7, NaN, +inf, 3.6, -inf, +inf] as float32
  //   y = np.isposinf(x)
  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    AddAttribute<int64_t>(node, "detect_negative", 0);
    Expect(registry, std::move(node), "test_isinf_positive", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {6}, {-1.7f, nan_v, inf_v, 3.6f, -inf_v, inf_v});
      Tensor y = isinf_kernel.Invoke([&](const auto &kernel) {
        return kernel(x, /*detect_positive=*/1, /*detect_negative=*/0);
      });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ``IsInf.export_negative_infinity_only`` (detect_positive=0):
  //   x = [-1.7, NaN, +inf, -3.6, -inf, +inf] as float32
  //   y = np.isneginf(x)
  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    AddAttribute<int64_t>(node, "detect_positive", 0);
    Expect(registry, std::move(node), "test_isinf_negative", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {6}, {-1.7f, nan_v, inf_v, -3.6f, -inf_v, inf_v});
      Tensor y = isinf_kernel.Invoke([&](const auto &kernel) {
        return kernel(x, /*detect_positive=*/0, /*detect_negative=*/1);
      });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // ``test_isinf_float16`` — IsInf on FLOAT16 input with hardcoded expected.
  {
    NodeProto node = MakeNode("IsInf", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_isinf_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {6}, {-inf_v, -1.0f, 0.0f, 1.0f, inf_v, nan_v});
      // Expected: [True, False, False, False, True, False]
      Tensor y = Tensor::FromBool("", {6}, {1, 0, 0, 0, 1, 0});
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <tuple>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// DynamicQuantizeLinear — per-tensor dynamic linear quantization of a FLOAT
// input ``x`` to a UINT8 output ``y`` together with the derived scalar
// ``y_scale`` (FLOAT) and scalar ``y_zero_point`` (UINT8). Mirrors the
// ai.onnx ``DynamicQuantizeLinear`` operator (opset 11).
//
// Cases registered (matching upstream
// ``onnx.backend.test.case.node.dynamicquantizelinear`` where the reference
// ``kernel::DynamicQuantizeLinear`` supports the input shape):
//
//   * ``test_dynamicquantizelinear`` — 1-D input straddling zero.
//   * ``test_dynamicquantizelinear_max_adjusted`` — 1-D input with all
//     negative values (max gets adjusted to 0).
//   * ``test_dynamicquantizelinear_min_adjusted`` — 2-D input with all
//     positive values (min gets adjusted to 0).
//
// Upstream cases with the ``_expanded`` suffix exercise the function-body
// decomposition of the operator (Constant/ReduceMin/.../QuantizeLinear) and
// are not imported here — they are covered through the function-body path
// once each sub-operator gains backend test coverage.
// ---------------------------------------------------------------------------
void RegisterDynamicQuantizeLinearCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::DynamicQuantizeLinear dyn_quantize_kernel{ctx};

  NodeProto node;
  node.set_op_type("DynamicQuantizeLinear");
  node.add_input("x");
  node.add_output("y");
  node.add_output("y_scale");
  node.add_output("y_zero_point");

  // From DynamicQuantizeLinear.export(): 1-D input straddling zero.
  {
    Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, -3.0f, -2.5f, 1.34f, 0.5f});
    auto [y, y_scale, y_zero_point] = dyn_quantize_kernel(x);
    Expect(node, {x}, {y, y_scale, y_zero_point}, "test_dynamicquantizelinear", {opset},
           "backend-test", registry);
  }

  // From DynamicQuantizeLinear.export_max_adjusted(): all-negative 1-D input,
  // so ``max`` gets clipped to 0.
  {
    Tensor x = Tensor::FromFloat("", {6}, {-1.0f, -2.1f, -1.3f, -2.5f, -3.34f, -4.0f});
    auto [y, y_scale, y_zero_point] = dyn_quantize_kernel(x);
    Expect(node, {x}, {y, y_scale, y_zero_point}, "test_dynamicquantizelinear_max_adjusted",
           {opset}, "backend-test", registry);
  }

  // From DynamicQuantizeLinear.export_min_adjusted(): all-positive 2-D input,
  // so ``min`` gets clipped to 0.
  {
    Tensor x = Tensor::FromFloat(
        "", {3, 4}, {1.0f, 2.1f, 1.3f, 2.5f, 3.34f, 4.0f, 1.5f, 2.6f, 3.9f, 4.0f, 3.0f, 2.345f});
    auto [y, y_scale, y_zero_point] = dyn_quantize_kernel(x);
    Expect(node, {x}, {y, y_scale, y_zero_point}, "test_dynamicquantizelinear_min_adjusted",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

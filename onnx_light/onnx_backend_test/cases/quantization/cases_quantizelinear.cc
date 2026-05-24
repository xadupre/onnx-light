// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// QuantizeLinear — per-tensor linear quantization of a FLOAT input ``x`` to
// an 8-bit integer output ``y`` (since opset 13 in the ai.onnx domain). The
// canonical formula is ``y = saturate(round(x / y_scale) + y_zero_point)``
// using IEEE 754 round-half-to-even.
//
// Two cases are registered:
//
//   * ``test_cc_quantizelinear`` — default UINT8 output (no ``y_zero_point``
//     input, equivalent to a zero point of 0).
//   * ``test_cc_quantizelinear_int8`` — explicit INT8 ``y_zero_point``, so
//     the output element type is INT8.
// ---------------------------------------------------------------------------
void RegisterQuantizeLinearCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::QuantizeLinear quantize_kernel{kernel::KernelContext(opset)};

  // Default UINT8 output (y_zero_point omitted).
  {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
    Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
    Tensor y = quantize_kernel(x, y_scale);

    Expect(node, {x, y_scale}, {y}, "test_cc_quantizelinear", {opset}, "backend-test", registry);
  }

  // Explicit INT8 ``y_zero_point``: the output element type follows it.
  {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_input("y_zero_point");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
    Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor y_zero_point(
        "", static_cast<int32_t>(TensorProto::DataType::INT8), {},
        std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
    Tensor y = quantize_kernel(x, y_scale, y_zero_point);

    Expect(node, {x, y_scale, y_zero_point}, {y}, "test_cc_quantizelinear_int8", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

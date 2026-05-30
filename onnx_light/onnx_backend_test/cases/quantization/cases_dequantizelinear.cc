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
// DequantizeLinear — per-tensor linear dequantization of an 8-bit integer
// input ``x`` to a FLOAT output ``y`` (since opset 10 in the ai.onnx domain).
// The canonical formula is ``y = (x - x_zero_point) * x_scale``.
//
// Two cases are registered:
//
//   * ``test_cc_dequantizelinear`` — default zero point (``x_zero_point``
//     omitted, UINT8 input).
//   * ``test_cc_dequantizelinear_int8`` — explicit INT8 ``x_zero_point``.
// ---------------------------------------------------------------------------
void RegisterDequantizeLinearCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::DequantizeLinear dequantize_kernel{ctx};

  // Default UINT8 input, x_zero_point omitted.
  {
    NodeProto node;
    node.set_op_type("DequantizeLinear");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_output("y");

    Tensor x = Tensor::FromUint8("", {4}, {0, 3, 128, 255});
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    Tensor y = dequantize_kernel(x, x_scale);

    Expect(node, {x, x_scale}, {y}, "test_cc_dequantizelinear", {opset}, "backend-test", registry);
  }

  // Explicit INT8 x_zero_point.
  {
    NodeProto node;
    node.set_op_type("DequantizeLinear");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_input("x_zero_point");
    node.add_output("y");

    Tensor x = Tensor::FromInt8("", {4}, {-10, -9, 0, 127});
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor x_zero_point(
        "", static_cast<int32_t>(TensorProto::DataType::INT8), {},
        std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
    Tensor y = dequantize_kernel(x, x_scale, x_zero_point);

    Expect(node, {x, x_scale, x_zero_point}, {y}, "test_cc_dequantizelinear_int8", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

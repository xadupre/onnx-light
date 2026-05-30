// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

Tensor MakeScalarTensor(int32_t data_type, const std::vector<uint8_t> &bytes) {
  return Tensor("", data_type, /*shape=*/{}, bytes);
}

Tensor Uint16ZeroPoint(uint16_t value) {
  std::vector<uint8_t> bytes(sizeof(uint16_t));
  std::memcpy(bytes.data(), &value, sizeof(uint16_t));
  return MakeScalarTensor(static_cast<int32_t>(TensorProto::DataType::UINT16), bytes);
}

Tensor Int16ZeroPoint(int16_t value) {
  std::vector<uint8_t> bytes(sizeof(int16_t));
  std::memcpy(bytes.data(), &value, sizeof(int16_t));
  return MakeScalarTensor(static_cast<int32_t>(TensorProto::DataType::INT16), bytes);
}

} // namespace

// ---------------------------------------------------------------------------
// DequantizeLinear — per-tensor linear dequantization of an integer input
// ``x`` to a FLOAT output ``y`` (since opset 10 in the ai.onnx domain). The
// canonical formula is ``y = (x - x_zero_point) * x_scale``.
//
// Cases registered (matching upstream
// ``onnx.backend.test.case.node.dequantizelinear.DequantizeLinear`` where the
// reference ``kernel::DequantizeLinear`` supports the input dtype):
//
//   * ``test_cc_dequantizelinear`` — default zero point (``x_zero_point``
//     omitted, UINT8 input).
//   * ``test_cc_dequantizelinear_int8`` — explicit INT8 ``x_zero_point``.
//   * ``test_dequantizelinear`` — upstream UINT8 case with explicit
//     ``x_zero_point=128`` (``DequantizeLinear.export``).
//   * ``test_dequantizelinear_uint16`` — upstream UINT16 case with
//     ``x_zero_point=32767`` (``DequantizeLinear.export_uint16``).
//   * ``test_dequantizelinear_int16`` — upstream INT16 case with
//     ``x_zero_point=-1024`` (``DequantizeLinear.export_int16``).
//
// Upstream cases that exercise per-axis dequantization (``axis`` attribute),
// blocked dequantization (``block_size`` attribute), sub-byte
// (UINT4/INT4/UINT2/INT2/FLOAT4E2M1) or float8 (E4M3FN/E5M2) dtypes are not
// imported because the reference kernel/Tensor helpers do not support them
// yet.
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

  // Upstream ONNX backend test cases for the ``DequantizeLinear`` operator
  // (mirror the ``onnx.backend.test.case.node.dequantizelinear`` Python
  // module) for the dtypes supported by ``kernel::DequantizeLinear``.
  NodeProto node;
  node.set_op_type("DequantizeLinear");
  node.add_input("x");
  node.add_input("x_scale");
  node.add_input("x_zero_point");
  node.add_output("y");

  // From DequantizeLinear.export(): UINT8 with explicit zero_point=128.
  {
    Tensor x = Tensor::FromUint8("", {4}, {0, 3, 128, 255});
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor x_zero_point("", static_cast<int32_t>(TensorProto::DataType::UINT8), {},
                              std::vector<uint8_t>(1, static_cast<uint8_t>(128)));
    Tensor y = dequantize_kernel(x, x_scale, x_zero_point);
    Expect(node, {x, x_scale, x_zero_point}, {y}, "test_dequantizelinear", {opset}, "backend-test",
           registry);
  }

  // From DequantizeLinear.export_uint16(): UINT16 with zero_point=32767.
  {
    Tensor x = Tensor::FromUint16("", {4}, {30000, 31000, 32768, 33000});
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor x_zero_point = Uint16ZeroPoint(32767);
    Tensor y = dequantize_kernel(x, x_scale, x_zero_point);
    Expect(node, {x, x_scale, x_zero_point}, {y}, "test_dequantizelinear_uint16", {opset},
           "backend-test", registry);
  }

  // From DequantizeLinear.export_int16(): INT16 with zero_point=-1024.
  {
    Tensor x = Tensor::FromInt16("", {4}, {-300, -30, -1025, 1270});
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor x_zero_point = Int16ZeroPoint(-1024);
    Tensor y = dequantize_kernel(x, x_scale, x_zero_point);
    Expect(node, {x, x_scale, x_zero_point}, {y}, "test_dequantizelinear_int16", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

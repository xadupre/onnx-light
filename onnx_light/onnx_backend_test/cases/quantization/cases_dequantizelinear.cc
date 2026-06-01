// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/kernels/tensor/cast_float8.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

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
  return MakeScalarTensor(static_cast<int32_t>(DataType::UINT16), bytes);
}

Tensor Int16ZeroPoint(int16_t value) {
  std::vector<uint8_t> bytes(sizeof(int16_t));
  std::memcpy(bytes.data(), &value, sizeof(int16_t));
  return MakeScalarTensor(static_cast<int32_t>(DataType::INT16), bytes);
}

// Builds a 1-D float8 tensor from the float32 sample values in *values*.
// ``encode`` is the saturating ``FloatToFloat8*Bits`` encoder declared in
// ``cast_float8.h``. Mirrors the way upstream ``onnx.helper.make_tensor``
// stores float8 scalars (one raw byte per element).
Tensor MakeFloat8Tensor(DataType dtype, const std::vector<int64_t> &shape,
                        const std::vector<float> &values, std::uint8_t (*encode)(float) noexcept) {
  std::vector<uint8_t> bytes(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bytes[i] = encode(values[i]);
  }
  return Tensor("", static_cast<int32_t>(dtype), shape, std::move(bytes));
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
//   * ``test_dequantizelinear_e4m3fn`` — upstream FLOAT8E4M3FN case with no
//     ``x_zero_point`` (``DequantizeLinear.export_e4m3fn``).
//   * ``test_dequantizelinear_e5m2`` — upstream FLOAT8E5M2 case with no
//     ``x_zero_point`` (``DequantizeLinear.export_e5m2``).
//   * ``test_dequantizelinear_e4m3fn_zero_point`` — upstream FLOAT8E4M3FN
//     case with an explicit FLOAT8E4M3FN ``x_zero_point=0``
//     (``DequantizeLinear.export_e4m3fn_zero_point``).
//
// Upstream cases that exercise per-axis dequantization (``axis`` attribute
// with non-scalar scale), blocked dequantization (``block_size`` attribute),
// sub-byte (UINT4/INT4/UINT2/INT2/FLOAT4E2M1) dtypes, or a FLOAT16 output
// (``test_dequantizelinear_e4m3fn_float16``) are not imported because the
// reference kernel/Tensor helpers do not support them yet.
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
        "", static_cast<int32_t>(DataType::INT8), {},
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
    const Tensor x_zero_point("", static_cast<int32_t>(DataType::UINT8), {},
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

  // The float8 upstream cases set ``axis=0``. The scalar ``x_scale`` makes
  // that attribute a no-op (per-tensor dequantization), but the registered
  // node still carries it verbatim so the serialized model matches the
  // upstream backend test data.
  const std::vector<float> f8_values = {0.0f, 0.5f, 1.0f, 448.0f, -104.0f};
  const std::vector<int64_t> f8_shape = {5};

  // From DequantizeLinear.export_e4m3fn(): FLOAT8E4M3FN, no zero_point,
  // axis=0.
  {
    NodeProto e4m3fn_node;
    e4m3fn_node.set_op_type("DequantizeLinear");
    e4m3fn_node.add_input("x");
    e4m3fn_node.add_input("x_scale");
    e4m3fn_node.add_output("y");
    AddAttribute<int64_t>(e4m3fn_node, "axis", 0);

    Tensor x = MakeFloat8Tensor(DataType::FLOAT8E4M3FN, f8_shape, f8_values,
                                &kernel::FloatToFloat8E4M3FNBits);
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    Tensor y = dequantize_kernel(x, x_scale);
    Expect(e4m3fn_node, {x, x_scale}, {y}, "test_dequantizelinear_e4m3fn", {opset}, "backend-test",
           registry);
  }

  // From DequantizeLinear.export_e5m2(): FLOAT8E5M2, no zero_point, axis=0.
  {
    NodeProto e5m2_node;
    e5m2_node.set_op_type("DequantizeLinear");
    e5m2_node.add_input("x");
    e5m2_node.add_input("x_scale");
    e5m2_node.add_output("y");
    AddAttribute<int64_t>(e5m2_node, "axis", 0);

    const std::vector<float> e5m2_values = {0.0f, 0.5f, 1.0f, 49152.0f, -96.0f};
    Tensor x = MakeFloat8Tensor(DataType::FLOAT8E5M2, f8_shape, e5m2_values,
                                &kernel::FloatToFloat8E5M2Bits);
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    Tensor y = dequantize_kernel(x, x_scale);
    Expect(e5m2_node, {x, x_scale}, {y}, "test_dequantizelinear_e5m2", {opset}, "backend-test",
           registry);
  }

  // From DequantizeLinear.export_e4m3fn_zero_point(): FLOAT8E4M3FN with
  // an explicit FLOAT8E4M3FN ``zero_point=0`` (1-D scalar), axis=0.
  {
    NodeProto e4m3fn_zp_node;
    e4m3fn_zp_node.set_op_type("DequantizeLinear");
    e4m3fn_zp_node.add_input("x");
    e4m3fn_zp_node.add_input("x_scale");
    e4m3fn_zp_node.add_input("zero_point");
    e4m3fn_zp_node.add_output("y");
    AddAttribute<int64_t>(e4m3fn_zp_node, "axis", 0);

    Tensor x = MakeFloat8Tensor(DataType::FLOAT8E4M3FN, f8_shape, f8_values,
                                &kernel::FloatToFloat8E4M3FNBits);
    Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
    // Upstream uses ``make_tensor("zero_point", FLOAT8E4M3FN, [1], [0])``
    // (a 1-D one-element tensor) for the zero point.
    const Tensor zero_point("", static_cast<int32_t>(DataType::FLOAT8E4M3FN), {1},
                            std::vector<uint8_t>{kernel::FloatToFloat8E4M3FNBits(0.0f)});
    Tensor y = dequantize_kernel(x, x_scale, zero_point);
    Expect(e4m3fn_zp_node, {x, x_scale, zero_point}, {y}, "test_dequantizelinear_e4m3fn_zero_point",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_kernels/kernels/tensor/cast_float8.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
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

// Packs ``values`` (one element per ``int8_t`` entry, range checked by the
// caller) into a 4-bit-per-element little-endian buffer matching the ONNX
// sub-byte layout (low nibble first per byte, trailing slot zero-padded).
std::vector<uint8_t> Pack4Bit(const std::vector<int8_t> &values) {
  std::vector<uint8_t> bytes((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    bytes[i / 2] |= static_cast<uint8_t>(nibble << (4 * (i % 2)));
  }
  return bytes;
}

// Packs ``values`` (one element per ``int8_t`` entry) into a 2-bit-per-element
// little-endian buffer matching the ONNX sub-byte layout (lowest pair first
// per byte, trailing slots zero-padded).
std::vector<uint8_t> Pack2Bit(const std::vector<int8_t> &values) {
  std::vector<uint8_t> bytes((values.size() + 3) / 4, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t pair = static_cast<uint8_t>(values[i]) & 0x03;
    bytes[i / 4] |= static_cast<uint8_t>(pair << (2 * (i % 4)));
  }
  return bytes;
}

// Builds a sub-byte tensor with the supplied ``dtype`` and ``shape`` from a
// flattened list of element values. ``bits`` is 4 or 2.
Tensor MakeSubByteTensor(DataType dtype, const std::vector<int64_t> &shape,
                         const std::vector<int8_t> &values, int bits) {
  std::vector<uint8_t> bytes = (bits == 4) ? Pack4Bit(values) : Pack2Bit(values);
  return Tensor("", static_cast<int32_t>(dtype), shape, std::move(bytes));
}

// Encodes a single float into FLOAT4E2M1 (4-bit float with 2 exponent bits
// and 1 mantissa bit). The format supports the values
// ``{+/-0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}``; the upstream
// backend test only uses these exact values so a small lookup is sufficient
// (no rounding is required).
uint8_t FloatToFloat4E2M1Nibble(float v) {
  struct Entry {
    float value;
    uint8_t bits;
  };
  static const Entry kTable[] = {
      {0.0f, 0x0},  {0.5f, 0x1},  {1.0f, 0x2},  {1.5f, 0x3},  {2.0f, 0x4},  {3.0f, 0x5},
      {4.0f, 0x6},  {6.0f, 0x7},  {-0.0f, 0x8}, {-0.5f, 0x9}, {-1.0f, 0xA}, {-1.5f, 0xB},
      {-2.0f, 0xC}, {-3.0f, 0xD}, {-4.0f, 0xE}, {-6.0f, 0xF},
  };
  for (const auto &e : kTable) {
    if (e.value == v && std::signbit(e.value) == std::signbit(v)) {
      return e.bits;
    }
  }
  throw std::invalid_argument("FloatToFloat4E2M1Nibble: value not representable in FLOAT4E2M1.");
}

Tensor MakeFloat4E2M1Tensor(const std::vector<int64_t> &shape, const std::vector<float> &values) {
  std::vector<uint8_t> bytes((values.size() + 1) / 2, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = FloatToFloat4E2M1Nibble(values[i]);
    bytes[i / 2] |= static_cast<uint8_t>(nibble << (4 * (i % 2)));
  }
  return Tensor("", static_cast<int32_t>(DataType::FLOAT4E2M1), shape, std::move(bytes));
}

} // namespace

// ---------------------------------------------------------------------------
// QuantizeLinear — per-tensor linear quantization of a FLOAT input ``x`` to
// an integer output ``y`` (since opset 13 in the ai.onnx domain). The
// canonical formula is ``y = saturate(round(x / y_scale) + y_zero_point)``
// using IEEE 754 round-half-to-even.
//
// Cases registered:
//
//   * ``test_cc_quantizelinear`` — default UINT8 output (no ``y_zero_point``
//     input, equivalent to a zero point of 0).
//   * ``test_cc_quantizelinear_int8`` — explicit INT8 ``y_zero_point``, so
//     the output element type is INT8.
//   * ``test_quantizelinear_uint16`` — upstream UINT16 case with
//     ``y_zero_point=32767`` (``QuantizeLinear.export_uint16``).
//   * ``test_quantizelinear_int16`` — upstream INT16 case with
//     ``y_zero_point=-1024`` (``QuantizeLinear.export_int16``).
//   * ``test_quantizelinear`` — upstream UINT8 case with explicit
//     ``y_zero_point=128`` (``QuantizeLinear.export``).
//   * ``test_quantizelinear_axis`` — per-axis UINT8 case with ``axis=1`` and
//     a 3-element scale/zero point (``QuantizeLinear.export_axis``).
//   * ``test_quantizelinear_e4m3fn`` — FLOAT8E4M3FN output with a 1-D
//     single-element ``y_zero_point=0`` (``QuantizeLinear.export_e4m3fn``).
//   * ``test_quantizelinear_e5m2`` — FLOAT8E5M2 output with a 1-D
//     single-element ``y_zero_point=0`` (``QuantizeLinear.export_e5m2``).
//   * ``test_quantizelinear_uint4`` / ``..._int4`` — sub-byte UINT4/INT4
//     per-axis cases (``QuantizeLinear.export_uint4`` / ``.export_int4``).
//   * ``test_quantizelinear_uint2`` / ``..._int2`` — sub-byte UINT2/INT2
//     per-axis cases (``QuantizeLinear.export_uint2`` / ``.export_int2``).
//   * ``test_quantizelinear_float4e2m1`` — FLOAT4E2M1 per-axis case
//     (``QuantizeLinear.export_float4e2m1``).
//
// Upstream cases that exercise blocked quantization (``block_size`` attribute)
// — ``test_quantizelinear_blocked_symmetric`` and
// ``test_quantizelinear_blocked_asymmetric`` — are not imported because the
// reference Tensor helpers do not support the blocked layout yet.
// ---------------------------------------------------------------------------
void RegisterQuantizeLinearCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::QuantizeLinear quantize_kernel{ctx};

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
        "", static_cast<int32_t>(DataType::INT8), {},
        std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
    Tensor y = quantize_kernel(x, y_scale, y_zero_point);

    Expect(node, {x, y_scale, y_zero_point}, {y}, "test_cc_quantizelinear_int8", {opset},
           "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``QuantizeLinear`` operator
  // (mirror the ``onnx.backend.test.case.node.quantizelinear`` Python module)
  // for the dtypes supported by ``kernel::QuantizeLinear``.
  NodeProto node;
  node.set_op_type("QuantizeLinear");
  node.add_input("x");
  node.add_input("y_scale");
  node.add_input("y_zero_point");
  node.add_output("y");

  // From QuantizeLinear.export_uint16().
  {
    Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, 3.0f, 200000.0f});
    Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor y_zero_point = Uint16ZeroPoint(32767);
    Tensor y = quantize_kernel(x, y_scale, y_zero_point);
    Expect(node, {x, y_scale, y_zero_point}, {y}, "test_quantizelinear_uint16", {opset},
           "backend-test", registry);
  }

  // From QuantizeLinear.export_int16().
  {
    Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, 3.0f, -100000.0f});
    Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor y_zero_point = Int16ZeroPoint(-1024);
    Tensor y = quantize_kernel(x, y_scale, y_zero_point);
    Expect(node, {x, y_scale, y_zero_point}, {y}, "test_quantizelinear_int16", {opset},
           "backend-test", registry);
  }

  // From QuantizeLinear.export(): default UINT8 with explicit
  // ``y_zero_point=128``. Inputs/outputs mirror the upstream Python case
  // verbatim so cross-runtime comparisons exercise the canonical reference
  // values.
  {
    Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
    Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
    const Tensor y_zero_point("", static_cast<int32_t>(DataType::UINT8), {},
                              std::vector<uint8_t>(1, static_cast<uint8_t>(128)));
    Tensor y = quantize_kernel(x, y_scale, y_zero_point);
    Expect(node, {x, y_scale, y_zero_point}, {y}, "test_quantizelinear", {opset}, "backend-test",
           registry);
  }

  // From QuantizeLinear.export_axis(): per-axis UINT8 quantization with
  // ``axis=1`` and a 3-element ``y_scale``/``y_zero_point`` (one entry per
  // channel along axis 1). Output values are pre-computed offline because the
  // reference ``kernel::QuantizeLinear`` only supports the per-tensor scalar
  // form.
  {
    NodeProto axis_node;
    axis_node.set_op_type("QuantizeLinear");
    axis_node.add_input("x");
    axis_node.add_input("y_scale");
    axis_node.add_input("y_zero_point");
    axis_node.add_output("y");
    AddAttribute<int64_t>(axis_node, "axis", 1);

    Tensor x = Tensor::FromFloat("", {1, 3, 3, 2},
                                 {-162.0f, 10.0f, -100.0f, 232.0f, -20.0f, -50.0f, -76.0f, 0.0f,
                                  0.0f, 252.0f, 32.0f, -44.0f, 245.0f, -485.0f, -960.0f, -270.0f,
                                  -375.0f, -470.0f});
    Tensor y_scale = Tensor::FromFloat("", {3}, {2.0f, 4.0f, 5.0f});
    Tensor y_zero_point = Tensor::FromUint8("", {3}, {84, 24, 196});
    Tensor y = Tensor::FromUint8(
        "", {1, 3, 3, 2},
        {3, 89, 34, 200, 74, 59, 5, 24, 24, 87, 32, 13, 245, 99, 4, 142, 121, 102});
    Expect(axis_node, {x, y_scale, y_zero_point}, {y}, "test_quantizelinear_axis", {opset},
           "backend-test", registry);
  }

  // From QuantizeLinear.export_e4m3fn(): FLOAT8E4M3FN output with a 1-D
  // single-element ``y_zero_point`` of 0.
  {
    NodeProto e4m3fn_node;
    e4m3fn_node.set_op_type("QuantizeLinear");
    e4m3fn_node.add_input("x");
    e4m3fn_node.add_input("y_scale");
    e4m3fn_node.add_input("y_zero_point");
    e4m3fn_node.add_output("y");

    Tensor x = Tensor::FromFloat("", {5}, {0.0f, 1.0f, 2.0f, 100000.0f, 200.0f});
    Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
    Tensor y_zero_point =
        MakeFloat8Tensor(DataType::FLOAT8E4M3FN, {1}, {0.0f}, &kernel::FloatToFloat8E4M3FNBits);
    Tensor y = MakeFloat8Tensor(DataType::FLOAT8E4M3FN, {5}, {0.0f, 0.5f, 1.0f, 448.0f, 96.0f},
                                &kernel::FloatToFloat8E4M3FNBits);
    Expect(e4m3fn_node, {x, y_scale, y_zero_point}, {y}, "test_quantizelinear_e4m3fn", {opset},
           "backend-test", registry);
  }

  // From QuantizeLinear.export_e5m2(): FLOAT8E5M2 output with a 1-D
  // single-element ``y_zero_point`` of 0.
  {
    NodeProto e5m2_node;
    e5m2_node.set_op_type("QuantizeLinear");
    e5m2_node.add_input("x");
    e5m2_node.add_input("y_scale");
    e5m2_node.add_input("y_zero_point");
    e5m2_node.add_output("y");

    Tensor x = Tensor::FromFloat("", {5}, {0.0f, 1.0f, 2.0f, 100000.0f, 200.0f});
    Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
    Tensor y_zero_point =
        MakeFloat8Tensor(DataType::FLOAT8E5M2, {1}, {0.0f}, &kernel::FloatToFloat8E5M2Bits);
    Tensor y = MakeFloat8Tensor(DataType::FLOAT8E5M2, {5}, {0.0f, 0.5f, 1.0f, 49152.0f, 96.0f},
                                &kernel::FloatToFloat8E5M2Bits);
    Expect(e5m2_node, {x, y_scale, y_zero_point}, {y}, "test_quantizelinear_e5m2", {opset},
           "backend-test", registry);
  }

  // The remaining upstream cases (UINT4/INT4/UINT2/INT2/FLOAT4E2M1) all use
  // per-axis quantization (``axis=0``) and a 3-element scale/zero point with
  // pre-computed sub-byte expected outputs. The reference
  // ``kernel::QuantizeLinear`` does not support those dtypes nor the
  // per-axis form so the expected outputs are encoded here from the upstream
  // values in
  // ``onnx.backend.test.case.node.quantizelinear.QuantizeLinear``.
  NodeProto sub_byte_node;
  sub_byte_node.set_op_type("QuantizeLinear");
  sub_byte_node.add_input("x");
  sub_byte_node.add_input("y_scale");
  sub_byte_node.add_input("y_zero_point");
  sub_byte_node.add_output("y");
  AddAttribute<int64_t>(sub_byte_node, "axis", 0);

  const std::vector<int64_t> sub_byte_x_shape = {3, 4};
  Tensor sub_byte_scale = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});

  // From QuantizeLinear.export_uint4().
  {
    Tensor x = Tensor::FromFloat(
        "", sub_byte_x_shape,
        {0.0f, 2.5f, 4.8f, 8.6f, -30.0f, -20.0f, 6.0f, 9.0f, 12.0f, 15.0f, 16.0f, 40.0f});
    Tensor y_zero_point = MakeSubByteTensor(DataType::UINT4, {3}, {1, 1, 1}, /*bits=*/4);
    Tensor y = MakeSubByteTensor(DataType::UINT4, sub_byte_x_shape,
                                 {1, 2, 3, 5, 0, 0, 3, 4, 4, 5, 5, 11}, /*bits=*/4);
    Expect(sub_byte_node, {x, sub_byte_scale, y_zero_point}, {y}, "test_quantizelinear_uint4",
           {opset}, "backend-test", registry);
  }

  // From QuantizeLinear.export_int4().
  {
    Tensor x = Tensor::FromFloat(
        "", sub_byte_x_shape,
        {0.0f, 2.5f, 4.8f, 8.6f, -30.0f, -20.0f, 6.0f, 9.0f, 12.0f, 15.0f, 16.0f, 40.0f});
    Tensor y_zero_point = MakeSubByteTensor(DataType::INT4, {3}, {1, 1, 1}, /*bits=*/4);
    Tensor y = MakeSubByteTensor(DataType::INT4, sub_byte_x_shape,
                                 {1, 2, 3, 5, -8, -6, 3, 4, 4, 5, 5, 7}, /*bits=*/4);
    Expect(sub_byte_node, {x, sub_byte_scale, y_zero_point}, {y}, "test_quantizelinear_int4",
           {opset}, "backend-test", registry);
  }

  // From QuantizeLinear.export_uint2().
  {
    Tensor x = Tensor::FromFloat(
        "", sub_byte_x_shape,
        {0.0f, 2.5f, 4.8f, 8.6f, -2.0f, -1.0f, 1.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    Tensor y_zero_point = MakeSubByteTensor(DataType::UINT2, {3}, {0, 0, 0}, /*bits=*/2);
    Tensor y = MakeSubByteTensor(DataType::UINT2, sub_byte_x_shape,
                                 {0, 1, 2, 3, 0, 0, 0, 1, 1, 1, 2, 2}, /*bits=*/2);
    Expect(sub_byte_node, {x, sub_byte_scale, y_zero_point}, {y}, "test_quantizelinear_uint2",
           {opset}, "backend-test", registry);
  }

  // From QuantizeLinear.export_int2().
  {
    Tensor x = Tensor::FromFloat(
        "", sub_byte_x_shape,
        {0.0f, 2.5f, 4.8f, 8.6f, -4.0f, -3.0f, 1.0f, 2.0f, -0.0f, -2.5f, -4.8f, -8.6f});
    Tensor y_zero_point = MakeSubByteTensor(DataType::INT2, {3}, {0, 0, 0}, /*bits=*/2);
    Tensor y = MakeSubByteTensor(DataType::INT2, sub_byte_x_shape,
                                 {0, 1, 1, 1, -1, -1, 0, 1, 0, -1, -1, -2}, /*bits=*/2);
    Expect(sub_byte_node, {x, sub_byte_scale, y_zero_point}, {y}, "test_quantizelinear_int2",
           {opset}, "backend-test", registry);
  }

  // From QuantizeLinear.export_float4e2m1().
  {
    Tensor x = Tensor::FromFloat(
        "", sub_byte_x_shape,
        {0.0f, 2.5f, 4.8f, 8.6f, -30.0f, -20.0f, 6.0f, 9.0f, -0.0f, -2.5f, -4.8f, -8.6f});
    Tensor y_zero_point = MakeFloat4E2M1Tensor({3}, {0.0f, 0.0f, 0.0f});
    Tensor y = MakeFloat4E2M1Tensor(sub_byte_x_shape, {0.0f, 1.0f, 2.0f, 4.0f, -6.0f, -6.0f, 2.0f,
                                                       3.0f, 0.0f, -0.5f, -1.0f, -2.0f});
    Expect(sub_byte_node, {x, sub_byte_scale, y_zero_point}, {y}, "test_quantizelinear_float4e2m1",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

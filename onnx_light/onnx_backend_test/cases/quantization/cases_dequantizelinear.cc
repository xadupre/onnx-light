// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/kernels/tensor/cast_float8.h"
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

// Encodes an IEEE-754 binary32 value as an IEEE-754 binary16 bit pattern
// using round-to-nearest-even. Mirrors the helper used by ``cases_attention``;
// duplicated here to keep the case file self-contained.
uint16_t FloatToFloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t e = static_cast<int32_t>((u >> 23) & 0xffu) - 127 + 15;
  const uint32_t m32 = u & 0x7fffffu;
  if (e >= 0x1f) {
    // Inf / NaN or overflow to Inf.
    if (((u >> 23) & 0xffu) == 0xffu) {
      const uint16_t mant = m32 ? static_cast<uint16_t>((m32 >> 13) | 0x200u) : 0u;
      return static_cast<uint16_t>(sign | 0x7c00u | mant);
    }
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t m = (m32 | 0x800000u) >> static_cast<uint32_t>(1 - e);
    const uint32_t round_bit = m & 0x00001000u;
    const uint32_t sticky = m & 0x00000fffu;
    uint16_t h = static_cast<uint16_t>(sign | (m >> 13));
    if (round_bit && (sticky != 0 || (h & 1))) {
      h = static_cast<uint16_t>(h + 1);
    }
    return h;
  }
  const uint32_t low = m32 & 0x1fffu;
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m32 >> 13));
  if (low > 0x1000u || (low == 0x1000u && (h & 1u))) {
    h = static_cast<uint16_t>(h + 1);
  }
  return h;
}

// Builds a FLOAT16 tensor with the supplied ``shape`` from a flattened list
// of float32 sample values rounded via ``FloatToFloat16Bits``.
Tensor MakeFloat16Tensor(const std::vector<int64_t> &shape, const std::vector<float> &values) {
  std::vector<uint16_t> bits(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bits[i] = FloatToFloat16Bits(values[i]);
  }
  Tensor t = Tensor::FromUint16("", shape, bits);
  t.data_type = static_cast<int32_t>(DataType::FLOAT16);
  return t;
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
//   * ``test_dequantizelinear_axis`` — upstream per-axis UINT8 case with
//     ``axis=1`` and a 3-element scale/zero point
//     (``DequantizeLinear.export_axis``).
//   * ``test_dequantizelinear_blocked`` — upstream blocked case with
//     ``axis=1``, ``block_size=2`` and a 4-D scale/zero point
//     (``DequantizeLinear.export_blocked``).
//   * ``test_dequantizelinear_e4m3fn_float16`` — upstream FLOAT8E4M3FN
//     case with a FLOAT16 ``x_scale`` and FLOAT16 output
//     (``DequantizeLinear.export_e4m3fn_float16``).
//   * ``test_dequantizelinear_uint4`` / ``..._int4`` — sub-byte UINT4/INT4
//     per-axis cases (``DequantizeLinear.export_uint4`` /
//     ``.export_int4``).
//   * ``test_dequantizelinear_uint2`` / ``..._int2`` — sub-byte UINT2/INT2
//     per-axis cases (``DequantizeLinear.export_uint2`` /
//     ``.export_int2``).
//   * ``test_dequantizelinear_float4e2m1`` — FLOAT4E2M1 per-axis case
//     (``DequantizeLinear.export_float4e2m1``).
//
// Expected outputs for the per-axis, blocked, sub-byte and FLOAT16 cases are
// computed offline because the reference ``kernel::DequantizeLinear`` only
// supports the per-tensor scalar form with FLOAT output for byte-sized
// integer or float8 inputs.
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

  // From DequantizeLinear.export_axis(): per-axis UINT8 dequantization with
  // ``axis=1`` and a 3-element ``x_scale``/``x_zero_point`` (one entry per
  // channel along axis 1). Expected outputs are pre-computed because the
  // reference ``kernel::DequantizeLinear`` only supports the per-tensor scalar
  // form.
  {
    NodeProto axis_node;
    axis_node.set_op_type("DequantizeLinear");
    axis_node.add_input("x");
    axis_node.add_input("x_scale");
    axis_node.add_input("x_zero_point");
    axis_node.add_output("y");

    Tensor x = Tensor::FromUint8(
        "", {1, 3, 3, 2},
        {3, 89, 34, 200, 74, 59, 5, 24, 24, 87, 32, 13, 245, 99, 4, 142, 121, 102});
    Tensor x_scale = Tensor::FromFloat("", {3}, {2.0f, 4.0f, 5.0f});
    Tensor x_zero_point = Tensor::FromUint8("", {3}, {84, 24, 196});
    Tensor y = Tensor::FromFloat("", {1, 3, 3, 2},
                                 {-162.0f, 10.0f, -100.0f, 232.0f, -20.0f, -50.0f, -76.0f, 0.0f,
                                  0.0f, 252.0f, 32.0f, -44.0f, 245.0f, -485.0f, -960.0f, -270.0f,
                                  -375.0f, -470.0f});
    Expect(axis_node, {x, x_scale, x_zero_point}, {y}, "test_dequantizelinear_axis", {opset},
           "backend-test", registry);
  }

  // From DequantizeLinear.export_blocked(): blocked UINT8 dequantization
  // with ``axis=1`` and ``block_size=2``. Expected outputs are pre-computed
  // (mirrors the upstream ``np.repeat`` expansion).
  {
    NodeProto blocked_node;
    blocked_node.set_op_type("DequantizeLinear");
    blocked_node.add_input("x");
    blocked_node.add_input("x_scale");
    blocked_node.add_input("x_zero_point");
    blocked_node.add_output("y");
    AddAttribute<int64_t>(blocked_node, "axis", 1);
    AddAttribute<int64_t>(blocked_node, "block_size", 2);

    Tensor x =
        Tensor::FromUint8("", {1, 4, 3, 2}, {3, 89, 34, 200, 74, 59, 5,   24, 24, 87,  32,  13,
                                             5, 12, 12, 33,  65, 42, 245, 99, 4,  142, 121, 102});
    Tensor x_scale = Tensor::FromFloat(
        "", {1, 2, 3, 2}, {3.0f, 2.0f, 4.0f, 1.0f, 2.0f, 2.0f, 5.0f, 2.0f, 4.0f, 3.0f, 5.0f, 2.0f});
    Tensor x_zero_point =
        Tensor::FromUint8("", {1, 2, 3, 2}, {1, 0, 0, 1, 2, 20, 3, 2, 4, 3, 15, 2});
    Tensor y = Tensor::FromFloat("", {1, 4, 3, 2},
                                 {6.0f,   178.0f, 136.0f,  199.0f, 144.0f, 78.0f,  12.0f,  48.0f,
                                  96.0f,  86.0f,  60.0f,   -14.0f, 10.0f,  20.0f,  32.0f,  90.0f,
                                  250.0f, 80.0f,  1210.0f, 194.0f, 0.0f,   417.0f, 530.0f, 200.0f});
    Expect(blocked_node, {x, x_scale, x_zero_point}, {y}, "test_dequantizelinear_blocked", {opset},
           "backend-test", registry);
  }

  // From DequantizeLinear.export_e4m3fn_float16(): FLOAT8E4M3FN input with a
  // FLOAT16 ``x_scale`` and FLOAT16 output, axis=0.
  {
    NodeProto f16_node;
    f16_node.set_op_type("DequantizeLinear");
    f16_node.add_input("x");
    f16_node.add_input("x_scale");
    f16_node.add_output("y");
    AddAttribute<int64_t>(f16_node, "axis", 0);

    Tensor x = MakeFloat8Tensor(DataType::FLOAT8E4M3FN, f8_shape, f8_values,
                                &kernel::FloatToFloat8E4M3FNBits);
    Tensor x_scale = MakeFloat16Tensor({}, {2.0f});
    Tensor y = MakeFloat16Tensor(f8_shape, {0.0f, 1.0f, 2.0f, 896.0f, -208.0f});
    Expect(f16_node, {x, x_scale}, {y}, "test_dequantizelinear_e4m3fn_float16", {opset},
           "backend-test", registry);
  }

  // Sub-byte upstream cases (UINT4/INT4/UINT2/INT2/FLOAT4E2M1). All use
  // ``axis=0``, a scalar ``x_scale=2`` and a 1-element ``x_zero_point``.
  // Expected outputs are encoded from the upstream values in
  // ``onnx.backend.test.case.node.dequantizelinear``.
  NodeProto sub_byte_node;
  sub_byte_node.set_op_type("DequantizeLinear");
  sub_byte_node.add_input("x");
  sub_byte_node.add_input("x_scale");
  sub_byte_node.add_input("x_zero_point");
  sub_byte_node.add_output("y");
  AddAttribute<int64_t>(sub_byte_node, "axis", 0);

  Tensor sub_byte_scale = Tensor::FromFloat("", {}, {2.0f});

  // From DequantizeLinear.export_uint4().
  {
    Tensor x = MakeSubByteTensor(DataType::UINT4, {5}, {0, 1, 7, 10, 15}, /*bits=*/4);
    Tensor x_zero_point = MakeSubByteTensor(DataType::UINT4, {1}, {1}, /*bits=*/4);
    Tensor y = Tensor::FromFloat("", {5}, {-2.0f, 0.0f, 12.0f, 18.0f, 28.0f});
    Expect(sub_byte_node, {x, sub_byte_scale, x_zero_point}, {y}, "test_dequantizelinear_uint4",
           {opset}, "backend-test", registry);
  }

  // From DequantizeLinear.export_int4().
  {
    Tensor x = MakeSubByteTensor(DataType::INT4, {5}, {0, 1, 7, -4, -8}, /*bits=*/4);
    Tensor x_zero_point = MakeSubByteTensor(DataType::INT4, {1}, {1}, /*bits=*/4);
    Tensor y = Tensor::FromFloat("", {5}, {-2.0f, 0.0f, 12.0f, -10.0f, -18.0f});
    Expect(sub_byte_node, {x, sub_byte_scale, x_zero_point}, {y}, "test_dequantizelinear_int4",
           {opset}, "backend-test", registry);
  }

  // From DequantizeLinear.export_uint2().
  {
    Tensor x = MakeSubByteTensor(DataType::UINT2, {4}, {0, 1, 2, 3}, /*bits=*/2);
    Tensor x_zero_point = MakeSubByteTensor(DataType::UINT2, {1}, {1}, /*bits=*/2);
    Tensor y = Tensor::FromFloat("", {4}, {-2.0f, 0.0f, 2.0f, 4.0f});
    Expect(sub_byte_node, {x, sub_byte_scale, x_zero_point}, {y}, "test_dequantizelinear_uint2",
           {opset}, "backend-test", registry);
  }

  // From DequantizeLinear.export_int2().
  {
    Tensor x = MakeSubByteTensor(DataType::INT2, {4}, {0, 1, -1, -2}, /*bits=*/2);
    Tensor x_zero_point = MakeSubByteTensor(DataType::INT2, {1}, {1}, /*bits=*/2);
    Tensor y = Tensor::FromFloat("", {4}, {-2.0f, 0.0f, -4.0f, -6.0f});
    Expect(sub_byte_node, {x, sub_byte_scale, x_zero_point}, {y}, "test_dequantizelinear_int2",
           {opset}, "backend-test", registry);
  }

  // From DequantizeLinear.export_float4e2m1().
  {
    Tensor x = MakeFloat4E2M1Tensor({5}, {0.0f, 1.0f, -1.0f, 1.5f, -4.0f});
    Tensor x_zero_point = MakeFloat4E2M1Tensor({1}, {0.0f});
    Tensor y = Tensor::FromFloat("", {5}, {0.0f, 2.0f, -2.0f, 3.0f, -8.0f});
    Expect(sub_byte_node, {x, sub_byte_scale, x_zero_point}, {y},
           "test_dequantizelinear_float4e2m1", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

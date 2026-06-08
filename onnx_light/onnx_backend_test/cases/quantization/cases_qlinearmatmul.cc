// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeQLinearMatMulNode() {
  NodeProto node;
  node.set_op_type("QLinearMatMul");
  node.add_input("a");
  node.add_input("a_scale");
  node.add_input("a_zero_point");
  node.add_input("b");
  node.add_input("b_scale");
  node.add_input("b_zero_point");
  node.add_input("y_scale");
  node.add_input("y_zero_point");
  node.add_output("y");
  return node;
}

// Encodes an IEEE-754 binary32 value as an IEEE-754 binary16 bit pattern
// using round-to-nearest-even. Mirrors the helper used by
// ``cases_dequantizelinear`` / ``cases_attention``; duplicated here to keep
// this case file self-contained.
uint16_t FloatToFloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t e = static_cast<int32_t>((u >> 23) & 0xffu) - 127 + 15;
  const uint32_t m32 = u & 0x7fffffu;
  if (e >= 0x1f) {
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

// Builds a FLOAT16 scalar tensor from a float32 value.
Tensor MakeFloat16Scalar(const std::string &name, float value) {
  Tensor t = Tensor::FromUint16(name, {}, {FloatToFloat16Bits(value)});
  t.data_type = static_cast<int32_t>(DataType::FLOAT16);
  return t;
}

// Builds an INT8/UINT8 scalar tensor (used for zero points). ``dtype`` must be
// ``DataType::INT8`` or ``DataType::UINT8``; ``value`` is reinterpreted as the
// underlying signed/unsigned byte.
Tensor MakeQuantScalar(const std::string &name, DataType dtype, int32_t value) {
  const uint8_t byte = static_cast<uint8_t>(static_cast<int8_t>(value) & 0xff);
  return Tensor(name, static_cast<int32_t>(dtype), {}, std::vector<uint8_t>{byte});
}

// Builds an INT8/UINT8 tensor of the given shape from an ``int32_t`` buffer
// (each element is truncated to a single signed/unsigned byte).
Tensor MakeQuantTensor(const std::string &name, DataType dtype, const std::vector<int64_t> &shape,
                       const std::vector<int32_t> &values) {
  std::vector<uint8_t> bytes(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(static_cast<int8_t>(values[i]) & 0xff);
  }
  return Tensor(name, static_cast<int32_t>(dtype), shape, std::move(bytes));
}

} // namespace

// ---------------------------------------------------------------------------
// QLinearMatMul — per-tensor 8-bit quantized matrix multiplication. Implements
// ``y = saturate(round(((a - a_zp) * a_scale) * ((b - b_zp) * b_scale) /
// y_scale) + y_zp)``.
//
// Cases registered (mirroring upstream
// ``QLinearMatMul.export_int`` in ``onnx.backend.test.case.node.qlinearmatmul``):
//
//   * ``test_cc_qlinearmatmul_2D_{uint8,int8}_{float32,float16}`` — 2-D
//     quantized matrix multiplication with FLOAT / FLOAT16 scales.
//   * ``test_cc_qlinearmatmul_3D_{uint8,int8}_{float32,float16}`` — 3-D
//     batched quantized matrix multiplication.
//
// FLOAT32 expected outputs are computed by the reference
// ``kernel::QLinearMatMul``. The kernel only accepts FLOAT scales, so the
// FLOAT16 variants reuse the FLOAT-derived expected outputs except for the
// 3-D INT8 case where upstream encodes a one-ULP rounding difference
// (``117/120`` vs ``116/119`` in the first row of each batch); that variation
// is encoded explicitly so the cases match
// ``onnx.backend.test.case.node.qlinearmatmul`` byte-for-byte.
// ---------------------------------------------------------------------------
void RegisterQLinearMatMulCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::QLinearMatMul ql{ctx};

  // Upstream UINT8 raw values; for INT8 each element is shifted by -127.
  const std::vector<int32_t> a_2d_raw = {208, 236, 0, 238, 3, 214, 255, 29};
  const std::vector<int32_t> b_raw = {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247};

  auto shift_int8 = [&](const std::vector<int32_t> &src, bool is_int8) {
    std::vector<int32_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      out[i] = is_int8 ? src[i] - 127 : src[i];
    }
    return out;
  };

  auto register_variant = [&](DataType dtype, const std::string &dtype_suffix, bool is_int8) {
    const int32_t a_zp_val = is_int8 ? 113 - 127 : 113;
    const int32_t b_zp_val_2d = is_int8 ? 114 - 127 : 114;
    const int32_t y_zp_val = is_int8 ? 118 - 127 : 118;

    Tensor a_scale_f = Tensor::FromFloat("a_scale", {}, {0.0066f});
    Tensor b_scale_f = Tensor::FromFloat("b_scale", {}, {0.00705f});
    Tensor y_scale_f = Tensor::FromFloat("y_scale", {}, {0.0107f});
    Tensor a_scale_h = MakeFloat16Scalar("a_scale", 0.0066f);
    Tensor b_scale_h = MakeFloat16Scalar("b_scale", 0.00705f);
    Tensor y_scale_h = MakeFloat16Scalar("y_scale", 0.0107f);

    Tensor a_zp = MakeQuantScalar("a_zero_point", dtype, a_zp_val);
    Tensor b_zp_2d = MakeQuantScalar("b_zero_point", dtype, b_zp_val_2d);
    Tensor y_zp = MakeQuantScalar("y_zero_point", dtype, y_zp_val);

    // 2-D case.
    Tensor a_2d = MakeQuantTensor("a", dtype, {2, 4}, shift_int8(a_2d_raw, is_int8));
    Tensor b_2d = MakeQuantTensor("b", dtype, {4, 3}, shift_int8(b_raw, is_int8));
    Tensor y_2d = ql(a_2d, a_scale_f, a_zp, b_2d, b_scale_f, b_zp_2d, y_scale_f, y_zp);
    y_2d.name = "y";
    {
      NodeProto node = MakeQLinearMatMulNode();
      Expect(node, {a_2d, a_scale_f, a_zp, b_2d, b_scale_f, b_zp_2d, y_scale_f, y_zp}, {y_2d},
             "test_cc_qlinearmatmul_2D_" + dtype_suffix + "_float32", {opset}, "backend-test",
             registry);
    }
    {
      NodeProto node = MakeQLinearMatMulNode();
      Expect(node, {a_2d, a_scale_h, a_zp, b_2d, b_scale_h, b_zp_2d, y_scale_h, y_zp}, {y_2d},
             "test_cc_qlinearmatmul_2D_" + dtype_suffix + "_float16", {opset}, "backend-test",
             registry);
    }

    // 3-D case. Upstream sets ``b_zero_point = 114`` for both UINT8 and INT8
    // here (i.e. it does NOT apply the -127 shift to b_zero_point for INT8),
    // which we mirror exactly. The 3-D input/weight tensors stack the 2-D
    // values twice along a new leading batch dimension.
    std::vector<int32_t> a_3d_src;
    a_3d_src.insert(a_3d_src.end(), a_2d_raw.begin(), a_2d_raw.end());
    a_3d_src.insert(a_3d_src.end(), a_2d_raw.begin(), a_2d_raw.end());
    std::vector<int32_t> b_3d_src;
    b_3d_src.insert(b_3d_src.end(), b_raw.begin(), b_raw.end());
    b_3d_src.insert(b_3d_src.end(), b_raw.begin(), b_raw.end());

    Tensor a_3d = MakeQuantTensor("a", dtype, {2, 2, 4}, shift_int8(a_3d_src, is_int8));
    Tensor b_3d = MakeQuantTensor("b", dtype, {2, 4, 3}, shift_int8(b_3d_src, is_int8));
    Tensor b_zp_3d = MakeQuantScalar("b_zero_point", dtype, /*value=*/114);

    Tensor y_3d_f32 = ql(a_3d, a_scale_f, a_zp, b_3d, b_scale_f, b_zp_3d, y_scale_f, y_zp);
    y_3d_f32.name = "y";
    {
      NodeProto node = MakeQLinearMatMulNode();
      Expect(node, {a_3d, a_scale_f, a_zp, b_3d, b_scale_f, b_zp_3d, y_scale_f, y_zp}, {y_3d_f32},
             "test_cc_qlinearmatmul_3D_" + dtype_suffix + "_float32", {opset}, "backend-test",
             registry);
    }

    // FLOAT16 3-D variant: UINT8 matches the FLOAT32 output exactly; INT8
    // encodes upstream's one-ULP rounding difference (first row of each
    // batch: ``-86, 116, 119`` instead of ``-86, 117, 120``).
    Tensor y_3d_f16 = y_3d_f32;
    if (is_int8) {
      y_3d_f16 = MakeQuantTensor("y", dtype, {2, 2, 3},
                                 {-86, 116, 119, 115, 39, -121, -86, 116, 119, 115, 39, -121});
    }
    {
      NodeProto node = MakeQLinearMatMulNode();
      Expect(node, {a_3d, a_scale_h, a_zp, b_3d, b_scale_h, b_zp_3d, y_scale_h, y_zp}, {y_3d_f16},
             "test_cc_qlinearmatmul_3D_" + dtype_suffix + "_float16", {opset}, "backend-test",
             registry);
    }
  };

  register_variant(DataType::UINT8, "uint8", /*is_int8=*/false);
  register_variant(DataType::INT8, "int8", /*is_int8=*/true);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

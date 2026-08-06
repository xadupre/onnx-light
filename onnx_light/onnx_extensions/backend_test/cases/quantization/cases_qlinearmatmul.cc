// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_extensions/backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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

// The IEEE-754 binary16 encoder and FLOAT16 scalar builder are provided by
// ``onnx_core/runtime/cast_helper.h`` as ``FloatToFloat16Bits``
// and ``MakeFloat16Scalar``.

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
//   * ``test_cc_qlinearmatmul_{overflow,underflow}_{uint8,int8}`` — saturation
//     of values that exceed the output dtype range (mirrors upstream
//     ``QLinearMatMul.export_overflow`` / ``export_underflow``).
//
// FLOAT32 expected outputs are computed by the reference
// ``kernel::QLinearMatMul``. The kernel evaluates the combined scale in
// FLOAT32, and for these inputs the FLOAT16 scales round-trip to the same
// result, so the FLOAT16 variants reuse the FLOAT-derived expected outputs.
// This keeps the cases matching ``onnx.backend.test.case.node.qlinearmatmul``
// byte-for-byte.
// ---------------------------------------------------------------------------
void RegisterQLinearMatMulCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(10);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::QLinearMatMul ql{ctx};

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> a_shape{512, 512};
    const std::vector<int64_t> b_shape{512, 512};
    const int64_t count = 512 * 512;
    NodeProto node = MakeQLinearMatMulNode();
    Expect(registry, std::move(node), "test_cc_qlinearmatmul_2D_uint8_float32_benchmark", {opset},
           {count, 1, 1, count, 1, 1, 1, 1}, {count}, [ql, a_shape, b_shape]() -> IoData {
             Tensor a_2d = Tensor::FromUint8("a", a_shape, RandUint<uint8_t>(256, a_shape, 2531));
             Tensor b_2d = Tensor::FromUint8("b", b_shape, RandUint<uint8_t>(256, b_shape, 2532));
             Tensor a_scale_f = Tensor::FromFloat("a_scale", {}, {0.0066f});
             Tensor b_scale_f = Tensor::FromFloat("b_scale", {}, {0.00705f});
             Tensor y_scale_f = Tensor::FromFloat("y_scale", {}, {0.0107f});
             Tensor a_zp = MakeQuantScalar("a_zero_point", DataType::UINT8, 113);
             Tensor b_zp_2d = MakeQuantScalar("b_zero_point", DataType::UINT8, 114);
             Tensor y_zp = MakeQuantScalar("y_zero_point", DataType::UINT8, 118);
             Tensor y_2d = ql(a_2d, a_scale_f, a_zp, b_2d, b_scale_f, b_zp_2d, y_scale_f, y_zp);
             y_2d.name = "y";
             return IoData{{std::move(a_2d), std::move(a_scale_f), std::move(a_zp), std::move(b_2d),
                            std::move(b_scale_f), std::move(b_zp_2d), std::move(y_scale_f),
                            std::move(y_zp)},
                           {std::move(y_2d)}};
           });
    return;
  }

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
      Expect(registry, std::move(node), "test_cc_qlinearmatmul_2D_" + dtype_suffix + "_float32",
             {opset}, [=]() -> IoData {
               return IoData{{std::move(a_2d), std::move(a_scale_f), std::move(a_zp),
                              std::move(b_2d), std::move(b_scale_f), std::move(b_zp_2d),
                              std::move(y_scale_f), std::move(y_zp)},
                             {std::move(y_2d)}};
             });
    }
    {
      NodeProto node = MakeQLinearMatMulNode();
      Expect(registry, std::move(node), "test_cc_qlinearmatmul_2D_" + dtype_suffix + "_float16",
             {opset}, [=]() -> IoData {
               return IoData{{std::move(a_2d), std::move(a_scale_h), std::move(a_zp),
                              std::move(b_2d), std::move(b_scale_h), std::move(b_zp_2d),
                              std::move(y_scale_h), std::move(y_zp)},
                             {std::move(y_2d)}};
             });
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
      Expect(registry, std::move(node), "test_cc_qlinearmatmul_3D_" + dtype_suffix + "_float32",
             {opset}, [=]() -> IoData {
               return IoData{{std::move(a_3d), std::move(a_scale_f), std::move(a_zp),
                              std::move(b_3d), std::move(b_scale_f), std::move(b_zp_3d),
                              std::move(y_scale_f), std::move(y_zp)},
                             {std::move(y_3d_f32)}};
             });
    }

    // FLOAT16 3-D variant: the FLOAT16 scales round-trip to the same combined
    // scale as the FLOAT32 ones for these inputs, so the expected output is
    // identical to the FLOAT32 case for both UINT8 and INT8. This matches
    // ``onnx.backend.test.case.node.qlinearmatmul`` byte-for-byte (the INT8 3-D
    // output is ``[[-86, -128, -128], [115, 39, -121]]`` per batch).
    {
      NodeProto node = MakeQLinearMatMulNode();
      Expect(registry, std::move(node), "test_cc_qlinearmatmul_3D_" + dtype_suffix + "_float16",
             {opset}, [=]() -> IoData {
               return IoData{{std::move(a_3d), std::move(a_scale_h), std::move(a_zp),
                              std::move(b_3d), std::move(b_scale_h), std::move(b_zp_3d),
                              std::move(y_scale_h), std::move(y_zp)},
                             {std::move(y_3d_f32)}};
             });
    }
  };

  register_variant(DataType::UINT8, "uint8", /*is_int8=*/false);
  register_variant(DataType::INT8, "int8", /*is_int8=*/true);

  // -------------------------------------------------------------------------
  // Overflow / underflow saturation cases (mirroring upstream
  // ``QLinearMatMul.export_overflow`` / ``export_underflow`` in
  // ``onnx.backend.test.case.node.qlinearmatmul``). These exercise the
  // saturation step that clips the rounded result to the output dtype's
  // representable range before downcasting.
  //
  // For the UINT8 underflow case, upstream relies on numpy 1.x silently
  // wrapping ``np.array([[-100]], dtype=np.uint8)`` to ``156``, which under
  // numpy 2.x raises ``OverflowError``. The C++ kernel always interprets the
  // underlying byte as unsigned, so to genuinely exercise the "result clipped
  // to 0" path we use a non-zero ``a_zero_point`` that drives the unscaled
  // accumulator negative — equivalent in spirit to the upstream test.
  // -------------------------------------------------------------------------
  auto register_saturation_case = [&](const std::string &name, DataType dtype, int32_t a_byte,
                                      int32_t b_byte, int32_t a_zp_val, int32_t b_zp_val, float a_s,
                                      float b_s, float y_s, int32_t y_zp_val,
                                      int32_t expected_byte) {
    Tensor a_t = MakeQuantTensor("a", dtype, {1, 1}, {a_byte});
    Tensor b_t = MakeQuantTensor("b", dtype, {1, 1}, {b_byte});
    Tensor a_scale_t = Tensor::FromFloat("a_scale", {}, {a_s});
    Tensor b_scale_t = Tensor::FromFloat("b_scale", {}, {b_s});
    Tensor y_scale_t = Tensor::FromFloat("y_scale", {}, {y_s});
    Tensor a_zp_t = MakeQuantScalar("a_zero_point", dtype, a_zp_val);
    Tensor b_zp_t = MakeQuantScalar("b_zero_point", dtype, b_zp_val);
    Tensor y_zp_t = MakeQuantScalar("y_zero_point", dtype, y_zp_val);
    Tensor y_t = MakeQuantTensor("y", dtype, {1, 1}, {expected_byte});

    NodeProto node = MakeQLinearMatMulNode();
    Expect(registry, std::move(node), name, {opset}, [=]() -> IoData {
      return IoData{{std::move(a_t), std::move(a_scale_t), std::move(a_zp_t), std::move(b_t),
                     std::move(b_scale_t), std::move(b_zp_t), std::move(y_scale_t),
                     std::move(y_zp_t)},
                    {std::move(y_t)}};
    });
  };

  // uint8 overflow: 100 * 100 / 0.2 = 50000 → clipped to 255.
  register_saturation_case("test_cc_qlinearmatmul_overflow_uint8", DataType::UINT8, 100, 100, 0, 0,
                           1.0f, 1.0f, 0.2f, 0, 255);
  // int8 overflow: 100 * 100 / 0.5 = 20000 → clipped to 127.
  register_saturation_case("test_cc_qlinearmatmul_overflow_int8", DataType::INT8, 100, 100, 0, 0,
                           1.0f, 1.0f, 0.5f, 0, 127);
  // uint8 underflow: (0 - 100) * 100 = -10000 → clipped to 0.
  register_saturation_case("test_cc_qlinearmatmul_underflow_uint8", DataType::UINT8, 0, 100, 100, 0,
                           1.0f, 1.0f, 1.0f, 0, 0);
  // int8 underflow: -100 * 100 / 0.5 = -20000 → clipped to -128.
  register_saturation_case("test_cc_qlinearmatmul_underflow_int8", DataType::INT8, -100, 100, 0, 0,
                           1.0f, 1.0f, 0.5f, 0, -128);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

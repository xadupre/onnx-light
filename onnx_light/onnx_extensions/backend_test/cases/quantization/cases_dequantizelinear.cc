// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Tensor builders and bit packing helpers are provided by
// ``onnx_core/runtime/kernels/cast_helper.h`` (Uint16ZeroPoint,
// Int16ZeroPoint, MakeFloat8Tensor, onnx_kernels::kernel::Pack2Bit,
// kernel::Pack4Bit, MakeSubByteTensor,
// kernel::FloatToFloat4E2M1Nibble, MakeFloat4E2M1Tensor,
// FloatToFloat16Bits, MakeFloat16Tensor).

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
// Expected outputs for the blocked and FLOAT16 cases are
// computed offline. The reference ``kernel::DequantizeLinear`` supports
// per-tensor scalar and per-axis FLOAT scale with FLOAT output for byte-sized
// integer, float8, and sub-byte (INT4/UINT4/INT2/UINT2/FLOAT4E2M1) inputs.
// ---------------------------------------------------------------------------
void RegisterDequantizeLinearCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(25);
  // Opset 21 introduced the ``block_size`` attribute and the UINT4/INT4
  // sub-byte input types; opset 23 added FLOAT4E2M1; opset 25 added UINT2/INT2.
  const OpsetId opset_v21 = DefaultOpset(25);
  const OpsetId opset_v23 = DefaultOpset(25);
  const OpsetId opset_v25 = DefaultOpset(25);
  const OpsetId opset_v28 = DefaultOpset(28);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("DequantizeLinear");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_output("y");

    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_dequantizelinear_benchmark", {opset}, {count, 1},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(25);

             const KernelContext dequantize_kernel_ctx{opset};
             const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

             Tensor x =
                 Tensor::FromUint8("", {kBenchmarkElementwiseSize},
                                   RandUint<uint8_t>(256, {kBenchmarkElementwiseSize}, 2511));
             Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
             Tensor y = dequantize_kernel(x, x_scale);
             return IoData{{std::move(x), std::move(x_scale)}, {std::move(y)}};
           });
    return;
  }

  for (const auto dtype : {DataType::FLOAT6E2M3, DataType::FLOAT6E3M2}) {
    NodeProto float6_node;
    float6_node.set_op_type("DequantizeLinear");
    float6_node.add_input("x");
    float6_node.add_input("x_scale");
    float6_node.add_output("y");
    AddAttribute<int64_t>(float6_node, "output_dtype", static_cast<int64_t>(DataType::FLOAT));
    const std::string name = dtype == DataType::FLOAT6E2M3 ? "test_dequantizelinear_float6e2m3"
                                                           : "test_dequantizelinear_float6e3m2";
    Expect(registry, std::move(float6_node), name, {opset_v28}, [dtype]() -> IoData {
      const std::vector<uint8_t> packed = dtype == DataType::FLOAT6E2M3
                                              ? std::vector<uint8_t>{0, 24, 32, 223, 7}
                                              : std::vector<uint8_t>{0, 40, 48, 216, 7};
      Tensor x("", static_cast<int32_t>(dtype), {6}, packed);
      Tensor scale = Tensor::FromFloat("", {}, {1.0f});
      // The final input saturates to each format's maximum finite value.
      Tensor y =
          Tensor::FromFloat("", {6},
                            dtype == DataType::FLOAT6E2M3
                                ? std::vector<float>{0.0f, -0.0f, 0.125f, 1.0f, 7.5f, 7.5f}
                                : std::vector<float>{0.0f, -0.0f, 0.125f, 1.0f, 8.0f, 28.0f});
      return IoData{{std::move(x), std::move(scale)}, {std::move(y)}};
    });
  }

  // Default UINT8 input, x_zero_point omitted.
  {
    NodeProto node;
    node.set_op_type("DequantizeLinear");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_dequantizelinear", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext dequantize_kernel_ctx{opset};
      const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

      Tensor x = Tensor::FromUint8("", {4}, {0, 3, 128, 255});
      Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
      Tensor y = dequantize_kernel(x, x_scale);

      return IoData{{std::move(x), std::move(x_scale)}, {std::move(y)}};
    });
  }

  // Explicit INT8 x_zero_point.
  {
    NodeProto node;
    node.set_op_type("DequantizeLinear");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_input("x_zero_point");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_dequantizelinear_int8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext dequantize_kernel_ctx{opset};
      const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

      Tensor x = Tensor::FromInt8("", {4}, {-10, -9, 0, 127});
      Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor x_zero_point(
          "", static_cast<int32_t>(DataType::INT8), {},
          std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
      Tensor y = dequantize_kernel(x, x_scale, x_zero_point);

      return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point)}, {std::move(y)}};
    });
  }

  // Per-axis UINT8 dequantization with ``x_zero_point`` omitted (defaults to
  // 0). Exercises the per-axis code path through the no-zero-point overload.
  {
    NodeProto node;
    node.set_op_type("DequantizeLinear");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 1);
    Expect(registry, std::move(node), "test_cc_dequantizelinear_axis_no_zero_point", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(25);

             const KernelContext dequantize_kernel_ctx{opset};
             const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

             Tensor x = Tensor::FromUint8(
                 "", {1, 3, 3, 2},
                 {3, 89, 34, 200, 74, 59, 5, 24, 24, 87, 32, 13, 245, 99, 4, 142, 121, 102});
             Tensor x_scale = Tensor::FromFloat("", {3}, {2.0f, 4.0f, 5.0f});
             Tensor y = dequantize_kernel(x, x_scale, /*axis=*/1);

             return IoData{{std::move(x), std::move(x_scale)}, {std::move(y)}};
           });
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
    Expect(registry, node, "test_dequantizelinear", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext dequantize_kernel_ctx{opset};
      const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

      Tensor x = Tensor::FromUint8("", {4}, {0, 3, 128, 255});
      Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor x_zero_point("", static_cast<int32_t>(DataType::UINT8), {},
                                std::vector<uint8_t>(1, static_cast<uint8_t>(128)));
      Tensor y = dequantize_kernel(x, x_scale, x_zero_point);
      return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point)}, {std::move(y)}};
    });
  }

  // From DequantizeLinear.export_uint16(): UINT16 with zero_point=32767.
  {
    Expect(registry, node, "test_dequantizelinear_uint16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext dequantize_kernel_ctx{opset};
      const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

      Tensor x = Tensor::FromUint16("", {4}, {30000, 31000, 32768, 33000});
      Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor x_zero_point = Uint16ZeroPoint(32767);
      Tensor y = dequantize_kernel(x, x_scale, x_zero_point);
      return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point)}, {std::move(y)}};
    });
  }

  // From DequantizeLinear.export_int16(): INT16 with zero_point=-1024.
  {
    Expect(registry, node, "test_dequantizelinear_int16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext dequantize_kernel_ctx{opset};
      const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

      Tensor x = Tensor::FromInt16("", {4}, {-300, -30, -1025, 1270});
      Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor x_zero_point = Int16ZeroPoint(-1024);
      Tensor y = dequantize_kernel(x, x_scale, x_zero_point);
      return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(e4m3fn_node), "test_dequantizelinear_e4m3fn", {opset_v21},
           [f8_shape, f8_values]() -> IoData {
             const OpsetId opset = DefaultOpset(25);

             const KernelContext dequantize_kernel_ctx{opset};
             const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

             Tensor x = MakeFloat8Tensor(DataType::FLOAT8E4M3FN, f8_shape, f8_values,
                                         &FloatToFloat8E4M3FNBits);
             Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
             Tensor y = dequantize_kernel(x, x_scale);
             return IoData{{std::move(x), std::move(x_scale)}, {std::move(y)}};
           });
  }

  // From DequantizeLinear.export_e5m2(): FLOAT8E5M2, no zero_point, axis=0.
  {
    NodeProto e5m2_node;
    e5m2_node.set_op_type("DequantizeLinear");
    e5m2_node.add_input("x");
    e5m2_node.add_input("x_scale");
    e5m2_node.add_output("y");
    AddAttribute<int64_t>(e5m2_node, "axis", 0);
    Expect(registry, std::move(e5m2_node), "test_dequantizelinear_e5m2", {opset_v21},
           [f8_shape]() -> IoData {
             const OpsetId opset = DefaultOpset(25);

             const KernelContext dequantize_kernel_ctx{opset};
             const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

             const std::vector<float> e5m2_values = {0.0f, 0.5f, 1.0f, 49152.0f, -96.0f};
             Tensor x = MakeFloat8Tensor(DataType::FLOAT8E5M2, f8_shape, e5m2_values,
                                         &FloatToFloat8E5M2Bits);
             Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
             Tensor y = dequantize_kernel(x, x_scale);
             return IoData{{std::move(x), std::move(x_scale)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(e4m3fn_zp_node), "test_dequantizelinear_e4m3fn_zero_point",
           {opset_v21}, [f8_shape, f8_values]() -> IoData {
             const OpsetId opset = DefaultOpset(25);

             const KernelContext dequantize_kernel_ctx{opset};
             const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

             Tensor x = MakeFloat8Tensor(DataType::FLOAT8E4M3FN, f8_shape, f8_values,
                                         &FloatToFloat8E4M3FNBits);
             Tensor x_scale = Tensor::FromFloat("", {}, {2.0f});
             // Upstream uses ``make_tensor("zero_point", FLOAT8E4M3FN, [1], [0])``
             // (a 1-D one-element tensor) for the zero point.
             const Tensor zero_point("", static_cast<int32_t>(DataType::FLOAT8E4M3FN), {1},
                                     std::vector<uint8_t>{FloatToFloat8E4M3FNBits(0.0f)});
             Tensor y = dequantize_kernel(x, x_scale, zero_point);
             return IoData{{std::move(x), std::move(x_scale), std::move(zero_point)},
                           {std::move(y)}};
           });
  }

  // From DequantizeLinear.export_axis(): per-axis UINT8 dequantization with
  // ``axis=1`` and a 3-element ``x_scale``/``x_zero_point`` (one entry per
  // channel along axis 1).
  {
    NodeProto axis_node;
    axis_node.set_op_type("DequantizeLinear");
    axis_node.add_input("x");
    axis_node.add_input("x_scale");
    axis_node.add_input("x_zero_point");
    axis_node.add_output("y");
    Expect(registry, std::move(axis_node), "test_dequantizelinear_axis", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext dequantize_kernel_ctx{opset};
      const onnx_kernels::kernel::DequantizeLinear dequantize_kernel{dequantize_kernel_ctx};

      Tensor x = Tensor::FromUint8(
          "", {1, 3, 3, 2},
          {3, 89, 34, 200, 74, 59, 5, 24, 24, 87, 32, 13, 245, 99, 4, 142, 121, 102});
      Tensor x_scale = Tensor::FromFloat("", {3}, {2.0f, 4.0f, 5.0f});
      Tensor x_zero_point = Tensor::FromUint8("", {3}, {84, 24, 196});
      Tensor y = dequantize_kernel(x, x_scale, x_zero_point, /*axis=*/1);
      return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(blocked_node), "test_dequantizelinear_blocked", {opset_v21},
           []() -> IoData {
             Tensor x = Tensor::FromUint8("", {1, 4, 3, 2},
                                          {3, 89, 34, 200, 74, 59, 5,   24, 24, 87,  32,  13,
                                           5, 12, 12, 33,  65, 42, 245, 99, 4,  142, 121, 102});
             Tensor x_scale = Tensor::FromFloat(
                 "", {1, 2, 3, 2},
                 {3.0f, 2.0f, 4.0f, 1.0f, 2.0f, 2.0f, 5.0f, 2.0f, 4.0f, 3.0f, 5.0f, 2.0f});
             Tensor x_zero_point =
                 Tensor::FromUint8("", {1, 2, 3, 2}, {1, 0, 0, 1, 2, 20, 3, 2, 4, 3, 15, 2});
             Tensor y = Tensor::FromFloat("", {1, 4, 3, 2},
                                          {6.0f,    178.0f, 136.0f, 199.0f, 144.0f, 78.0f,
                                           12.0f,   48.0f,  96.0f,  86.0f,  60.0f,  -14.0f,
                                           10.0f,   20.0f,  32.0f,  90.0f,  250.0f, 80.0f,
                                           1210.0f, 194.0f, 0.0f,   417.0f, 530.0f, 200.0f});
             return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point)},
                           {std::move(y)}};
           });
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
    Expect(registry, std::move(f16_node), "test_dequantizelinear_e4m3fn_float16", {opset_v21},
           [f8_shape, f8_values]() -> IoData {
             Tensor x = MakeFloat8Tensor(DataType::FLOAT8E4M3FN, f8_shape, f8_values,
                                         &FloatToFloat8E4M3FNBits);
             Tensor x_scale = MakeFloat16Tensor("", {}, {2.0f});
             Tensor y = MakeFloat16Tensor("", f8_shape, {0.0f, 1.0f, 2.0f, 896.0f, -208.0f});
             return IoData{{std::move(x), std::move(x_scale)}, {std::move(y)}};
           });
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
    Expect(registry, sub_byte_node, "test_dequantizelinear_uint4", {opset_v21}, []() -> IoData {
      Tensor sub_byte_scale = Tensor::FromFloat("", {}, {2.0f});

      Tensor x = MakeSubByteTensor(DataType::UINT4, {5}, {0, 1, 7, 10, 15}, /*bits=*/4);
      Tensor x_zero_point = MakeSubByteTensor(DataType::UINT4, {1}, {1}, /*bits=*/4);
      Tensor y = Tensor::FromFloat("", {5}, {-2.0f, 0.0f, 12.0f, 18.0f, 28.0f});
      return IoData{{std::move(x), std::move(sub_byte_scale), std::move(x_zero_point)},
                    {std::move(y)}};
    });
  }

  // From DequantizeLinear.export_int4().
  {
    Expect(registry, sub_byte_node, "test_dequantizelinear_int4", {opset_v21}, []() -> IoData {
      Tensor sub_byte_scale = Tensor::FromFloat("", {}, {2.0f});

      Tensor x = MakeSubByteTensor(DataType::INT4, {5}, {0, 1, 7, -4, -8}, /*bits=*/4);
      Tensor x_zero_point = MakeSubByteTensor(DataType::INT4, {1}, {1}, /*bits=*/4);
      Tensor y = Tensor::FromFloat("", {5}, {-2.0f, 0.0f, 12.0f, -10.0f, -18.0f});
      return IoData{{std::move(x), std::move(sub_byte_scale), std::move(x_zero_point)},
                    {std::move(y)}};
    });
  }

  // From DequantizeLinear.export_uint2().
  {
    Expect(registry, sub_byte_node, "test_dequantizelinear_uint2", {opset_v25}, []() -> IoData {
      Tensor sub_byte_scale = Tensor::FromFloat("", {}, {2.0f});

      Tensor x = MakeSubByteTensor(DataType::UINT2, {4}, {0, 1, 2, 3}, /*bits=*/2);
      Tensor x_zero_point = MakeSubByteTensor(DataType::UINT2, {1}, {1}, /*bits=*/2);
      Tensor y = Tensor::FromFloat("", {4}, {-2.0f, 0.0f, 2.0f, 4.0f});
      return IoData{{std::move(x), std::move(sub_byte_scale), std::move(x_zero_point)},
                    {std::move(y)}};
    });
  }

  // From DequantizeLinear.export_int2().
  {
    Expect(registry, sub_byte_node, "test_dequantizelinear_int2", {opset_v25}, []() -> IoData {
      Tensor sub_byte_scale = Tensor::FromFloat("", {}, {2.0f});

      Tensor x = MakeSubByteTensor(DataType::INT2, {4}, {0, 1, -1, -2}, /*bits=*/2);
      Tensor x_zero_point = MakeSubByteTensor(DataType::INT2, {1}, {1}, /*bits=*/2);
      Tensor y = Tensor::FromFloat("", {4}, {-2.0f, 0.0f, -4.0f, -6.0f});
      return IoData{{std::move(x), std::move(sub_byte_scale), std::move(x_zero_point)},
                    {std::move(y)}};
    });
  }

  // From DequantizeLinear.export_float4e2m1().
  {
    Expect(registry, sub_byte_node, "test_dequantizelinear_float4e2m1", {opset_v23},
           []() -> IoData {
             Tensor sub_byte_scale = Tensor::FromFloat("", {}, {2.0f});

             Tensor x = MakeFloat4E2M1Tensor({5}, {0.0f, 1.0f, -1.0f, 1.5f, -4.0f});
             Tensor x_zero_point = MakeFloat4E2M1Tensor({1}, {0.0f});
             Tensor y = Tensor::FromFloat("", {5}, {0.0f, 2.0f, -2.0f, 3.0f, -8.0f});
             return IoData{{std::move(x), std::move(sub_byte_scale), std::move(x_zero_point)},
                           {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

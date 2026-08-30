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
// kernel::FloatToFloat4E2M1Nibble, MakeFloat4E2M1Tensor).

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
//   * ``test_quantizelinear_float4e2m1_signed_zero`` — FLOAT4E2M1 output with
//     no ``y_zero_point`` preserves signed zero (onnx#8230).
//
// Upstream cases that exercise blocked quantization (``block_size`` attribute)
// — ``test_quantizelinear_blocked_symmetric`` and
// ``test_quantizelinear_blocked_asymmetric`` — are not imported because the
// reference Tensor helpers do not support the blocked layout yet.
// ---------------------------------------------------------------------------
void RegisterQuantizeLinearCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(25);
  const OpsetId opset_v21 = DefaultOpset(25); // For FLOAT8, INT4, UINT4
  const OpsetId opset_v23 = DefaultOpset(25); // For FLOAT4E2M1
  const OpsetId opset_v25 = DefaultOpset(25); // For INT2, UINT2

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_output("y");

    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_quantizelinear_benchmark", {opset}, {count, 1},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(25);

             const KernelContext quantize_kernel_ctx{opset};
             const onnx_kernels::kernel::QuantizeLinear quantize_kernel{quantize_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 2501);
             Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
             Tensor y = quantize_kernel(x, y_scale);
             return IoData{{std::move(x), std::move(y_scale)}, {std::move(y)}};
           });
    return;
  }

  // Default UINT8 output (y_zero_point omitted).
  {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_quantizelinear", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext quantize_kernel_ctx{opset};
      const onnx_kernels::kernel::QuantizeLinear quantize_kernel{quantize_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
      Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
      Tensor y = quantize_kernel(x, y_scale);

      return IoData{{std::move(x), std::move(y_scale)}, {std::move(y)}};
    });
  }

  // Explicit INT8 ``y_zero_point``: the output element type follows it.
  {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_input("y_zero_point");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_quantizelinear_int8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext quantize_kernel_ctx{opset};
      const onnx_kernels::kernel::QuantizeLinear quantize_kernel{quantize_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
      Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor y_zero_point(
          "", static_cast<int32_t>(DataType::INT8), {},
          std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
      Tensor y = quantize_kernel(x, y_scale, y_zero_point);

      return IoData{{std::move(x), std::move(y_scale), std::move(y_zero_point)}, {std::move(y)}};
    });
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
    Expect(registry, node, "test_quantizelinear_uint16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext quantize_kernel_ctx{opset};
      const onnx_kernels::kernel::QuantizeLinear quantize_kernel{quantize_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, 3.0f, 200000.0f});
      Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor y_zero_point = Uint16ZeroPoint(32767);
      Tensor y = quantize_kernel(x, y_scale, y_zero_point);
      return IoData{{std::move(x), std::move(y_scale), std::move(y_zero_point)}, {std::move(y)}};
    });
  }

  // From QuantizeLinear.export_int16().
  {
    Expect(registry, node, "test_quantizelinear_int16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext quantize_kernel_ctx{opset};
      const onnx_kernels::kernel::QuantizeLinear quantize_kernel{quantize_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, 3.0f, -100000.0f});
      Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor y_zero_point = Int16ZeroPoint(-1024);
      Tensor y = quantize_kernel(x, y_scale, y_zero_point);
      return IoData{{std::move(x), std::move(y_scale), std::move(y_zero_point)}, {std::move(y)}};
    });
  }

  // From QuantizeLinear.export(): default UINT8 with explicit
  // ``y_zero_point=128``. Inputs/outputs mirror the upstream Python case
  // verbatim so cross-runtime comparisons exercise the canonical reference
  // values.
  {
    Expect(registry, node, "test_quantizelinear", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(25);

      const KernelContext quantize_kernel_ctx{opset};
      const onnx_kernels::kernel::QuantizeLinear quantize_kernel{quantize_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
      Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
      const Tensor y_zero_point("", static_cast<int32_t>(DataType::UINT8), {},
                                std::vector<uint8_t>(1, static_cast<uint8_t>(128)));
      Tensor y = quantize_kernel(x, y_scale, y_zero_point);
      return IoData{{std::move(x), std::move(y_scale), std::move(y_zero_point)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(axis_node), "test_quantizelinear_axis", {opset}, []() -> IoData {
      Tensor x = Tensor::FromFloat("", {1, 3, 3, 2},
                                   {-162.0f, 10.0f, -100.0f, 232.0f, -20.0f, -50.0f, -76.0f, 0.0f,
                                    0.0f, 252.0f, 32.0f, -44.0f, 245.0f, -485.0f, -960.0f, -270.0f,
                                    -375.0f, -470.0f});
      Tensor y_scale = Tensor::FromFloat("", {3}, {2.0f, 4.0f, 5.0f});
      Tensor y_zero_point = Tensor::FromUint8("", {3}, {84, 24, 196});
      Tensor y = Tensor::FromUint8(
          "", {1, 3, 3, 2},
          {3, 89, 34, 200, 74, 59, 5, 24, 24, 87, 32, 13, 245, 99, 4, 142, 121, 102});
      return IoData{{std::move(x), std::move(y_scale), std::move(y_zero_point)}, {std::move(y)}};
    });
  }

  // From QuantizeLinear.export_e4m3fn(): FLOAT8E4M3FN output with a 1-D
  // single-element ``y_zero_point`` of 0. FLOAT8 types were introduced in opset 21.
  {
    NodeProto e4m3fn_node;
    e4m3fn_node.set_op_type("QuantizeLinear");
    e4m3fn_node.add_input("x");
    e4m3fn_node.add_input("y_scale");
    e4m3fn_node.add_input("y_zero_point");
    e4m3fn_node.add_output("y");
    Expect(registry, std::move(e4m3fn_node), "test_quantizelinear_e4m3fn", {opset_v21},
           []() -> IoData {
             Tensor x = Tensor::FromFloat("", {5}, {0.0f, 1.0f, 2.0f, 100000.0f, 200.0f});
             Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
             Tensor y_zero_point =
                 MakeFloat8Tensor(DataType::FLOAT8E4M3FN, {1}, {0.0f}, &FloatToFloat8E4M3FNBits);
             Tensor y =
                 MakeFloat8Tensor(DataType::FLOAT8E4M3FN, {5}, {0.0f, 0.5f, 1.0f, 448.0f, 96.0f},
                                  &FloatToFloat8E4M3FNBits);
             return IoData{{std::move(x), std::move(y_scale), std::move(y_zero_point)},
                           {std::move(y)}};
           });
  }

  // From QuantizeLinear.export_e5m2(): FLOAT8E5M2 output with a 1-D
  // single-element ``y_zero_point`` of 0. FLOAT8 types were introduced in opset 21.
  {
    NodeProto e5m2_node;
    e5m2_node.set_op_type("QuantizeLinear");
    e5m2_node.add_input("x");
    e5m2_node.add_input("y_scale");
    e5m2_node.add_input("y_zero_point");
    e5m2_node.add_output("y");
    Expect(registry, std::move(e5m2_node), "test_quantizelinear_e5m2", {opset_v21}, []() -> IoData {
      Tensor x = Tensor::FromFloat("", {5}, {0.0f, 1.0f, 2.0f, 100000.0f, 200.0f});
      Tensor y_scale = Tensor::FromFloat("", {}, {2.0f});
      Tensor y_zero_point =
          MakeFloat8Tensor(DataType::FLOAT8E5M2, {1}, {0.0f}, &FloatToFloat8E5M2Bits);
      Tensor y = MakeFloat8Tensor(DataType::FLOAT8E5M2, {5}, {0.0f, 0.5f, 1.0f, 49152.0f, 96.0f},
                                  &FloatToFloat8E5M2Bits);
      return IoData{{std::move(x), std::move(y_scale), std::move(y_zero_point)}, {std::move(y)}};
    });
  }

  // The remaining upstream cases (UINT4/INT4/UINT2/INT2/FLOAT4E2M1) all use
  // per-axis quantization (``axis=0``) and a 3-element scale/zero point with
  // pre-computed sub-byte expected outputs.
  // INT4/UINT4 were introduced in opset 21, INT2/UINT2 in opset 25, FLOAT4E2M1 in opset 23.
  NodeProto sub_byte_node;
  sub_byte_node.set_op_type("QuantizeLinear");
  sub_byte_node.add_input("x");
  sub_byte_node.add_input("y_scale");
  sub_byte_node.add_input("y_zero_point");
  sub_byte_node.add_output("y");
  AddAttribute<int64_t>(sub_byte_node, "axis", 0);

  const std::vector<int64_t> sub_byte_x_shape = {3, 4};
  Tensor sub_byte_scale = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});

  // From QuantizeLinear.export_uint4(). INT4/UINT4 introduced in opset 21.
  {
    Expect(registry, sub_byte_node, "test_quantizelinear_uint4", {opset_v21},
           [sub_byte_x_shape]() -> IoData {
             Tensor sub_byte_scale = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});

             Tensor x = Tensor::FromFloat(
                 "", sub_byte_x_shape,
                 {0.0f, 2.5f, 4.8f, 8.6f, -30.0f, -20.0f, 6.0f, 9.0f, 12.0f, 15.0f, 16.0f, 40.0f});
             Tensor y_zero_point = MakeSubByteTensor(DataType::UINT4, {3}, {1, 1, 1}, /*bits=*/4);
             Tensor y = MakeSubByteTensor(DataType::UINT4, sub_byte_x_shape,
                                          {1, 2, 3, 5, 0, 0, 3, 4, 4, 5, 5, 11}, /*bits=*/4);
             return IoData{{std::move(x), std::move(sub_byte_scale), std::move(y_zero_point)},
                           {std::move(y)}};
           });
  }

  // From QuantizeLinear.export_int4(). INT4/UINT4 introduced in opset 21.
  {
    Expect(registry, sub_byte_node, "test_quantizelinear_int4", {opset_v21},
           [sub_byte_x_shape]() -> IoData {
             Tensor sub_byte_scale = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});

             Tensor x = Tensor::FromFloat(
                 "", sub_byte_x_shape,
                 {0.0f, 2.5f, 4.8f, 8.6f, -30.0f, -20.0f, 6.0f, 9.0f, 12.0f, 15.0f, 16.0f, 40.0f});
             Tensor y_zero_point = MakeSubByteTensor(DataType::INT4, {3}, {1, 1, 1}, /*bits=*/4);
             Tensor y = MakeSubByteTensor(DataType::INT4, sub_byte_x_shape,
                                          {1, 2, 3, 5, -8, -6, 3, 4, 4, 5, 5, 7}, /*bits=*/4);
             return IoData{{std::move(x), std::move(sub_byte_scale), std::move(y_zero_point)},
                           {std::move(y)}};
           });
  }

  // From QuantizeLinear.export_uint2(). INT2/UINT2 introduced in opset 25.
  {
    Expect(registry, sub_byte_node, "test_quantizelinear_uint2", {opset_v25},
           [sub_byte_x_shape]() -> IoData {
             Tensor sub_byte_scale = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});

             Tensor x = Tensor::FromFloat(
                 "", sub_byte_x_shape,
                 {0.0f, 2.5f, 4.8f, 8.6f, -2.0f, -1.0f, 1.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
             Tensor y_zero_point = MakeSubByteTensor(DataType::UINT2, {3}, {0, 0, 0}, /*bits=*/2);
             Tensor y = MakeSubByteTensor(DataType::UINT2, sub_byte_x_shape,
                                          {0, 1, 2, 3, 0, 0, 0, 1, 1, 1, 2, 2}, /*bits=*/2);
             return IoData{{std::move(x), std::move(sub_byte_scale), std::move(y_zero_point)},
                           {std::move(y)}};
           });
  }

  // From QuantizeLinear.export_int2(). INT2/UINT2 introduced in opset 25.
  {
    Expect(registry, sub_byte_node, "test_quantizelinear_int2", {opset_v25},
           [sub_byte_x_shape]() -> IoData {
             Tensor sub_byte_scale = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});

             Tensor x = Tensor::FromFloat(
                 "", sub_byte_x_shape,
                 {0.0f, 2.5f, 4.8f, 8.6f, -4.0f, -3.0f, 1.0f, 2.0f, -0.0f, -2.5f, -4.8f, -8.6f});
             Tensor y_zero_point = MakeSubByteTensor(DataType::INT2, {3}, {0, 0, 0}, /*bits=*/2);
             Tensor y = MakeSubByteTensor(DataType::INT2, sub_byte_x_shape,
                                          {0, 1, 1, 1, -1, -1, 0, 1, 0, -1, -1, -2}, /*bits=*/2);
             return IoData{{std::move(x), std::move(sub_byte_scale), std::move(y_zero_point)},
                           {std::move(y)}};
           });
  }

  // From QuantizeLinear.export_float4e2m1(). FLOAT4E2M1 introduced in opset 23.
  {
    Expect(registry, sub_byte_node, "test_quantizelinear_float4e2m1", {opset_v23},
           [sub_byte_x_shape]() -> IoData {
             Tensor sub_byte_scale = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});

             Tensor x = Tensor::FromFloat(
                 "", sub_byte_x_shape,
                 {0.0f, 2.5f, 4.8f, 8.6f, -30.0f, -20.0f, 6.0f, 9.0f, -0.0f, -2.5f, -4.8f, -8.6f});
             Tensor y_zero_point = MakeFloat4E2M1Tensor({3}, {0.0f, 0.0f, 0.0f});
             Tensor y =
                 MakeFloat4E2M1Tensor(sub_byte_x_shape, {0.0f, 1.0f, 2.0f, 4.0f, -6.0f, -6.0f, 2.0f,
                                                         3.0f, 0.0f, -0.5f, -1.0f, -2.0f});
             return IoData{{std::move(x), std::move(sub_byte_scale), std::move(y_zero_point)},
                           {std::move(y)}};
           });
  }

  // From QuantizeLinear signed-zero handling (onnx#8230). With no
  // ``y_zero_point`` input, FLOAT4E2M1 output must preserve signed zero:
  // ``-0.0`` maps to the negative-zero nibble (0x8) and ``0.0`` to 0x0.
  {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_output("y");
    AddAttribute<int64_t>(node, "output_dtype", static_cast<int64_t>(DataType::FLOAT4E2M1));
    Expect(registry, std::move(node), "test_quantizelinear_float4e2m1_signed_zero", {opset_v23},
           []() -> IoData {
             Tensor x = Tensor::FromFloat("", {2}, {-0.0f, 0.0f});
             Tensor y_scale = Tensor::FromFloat("", {}, {1.0f});
             Tensor y = MakeFloat4E2M1Tensor({2}, {-0.0f, 0.0f});
             return IoData{{std::move(x), std::move(y_scale)}, {std::move(y)}};
           });
  }

  // --- Blocked quantization (opset 21+) ---

  // ``test_quantizelinear_blocked_asymmetric``: blocked per-axis with ZP.
  // x=(3,4), y_scale=(3,2), y_zero_point=(3,2), axis=1, block_size=2.
  {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_input("y_zero_point");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 1);
    AddAttribute<int64_t>(node, "block_size", 2);
    Expect(registry, std::move(node), "test_quantizelinear_blocked_asymmetric", {opset_v21},
           []() -> IoData {
             Tensor x = Tensor::FromFloat(
                 "", {3, 4},
                 {6.0f, 12.0f, 50.0f, 5.0f, 1.0f, 8.0f, 4.0f, 5.0f, 0.0f, 20.0f, 10.0f, 4.0f});
             Tensor y_scale = Tensor::FromFloat("", {3, 2}, {1.5f, 2.5f, 3.0f, 4.9f, 5.1f, 6.9f});
             Tensor y_zp = Tensor::FromUint8("", {3, 2}, {0, 1, 1, 0, 2, 3});
             Tensor y = Tensor::FromUint8("", {3, 4}, {4, 8, 21, 3, 1, 4, 1, 1, 2, 6, 4, 4});
             return IoData{{std::move(x), std::move(y_scale), std::move(y_zp)}, {std::move(y)}};
           });
  }

  // ``test_quantizelinear_blocked_symmetric``: blocked per-axis without ZP.
  // x=(3,4), y_scale=(3,2), axis=1, block_size=2, output_dtype=INT16(5).
  {
    NodeProto node;
    node.set_op_type("QuantizeLinear");
    node.add_input("x");
    node.add_input("y_scale");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 1);
    AddAttribute<int64_t>(node, "block_size", 2);
    AddAttribute<int64_t>(node, "output_dtype", 5);
    Expect(registry, std::move(node), "test_quantizelinear_blocked_symmetric", {opset_v21},
           []() -> IoData {
             Tensor x = Tensor::FromFloat(
                 "", {3, 4},
                 {6.0f, -8.0f, -10.0f, 5.0f, 1.0f, 8.0f, 4.0f, 5.0f, 0.0f, 20.0f, 10.0f, 4.0f});
             Tensor y_scale = Tensor::FromFloat("", {3, 2}, {1.5f, 2.5f, 3.0f, 4.9f, 5.1f, 6.9f});
             Tensor y = Tensor::FromInt16("", {3, 4},
                                          {int16_t{4}, int16_t{-5}, int16_t{-4}, int16_t{2},
                                           int16_t{0}, int16_t{3}, int16_t{1}, int16_t{1},
                                           int16_t{0}, int16_t{4}, int16_t{1}, int16_t{1}});
             return IoData{{std::move(x), std::move(y_scale)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

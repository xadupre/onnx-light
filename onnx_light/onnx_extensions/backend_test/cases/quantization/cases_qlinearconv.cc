// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// QLinearConv — per-tensor (or per-output-channel ``w``-side) 8-bit quantized
// convolution. Implements
// ``y = saturate(round(((x - x_zp) * x_scale) * ((w - w_zp) * w_scale) /
// y_scale + bias * x_scale * w_scale / y_scale) + y_zp)``.
//
// Cases registered:
//
//   * ``test_cc_qlinearconv`` — 1x1 UINT8 convolution mirroring the upstream
//     ``test_qlinearconv`` backend test, with length-1 1-D ``w_scale``/
//     ``w_zero_point`` (per-channel for the single output channel).
//   * ``test_cc_qlinearconv_int8`` — self-contained 2x2 INT8 convolution
//     covering signed zero-points and a negative INT8 output.
// ---------------------------------------------------------------------------
void RegisterQLinearConvCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(10);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("QLinearConv");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_input("x_zero_point");
    node.add_input("w");
    node.add_input("w_scale");
    node.add_input("w_zero_point");
    node.add_input("y_scale");
    node.add_input("y_zero_point");
    node.add_output("y");

    const std::vector<int64_t> x_shape{1, 1, 1024, 1024};
    const int64_t count = 1024 * 1024;
    Expect(registry, std::move(node), "test_cc_qlinearconv_benchmark", {opset},
           {count, 1, 1, 1, 1, 1, 1, 1}, {count}, [x_shape]() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext qc_ctx{opset};
             const onnx_kernels::kernel::QLinearConv qc{qc_ctx};

             Tensor x = Tensor::FromUint8("x", x_shape, RandUint<uint8_t>(256, x_shape, 2541));
             Tensor x_scale = Tensor::FromFloat("x_scale", {}, {0.00369204697f});
             Tensor x_zero_point("x_zero_point", static_cast<int32_t>(DataType::UINT8), {},
                                 std::vector<uint8_t>{132});
             Tensor w = Tensor::FromUint8("w", {1, 1, 1, 1}, {0});
             Tensor w_scale = Tensor::FromFloat("w_scale", {1}, {0.00172794575f});
             Tensor w_zero_point("w_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                                 std::vector<uint8_t>{255});
             Tensor y_scale = Tensor::FromFloat("y_scale", {}, {0.00162681262f});
             Tensor y_zero_point("y_zero_point", static_cast<int32_t>(DataType::UINT8), {},
                                 std::vector<uint8_t>{123});
             Tensor B;

             onnx_kernels::kernel::QLinearConv::Attributes attrs;
             Tensor y = qc(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale,
                           y_zero_point, B, attrs);
             y.name = "y";
             return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point), std::move(w),
                            std::move(w_scale), std::move(w_zero_point), std::move(y_scale),
                            std::move(y_zero_point)},
                           {std::move(y)}};
           });
    return;
  }

  {
    // 1x1 conv over a 7x7 UINT8 plane, mirroring the shape and dtype layout
    // of the upstream ``test_qlinearconv`` backend test. Input values follow
    // a fixed ``i * 4`` pattern so the expected output is fully reproducible.
    std::vector<uint8_t> x_vals(49);
    for (size_t i = 0; i < x_vals.size(); ++i) {
      x_vals[i] = static_cast<uint8_t>(i * 4);
    }
    Tensor x = Tensor::FromUint8("x", {1, 1, 7, 7}, x_vals);
    Tensor x_scale = Tensor::FromFloat("x_scale", {}, {0.00369204697f});
    Tensor x_zero_point("x_zero_point", static_cast<int32_t>(DataType::UINT8), {},
                        std::vector<uint8_t>{132});
    Tensor w = Tensor::FromUint8("w", {1, 1, 1, 1}, {0});
    Tensor w_scale = Tensor::FromFloat("w_scale", {1}, {0.00172794575f});
    Tensor w_zero_point("w_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                        std::vector<uint8_t>{255});
    Tensor y_scale = Tensor::FromFloat("y_scale", {}, {0.00162681262f});
    Tensor y_zero_point("y_zero_point", static_cast<int32_t>(DataType::UINT8), {},
                        std::vector<uint8_t>{123});
    Tensor B;

    onnx_kernels::kernel::QLinearConv::Attributes attrs;
    NodeProto node;
    node.set_op_type("QLinearConv");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_input("x_zero_point");
    node.add_input("w");
    node.add_input("w_scale");
    node.add_input("w_zero_point");
    node.add_input("y_scale");
    node.add_input("y_zero_point");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_qlinearconv", {opset},
           [opset, x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B,
            attrs]() -> IoData {
             const KernelContext qc_ctx{opset};
             const onnx_kernels::kernel::QLinearConv qc{qc_ctx};

             Tensor y = qc(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale,
                           y_zero_point, B, attrs);
             y.name = "y";
             return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point), std::move(w),
                            std::move(w_scale), std::move(w_zero_point), std::move(y_scale),
                            std::move(y_zero_point)},
                           {std::move(y)}};
           });
  }

  {
    const auto to_byte = [](int32_t value) {
      return static_cast<uint8_t>(static_cast<int8_t>(value));
    };

    Tensor x("x", static_cast<int32_t>(DataType::INT8), {1, 1, 2, 2},
             std::vector<uint8_t>{to_byte(10), to_byte(20), to_byte(30), to_byte(40)});
    Tensor x_scale = Tensor::FromFloat("x_scale", {}, {0.1f});
    Tensor x_zero_point("x_zero_point", static_cast<int32_t>(DataType::INT8), {},
                        std::vector<uint8_t>{to_byte(5)});
    Tensor w("w", static_cast<int32_t>(DataType::INT8), {1, 1, 2, 2},
             std::vector<uint8_t>{to_byte(1), to_byte(1), to_byte(1), to_byte(1)});
    Tensor w_scale = Tensor::FromFloat("w_scale", {1}, {1.0f});
    Tensor w_zero_point("w_zero_point", static_cast<int32_t>(DataType::INT8), {1},
                        std::vector<uint8_t>{to_byte(0)});
    Tensor y_scale = Tensor::FromFloat("y_scale", {}, {1.0f});
    Tensor y_zero_point("y_zero_point", static_cast<int32_t>(DataType::INT8), {},
                        std::vector<uint8_t>{to_byte(-10)});
    Tensor y("y", static_cast<int32_t>(DataType::INT8), {1, 1, 1, 1},
             std::vector<uint8_t>{to_byte(-2)});

    NodeProto node;
    node.set_op_type("QLinearConv");
    node.add_input("x");
    node.add_input("x_scale");
    node.add_input("x_zero_point");
    node.add_input("w");
    node.add_input("w_scale");
    node.add_input("w_zero_point");
    node.add_input("y_scale");
    node.add_input("y_zero_point");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_qlinearconv_int8", {opset}, [to_byte]() -> IoData {
      Tensor x("x", static_cast<int32_t>(DataType::INT8), {1, 1, 2, 2},
               std::vector<uint8_t>{to_byte(10), to_byte(20), to_byte(30), to_byte(40)});
      Tensor x_scale = Tensor::FromFloat("x_scale", {}, {0.1f});
      Tensor x_zero_point("x_zero_point", static_cast<int32_t>(DataType::INT8), {},
                          std::vector<uint8_t>{to_byte(5)});
      Tensor w("w", static_cast<int32_t>(DataType::INT8), {1, 1, 2, 2},
               std::vector<uint8_t>{to_byte(1), to_byte(1), to_byte(1), to_byte(1)});
      Tensor w_scale = Tensor::FromFloat("w_scale", {1}, {1.0f});
      Tensor w_zero_point("w_zero_point", static_cast<int32_t>(DataType::INT8), {1},
                          std::vector<uint8_t>{to_byte(0)});
      Tensor y_scale = Tensor::FromFloat("y_scale", {}, {1.0f});
      Tensor y_zero_point("y_zero_point", static_cast<int32_t>(DataType::INT8), {},
                          std::vector<uint8_t>{to_byte(-10)});
      Tensor y("y", static_cast<int32_t>(DataType::INT8), {1, 1, 1, 1},
               std::vector<uint8_t>{to_byte(-2)});

      return IoData{{std::move(x), std::move(x_scale), std::move(x_zero_point), std::move(w),
                     std::move(w_scale), std::move(w_zero_point), std::move(y_scale),
                     std::move(y_zero_point)},
                    {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

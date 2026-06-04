// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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
// ---------------------------------------------------------------------------
void RegisterQLinearConvCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::QLinearConv qc{ctx};

  {
    // 1x1 conv over a 7x7 UINT8 plane. Inputs are constructed similarly to
    // the upstream ``QLinearConv`` backend test case but with deterministic
    // values, so the expected output is fully reproducible.
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

    kernel::QLinearConv::Attributes attrs;
    Tensor y =
        qc(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, attrs);
    y.name = "y";

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

    Expect(node, {x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point}, {y},
           "test_cc_qlinearconv", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

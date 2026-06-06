// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

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

} // namespace

// ---------------------------------------------------------------------------
// QLinearMatMul — per-tensor 8-bit quantized matrix multiplication. Implements
// ``y = saturate(round(((a - a_zp) * a_scale) * ((b - b_zp) * b_scale) /
// y_scale) + y_zp)``.
//
// Cases registered:
//
//   * ``test_cc_qlinearmatmul_2D`` — 2-D UINT8 matrix multiplication with
//     scalar scales and zero points.
//   * ``test_cc_qlinearmatmul_3D`` — 3-D batched INT8 matrix multiplication
//     with scalar scales and zero points.
// ---------------------------------------------------------------------------
void RegisterQLinearMatMulCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::QLinearMatMul ql{ctx};

  // 2-D UINT8 case.
  {
    Tensor a = Tensor::FromUint8("a", {2, 4}, {208, 236, 0, 238, 3, 214, 255, 29});
    Tensor a_scale = Tensor::FromFloat("a_scale", {}, {0.0066f});
    Tensor a_zero_point("a_zero_point", static_cast<int32_t>(DataType::UINT8), {},
                        std::vector<uint8_t>{113});
    Tensor b =
        Tensor::FromUint8("b", {4, 3}, {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247});
    Tensor b_scale = Tensor::FromFloat("b_scale", {}, {0.00705f});
    Tensor b_zero_point("b_zero_point", static_cast<int32_t>(DataType::UINT8), {},
                        std::vector<uint8_t>{114});
    Tensor y_scale = Tensor::FromFloat("y_scale", {}, {0.0107f});
    Tensor y_zero_point("y_zero_point", static_cast<int32_t>(DataType::UINT8), {},
                        std::vector<uint8_t>{118});

    Tensor y = ql(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point);
    y.name = "y";
    NodeProto node = MakeQLinearMatMulNode();
    Expect(node, {a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point}, {y},
           "test_cc_qlinearmatmul_2D", {opset}, "backend-test", registry);
  }

  // 3-D batched INT8 case.
  {
    Tensor a = Tensor::FromInt8(
        "a", {2, 2, 4}, {-10, 20, -30, 40, 50, -60, 70, -80, 5, 15, 25, 35, -45, -55, -65, -75});
    Tensor a_scale = Tensor::FromFloat("a_scale", {}, {0.01f});
    Tensor a_zero_point("a_zero_point", static_cast<int32_t>(DataType::INT8), {},
                        std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(-5))});
    Tensor b = Tensor::FromInt8("b", {2, 4, 3}, {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12,
                                                 0, 1,  2, 3,  4, 5,  6, 7,  8, 9,   10, 11});
    Tensor b_scale = Tensor::FromFloat("b_scale", {}, {0.02f});
    Tensor b_zero_point("b_zero_point", static_cast<int32_t>(DataType::INT8), {},
                        std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(2))});
    Tensor y_scale = Tensor::FromFloat("y_scale", {}, {0.05f});
    Tensor y_zero_point("y_zero_point", static_cast<int32_t>(DataType::INT8), {},
                        std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(0))});

    Tensor y = ql(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point);
    y.name = "y";
    NodeProto node = MakeQLinearMatMulNode();
    Expect(node, {a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point}, {y},
           "test_cc_qlinearmatmul_3D", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

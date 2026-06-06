// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeMatMulIntegerNode() {
  NodeProto node;
  node.set_op_type("MatMulInteger");
  node.add_input("A");
  node.add_input("B");
  node.add_input("a_zero_point");
  node.add_input("b_zero_point");
  node.add_output("Y");
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// MatMulInteger — 8-bit integer matrix multiplication with per-tensor zero
// points. Computes ``Y = matmul(A - a_zp, B - b_zp)`` with INT32 output. The
// case below reproduces the ONNX reference ``test_matmulinteger`` example.
// ---------------------------------------------------------------------------
void RegisterMatMulIntegerCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::MatMulInteger mmi{ctx};

  // 2-D UINT8 case, mirrors ``test_matmulinteger`` from onnx/backend.
  {
    Tensor A = Tensor::FromUint8("A", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
    Tensor B = Tensor::FromUint8("B", {3, 2}, {1, 4, 2, 5, 3, 6});
    Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                std::vector<uint8_t>{12});
    Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                std::vector<uint8_t>{0});

    Tensor Y = mmi(A, B, a_zp, b_zp);
    Y.name = "Y";
    NodeProto node = MakeMatMulIntegerNode();
    Expect(node, {A, B, a_zp, b_zp}, {Y}, "test_cc_matmulinteger", {opset}, "backend-test",
           registry);
  }

  // 2-D INT8 x INT8 case with scalar zero points.
  {
    Tensor A = Tensor::FromInt8("A", {2, 3}, {1, -2, 3, -4, 5, -6});
    Tensor B = Tensor::FromInt8("B", {3, 2}, {1, 2, -3, 4, 5, -6});
    Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::INT8), {},
                std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(1))});
    Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::INT8), {},
                std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(-1))});

    Tensor Y = mmi(A, B, a_zp, b_zp);
    Y.name = "Y";
    NodeProto node = MakeMatMulIntegerNode();
    Expect(node, {A, B, a_zp, b_zp}, {Y}, "test_cc_matmulinteger_int8", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

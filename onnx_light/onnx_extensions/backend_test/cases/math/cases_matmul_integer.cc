// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Node with all optional zero-point inputs present.
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

// Node with a_zero_point absent (empty string) and b_zero_point present.
NodeProto MakeMatMulIntegerNodeNoAZP() {
  NodeProto node;
  node.set_op_type("MatMulInteger");
  node.add_input("A");
  node.add_input("B");
  node.add_input(""); // a_zero_point absent
  node.add_input("b_zero_point");
  node.add_output("Y");
  return node;
}

// Node with a_zero_point present and b_zero_point absent (empty string).
NodeProto MakeMatMulIntegerNodeNoBZP() {
  NodeProto node;
  node.set_op_type("MatMulInteger");
  node.add_input("A");
  node.add_input("B");
  node.add_input("a_zero_point");
  node.add_input(""); // b_zero_point absent
  node.add_output("Y");
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// MatMulInteger — 8-bit integer matrix multiplication with per-tensor zero
// points. Computes ``Y = matmul(A - a_zp, B - b_zp)`` with INT32 output. The
// case below reproduces the ONNX reference ``test_matmulinteger`` example.
// ---------------------------------------------------------------------------
void RegisterMatMulIntegerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(10);

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> shape = {512, 512};
    NodeProto node = MakeMatMulIntegerNode();
    const int64_t count = 512 * 512;
    Expect(registry, std::move(node), "test_cc_matmulinteger_benchmark", {opset},
           {count, count, 1, 1}, {count}, [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext mmi_ctx{opset};
             const onnx_kernels::kernel::MatMulInteger mmi{mmi_ctx};

             Tensor A = Tensor::FromUint8("A", shape, RandUint<uint8_t>(255, shape, 437));
             Tensor B = Tensor::FromUint8("B", shape, RandUint<uint8_t>(255, shape, 438));
             Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                         std::vector<uint8_t>{0});
             Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                         std::vector<uint8_t>{0});
             Tensor Y = mmi(A, B, a_zp, b_zp);
             Y.name = "Y";
             return IoData{{std::move(A), std::move(B), std::move(a_zp), std::move(b_zp)},
                           {std::move(Y)}};
           });
    return;
  }

  // 2-D UINT8 case, mirrors ``test_matmulinteger`` from onnx/backend.
  {
    Tensor A = Tensor::FromUint8("A", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
    Tensor B = Tensor::FromUint8("B", {3, 2}, {1, 4, 2, 5, 3, 6});
    Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                std::vector<uint8_t>{12});
    Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                std::vector<uint8_t>{0});

    NodeProto node = MakeMatMulIntegerNode();
    Expect(registry, std::move(node), "test_cc_matmulinteger", {opset}, []() -> IoData {
      Tensor A = Tensor::FromUint8("A", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
      Tensor B = Tensor::FromUint8("B", {3, 2}, {1, 4, 2, 5, 3, 6});
      Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                  std::vector<uint8_t>{12});
      Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::UINT8), {1},
                  std::vector<uint8_t>{0});

      const OpsetId opset = DefaultOpset(10);

      const KernelContext mmi_ctx{opset};
      const onnx_kernels::kernel::MatMulInteger mmi{mmi_ctx};

      Tensor Y = mmi(A, B, a_zp, b_zp);
      Y.name = "Y";
      return IoData{{std::move(A), std::move(B), std::move(a_zp), std::move(b_zp)}, {std::move(Y)}};
    });
  }

  // 2-D INT8 x INT8 case with scalar zero points.
  {
    Tensor A = Tensor::FromInt8("A", {2, 3}, {1, -2, 3, -4, 5, -6});
    Tensor B = Tensor::FromInt8("B", {3, 2}, {1, 2, -3, 4, 5, -6});
    Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::INT8), {},
                std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(1))});
    Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::INT8), {},
                std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(-1))});

    NodeProto node = MakeMatMulIntegerNode();
    Expect(registry, std::move(node), "test_cc_matmulinteger_int8", {opset}, []() -> IoData {
      Tensor A = Tensor::FromInt8("A", {2, 3}, {1, -2, 3, -4, 5, -6});
      Tensor B = Tensor::FromInt8("B", {3, 2}, {1, 2, -3, 4, 5, -6});
      Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::INT8), {},
                  std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(1))});
      Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::INT8), {},
                  std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(-1))});

      const OpsetId opset = DefaultOpset(10);

      const KernelContext mmi_ctx{opset};
      const onnx_kernels::kernel::MatMulInteger mmi{mmi_ctx};

      Tensor Y = mmi(A, B, a_zp, b_zp);
      Y.name = "Y";
      return IoData{{std::move(A), std::move(B), std::move(a_zp), std::move(b_zp)}, {std::move(Y)}};
    });
  }

  // 2-D UINT8 case with per-column b_zero_point (a_zero_point absent).
  {
    Tensor A = Tensor::FromUint8("A", {2, 3}, {11, 7, 3, 10, 6, 2});
    Tensor B = Tensor::FromUint8("B", {3, 2}, {1, 4, 2, 5, 3, 6});
    Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::UINT8), {2},
                std::vector<uint8_t>{1, 2});

    // Default-constructed Tensor signals an absent optional input to the kernel.
    const Tensor no_a_zp;
    NodeProto node = MakeMatMulIntegerNodeNoAZP();
    Expect(registry, std::move(node), "test_cc_matmulinteger_per_col_b_zp", {opset},
           []() -> IoData {
             Tensor A = Tensor::FromUint8("A", {2, 3}, {11, 7, 3, 10, 6, 2});
             Tensor B = Tensor::FromUint8("B", {3, 2}, {1, 4, 2, 5, 3, 6});
             Tensor b_zp("b_zero_point", static_cast<int32_t>(DataType::UINT8), {2},
                         std::vector<uint8_t>{1, 2});
             const Tensor no_a_zp;

             const OpsetId opset = DefaultOpset(10);

             const KernelContext mmi_ctx{opset};
             const onnx_kernels::kernel::MatMulInteger mmi{mmi_ctx};

             Tensor Y = mmi(A, B, no_a_zp, b_zp);
             Y.name = "Y";
             return IoData{{std::move(A), std::move(B), std::move(b_zp)}, {std::move(Y)}};
           });
  }

  // 2-D UINT8 case with per-row a_zero_point (b_zero_point absent).
  {
    Tensor A = Tensor::FromUint8("A", {2, 3}, {11, 7, 3, 10, 6, 2});
    Tensor B = Tensor::FromUint8("B", {3, 2}, {1, 4, 2, 5, 3, 6});
    Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::UINT8), {2},
                std::vector<uint8_t>{1, 2});

    // Default-constructed Tensor signals an absent optional input to the kernel.
    const Tensor no_b_zp;
    NodeProto node = MakeMatMulIntegerNodeNoBZP();
    Expect(registry, std::move(node), "test_cc_matmulinteger_per_row_a_zp", {opset},
           []() -> IoData {
             Tensor A = Tensor::FromUint8("A", {2, 3}, {11, 7, 3, 10, 6, 2});
             Tensor B = Tensor::FromUint8("B", {3, 2}, {1, 4, 2, 5, 3, 6});
             Tensor a_zp("a_zero_point", static_cast<int32_t>(DataType::UINT8), {2},
                         std::vector<uint8_t>{1, 2});
             const Tensor no_b_zp;

             const OpsetId opset = DefaultOpset(10);

             const KernelContext mmi_ctx{opset};
             const onnx_kernels::kernel::MatMulInteger mmi{mmi_ctx};

             Tensor Y = mmi(A, B, a_zp, no_b_zp);
             Y.name = "Y";
             return IoData{{std::move(A), std::move(B), std::move(a_zp)}, {std::move(Y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

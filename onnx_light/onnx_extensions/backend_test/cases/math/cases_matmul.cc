// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterMatMulCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::MatMul matmul_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    const std::vector<int64_t> shape = {512, 512};
    const int64_t count = 512 * 512;
    Expect(registry, std::move(node), "test_cc_matmul_benchmark", {opset}, {count, count}, {count},
           [matmul_kernel, shape]() -> IoData {
             Tensor a = RandnTensor(DataType::FLOAT, shape, 435);
             Tensor b = RandnTensor(DataType::FLOAT, shape, 436);
             Tensor y = matmul_kernel(a, b);
             return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
           });
    return;
  }

  // 2-D x 2-D matrix product.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_2d", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {2, 3}, /*seed=*/21);
      Tensor b = RandnTensor(DataType::FLOAT, {3, 4}, /*seed=*/22);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // 1-D x 2-D: vector-matrix product.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_vector_matrix", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {3}, /*seed=*/23);
      Tensor b = RandnTensor(DataType::FLOAT, {3, 2}, /*seed=*/24);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // Batched MatMul with broadcast on leading dimensions.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_batch_broadcast", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {2, 2, 3}, /*seed=*/25);
      Tensor b = RandnTensor(DataType::FLOAT, {1, 3, 4}, /*seed=*/26);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // Upstream ``onnx.backend.test.case.node.matmul.MatMul`` registers six
  // additional scenarios with deterministic ``np.random.randn`` inputs. They
  // are mirrored here using the seeded ``Randn`` helper so that the substring
  // check in ``test_backend_test_names_onnx_vs_onnxlight`` finds a matching
  // ``onnx_light`` case for each upstream name.

  // 1-D x 1-D — dot product (output is a scalar).
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_1d_1d", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {3}, /*seed=*/27);
      Tensor b = RandnTensor(DataType::FLOAT, {3}, /*seed=*/28);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // 1-D x 3-D — prepends a 1 to A, broadcasts the leading batch dim, then
  // removes the prepended axis.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_1d_3d", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {4}, /*seed=*/29);
      Tensor b = RandnTensor(DataType::FLOAT, {2, 4, 3}, /*seed=*/30);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // 3-D x 3-D — batched matrix product.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_3d", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {2, 3, 4}, /*seed=*/31);
      Tensor b = RandnTensor(DataType::FLOAT, {2, 4, 3}, /*seed=*/32);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // 4-D x 4-D — batched matrix product over two leading batch dims.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_4d", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {1, 2, 3, 4}, /*seed=*/33);
      Tensor b = RandnTensor(DataType::FLOAT, {1, 2, 4, 3}, /*seed=*/34);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // 4-D x 1-D — appends a 1 to B, performs the batched matmul, then removes
  // the appended axis.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_4d_1d", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {2, 3, 4, 5}, /*seed=*/35);
      Tensor b = RandnTensor(DataType::FLOAT, {5}, /*seed=*/36);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }

  // Broadcast between a 3-D and a 2-D operand. The 2-D ``B`` is broadcast
  // across the leading batch dim of ``A``.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_matmul_bcast", {opset}, [=]() -> IoData {
      Tensor a = RandnTensor(DataType::FLOAT, {2, 3, 4}, /*seed=*/37);
      Tensor b = RandnTensor(DataType::FLOAT, {4, 5}, /*seed=*/38);
      Tensor y = matmul_kernel(a, b);
      return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

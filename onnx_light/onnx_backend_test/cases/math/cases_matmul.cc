// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterMatMulCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::MatMul matmul_kernel{ctx};

  // 2-D x 2-D matrix product.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {2, 3}, Randn<float>({2, 3}, /*seed=*/21));
    Tensor b = Tensor::FromFloat("", {3, 4}, Randn<float>({3, 4}, /*seed=*/22));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_2d", {opset}, "backend-test", registry);
  }

  // 1-D x 2-D: vector-matrix product.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {3}, Randn<float>({3}, /*seed=*/23));
    Tensor b = Tensor::FromFloat("", {3, 2}, Randn<float>({3, 2}, /*seed=*/24));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_vector_matrix", {opset}, "backend-test", registry);
  }

  // Batched MatMul with broadcast on leading dimensions.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {2, 2, 3}, Randn<float>({2, 2, 3}, /*seed=*/25));
    Tensor b = Tensor::FromFloat("", {1, 3, 4}, Randn<float>({1, 3, 4}, /*seed=*/26));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_batch_broadcast", {opset}, "backend-test", registry);
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

    Tensor a = Tensor::FromFloat("", {3}, Randn<float>({3}, /*seed=*/27));
    Tensor b = Tensor::FromFloat("", {3}, Randn<float>({3}, /*seed=*/28));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_1d_1d", {opset}, "backend-test", registry);
  }

  // 1-D x 3-D — prepends a 1 to A, broadcasts the leading batch dim, then
  // removes the prepended axis.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {4}, Randn<float>({4}, /*seed=*/29));
    Tensor b = Tensor::FromFloat("", {2, 4, 3}, Randn<float>({2, 4, 3}, /*seed=*/30));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_1d_3d", {opset}, "backend-test", registry);
  }

  // 3-D x 3-D — batched matrix product.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {2, 3, 4}, Randn<float>({2, 3, 4}, /*seed=*/31));
    Tensor b = Tensor::FromFloat("", {2, 4, 3}, Randn<float>({2, 4, 3}, /*seed=*/32));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_3d", {opset}, "backend-test", registry);
  }

  // 4-D x 4-D — batched matrix product over two leading batch dims.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {1, 2, 3, 4}, Randn<float>({1, 2, 3, 4}, /*seed=*/33));
    Tensor b = Tensor::FromFloat("", {1, 2, 4, 3}, Randn<float>({1, 2, 4, 3}, /*seed=*/34));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_4d", {opset}, "backend-test", registry);
  }

  // 4-D x 1-D — appends a 1 to B, performs the batched matmul, then removes
  // the appended axis.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {2, 3, 4, 5}, Randn<float>({2, 3, 4, 5}, /*seed=*/35));
    Tensor b = Tensor::FromFloat("", {5}, Randn<float>({5}, /*seed=*/36));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_4d_1d", {opset}, "backend-test", registry);
  }

  // Broadcast between a 3-D and a 2-D operand. The 2-D ``B`` is broadcast
  // across the leading batch dim of ``A``.
  {
    NodeProto node;
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");

    Tensor a = Tensor::FromFloat("", {2, 3, 4}, Randn<float>({2, 3, 4}, /*seed=*/37));
    Tensor b = Tensor::FromFloat("", {4, 5}, Randn<float>({4, 5}, /*seed=*/38));
    Tensor y = matmul_kernel(a, b);
    Expect(node, {a, b}, {y}, "test_cc_matmul_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

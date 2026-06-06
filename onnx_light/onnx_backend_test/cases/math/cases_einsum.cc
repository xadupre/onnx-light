// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Builds an Einsum NodeProto for ``n_inputs`` input names and the given
// ``equation`` attribute.
NodeProto MakeEinsumNode(int n_inputs, const std::string &equation) {
  NodeProto node;
  node.set_op_type("Einsum");
  for (int i = 0; i < n_inputs; ++i) {
    node.add_input("X" + std::to_string(i));
  }
  node.add_output("Y");

  AttributeProto *attr = node.add_attribute();
  attr->set_name("equation");
  attr->set_type(AttributeProto::STRING);
  attr->set_s(equation);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Einsum — Einstein summation (since opset 12).
// ---------------------------------------------------------------------------
void RegisterEinsumCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Einsum einsum_kernel{ctx};

  // Transpose: "ij->ji" (explicit).
  {
    const std::string eq = "ij->ji";
    NodeProto node = MakeEinsumNode(1, eq);
    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor z = einsum_kernel({x}, eq);
    Expect(node, {x}, {z}, "test_cc_einsum_transpose", {opset}, "backend-test", registry);
  }

  // Trace: "ii->" (explicit scalar).
  {
    const std::string eq = "ii->";
    NodeProto node = MakeEinsumNode(1, eq);
    Tensor x =
        Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor z = einsum_kernel({x}, eq);
    Expect(node, {x}, {z}, "test_cc_einsum_trace", {opset}, "backend-test", registry);
  }

  // Sum over an axis: "ij->i" (explicit).
  {
    const std::string eq = "ij->i";
    NodeProto node = MakeEinsumNode(1, eq);
    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor z = einsum_kernel({x}, eq);
    Expect(node, {x}, {z}, "test_cc_einsum_sum_axis", {opset}, "backend-test", registry);
  }

  // Matrix multiplication: "ij,jk->ik".
  {
    const std::string eq = "ij,jk->ik";
    NodeProto node = MakeEinsumNode(2, eq);
    Tensor a = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b = Tensor::FromFloat("", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor z = einsum_kernel({a, b}, eq);
    Expect(node, {a, b}, {z}, "test_cc_einsum_matmul_2d", {opset}, "backend-test", registry);
  }

  // Batched matrix multiplication: "bij,bjk->bik".
  {
    const std::string eq = "bij,bjk->bik";
    NodeProto node = MakeEinsumNode(2, eq);
    Tensor a = Tensor::FromFloat("", {2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor b = Tensor::FromFloat(
        "", {2, 3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f});
    Tensor z = einsum_kernel({a, b}, eq);
    Expect(node, {a, b}, {z}, "test_cc_einsum_batch_matmul", {opset}, "backend-test", registry);
  }

  // Inner product: "i,i->" (explicit scalar).
  {
    const std::string eq = "i,i->";
    NodeProto node = MakeEinsumNode(2, eq);
    Tensor a = Tensor::FromFloat("", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = Tensor::FromFloat("", {4}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor z = einsum_kernel({a, b}, eq);
    Expect(node, {a, b}, {z}, "test_cc_einsum_inner", {opset}, "backend-test", registry);
  }

  // Outer product: "i,j->ij" (explicit).
  {
    const std::string eq = "i,j->ij";
    NodeProto node = MakeEinsumNode(2, eq);
    Tensor a = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor b = Tensor::FromFloat("", {2}, {4.0f, 5.0f});
    Tensor z = einsum_kernel({a, b}, eq);
    Expect(node, {a, b}, {z}, "test_cc_einsum_outer", {opset}, "backend-test", registry);
  }

  // Implicit mode: "ij" — output keeps both labels (alphabetical order).
  {
    const std::string eq = "ij";
    NodeProto node = MakeEinsumNode(1, eq);
    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor z = einsum_kernel({x}, eq);
    Expect(node, {x}, {z}, "test_cc_einsum_implicit_identity", {opset}, "backend-test", registry);
  }

  // Ellipsis batch matmul: "...ij,...jk->...ik".
  {
    const std::string eq = "...ij,...jk->...ik";
    NodeProto node = MakeEinsumNode(2, eq);
    Tensor a = Tensor::FromFloat("", {2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor b = Tensor::FromFloat(
        "", {2, 3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f});
    Tensor z = einsum_kernel({a, b}, eq);
    Expect(node, {a, b}, {z}, "test_cc_einsum_ellipsis_matmul", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

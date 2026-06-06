// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

/// Builds a Gemm NodeProto with the given attributes.  Omitting a parameter
/// keeps the ONNX default (transA/transB = 0, alpha/beta = 1.0).
NodeProto MakeGemmNode(bool has_bias, float alpha = 1.0f, float beta = 1.0f, int64_t transA = 0,
                       int64_t transB = 0) {
  NodeProto node;
  node.set_op_type("Gemm");
  node.add_input("A");
  node.add_input("B");
  if (has_bias) {
    node.add_input("C");
  }
  node.add_output("Y");

  if (transA != 0) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("transA");
    attr->set_type(AttributeProto::INT);
    attr->set_i(transA);
  }
  if (transB != 0) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("transB");
    attr->set_type(AttributeProto::INT);
    attr->set_i(transB);
  }
  if (alpha != 1.0f) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("alpha");
    attr->set_type(AttributeProto::FLOAT);
    attr->set_f(alpha);
  }
  if (beta != 1.0f) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("beta");
    attr->set_type(AttributeProto::FLOAT);
    attr->set_f(beta);
  }

  return node;
}

} // namespace

void RegisterGemmCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Gemm gemm_kernel{ctx};

  // test_cc_gemm_default — Y = A * B, no bias, default attributes.
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/false);
    Tensor a = Tensor::FromFloat("", {3, 4}, Randn<float>({3, 4}, /*seed=*/1));
    Tensor b = Tensor::FromFloat("", {4, 3}, Randn<float>({4, 3}, /*seed=*/2));
    Tensor y = gemm_kernel(a, b, nullptr, 1.0f, 1.0f, 0, 0);
    Expect(node, {a, b}, {y}, "test_cc_gemm_default", {opset}, "backend-test", registry);
  }

  // test_cc_gemm_default_matrix_bias — Y = A * B + C (2-D bias).
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/true);
    Tensor a = Tensor::FromFloat("", {3, 4}, Randn<float>({3, 4}, /*seed=*/3));
    Tensor b = Tensor::FromFloat("", {4, 3}, Randn<float>({4, 3}, /*seed=*/4));
    Tensor c = Tensor::FromFloat("", {3, 3}, Randn<float>({3, 3}, /*seed=*/5));
    Tensor y = gemm_kernel(a, b, &c, 1.0f, 1.0f, 0, 0);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_default_matrix_bias", {opset}, "backend-test",
           registry);
  }

  // test_cc_gemm_default_vector_bias — Y = A * B + C (1-D broadcast bias).
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/true);
    Tensor a = Tensor::FromFloat("", {2, 7}, Randn<float>({2, 7}, /*seed=*/6));
    Tensor b = Tensor::FromFloat("", {7, 4}, Randn<float>({7, 4}, /*seed=*/7));
    Tensor c = Tensor::FromFloat("", {4}, Randn<float>({4}, /*seed=*/8));
    Tensor y = gemm_kernel(a, b, &c, 1.0f, 1.0f, 0, 0);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_default_vector_bias", {opset}, "backend-test",
           registry);
  }

  // test_cc_gemm_transposeA — Y = A' * B (transA=1).
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/false, 1.0f, 1.0f, /*transA=*/1);
    Tensor a = Tensor::FromFloat("", {4, 3}, Randn<float>({4, 3}, /*seed=*/9));
    Tensor b = Tensor::FromFloat("", {4, 5}, Randn<float>({4, 5}, /*seed=*/10));
    Tensor y = gemm_kernel(a, b, nullptr, 1.0f, 1.0f, 1, 0);
    Expect(node, {a, b}, {y}, "test_cc_gemm_transposeA", {opset}, "backend-test", registry);
  }

  // test_cc_gemm_transposeB — Y = A * B' (transB=1).
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/false, 1.0f, 1.0f, /*transA=*/0, /*transB=*/1);
    Tensor a = Tensor::FromFloat("", {3, 5}, Randn<float>({3, 5}, /*seed=*/11));
    Tensor b = Tensor::FromFloat("", {4, 5}, Randn<float>({4, 5}, /*seed=*/12));
    Tensor y = gemm_kernel(a, b, nullptr, 1.0f, 1.0f, 0, 1);
    Expect(node, {a, b}, {y}, "test_cc_gemm_transposeB", {opset}, "backend-test", registry);
  }

  // test_cc_gemm_all_attributes — Y = alpha * A' * B' + beta * C.
  {
    const float alpha = 0.5f;
    const float beta = 2.0f;
    NodeProto node = MakeGemmNode(/*has_bias=*/true, alpha, beta, /*transA=*/1, /*transB=*/1);
    Tensor a = Tensor::FromFloat("", {5, 3}, Randn<float>({5, 3}, /*seed=*/13));
    Tensor b = Tensor::FromFloat("", {4, 5}, Randn<float>({4, 5}, /*seed=*/14));
    Tensor c = Tensor::FromFloat("", {3, 4}, Randn<float>({3, 4}, /*seed=*/15));
    Tensor y = gemm_kernel(a, b, &c, alpha, beta, 1, 1);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_all_attributes", {opset}, "backend-test", registry);
  }

  // test_cc_gemm_default_zero_bias — Y = A * B + 0 (zero matrix bias).
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/true);
    Tensor a = Tensor::FromFloat("", {3, 5}, Randn<float>({3, 5}, /*seed=*/16));
    Tensor b = Tensor::FromFloat("", {5, 4}, Randn<float>({5, 4}, /*seed=*/17));
    Tensor c = Tensor::FromFloat("", {1, 4}, std::vector<float>(4, 0.0f));
    Tensor y = gemm_kernel(a, b, &c, 1.0f, 1.0f, 0, 0);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_default_zero_bias", {opset}, "backend-test",
           registry);
  }

  // test_cc_gemm_default_scalar_bias — Y = A * B + C (0-D scalar bias).
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/true);
    Tensor a = Tensor::FromFloat("", {2, 3}, Randn<float>({2, 3}, /*seed=*/18));
    Tensor b = Tensor::FromFloat("", {3, 4}, Randn<float>({3, 4}, /*seed=*/19));
    Tensor c = Tensor::FromFloat("", {}, {3.14f});
    Tensor y = gemm_kernel(a, b, &c, 1.0f, 1.0f, 0, 0);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_default_scalar_bias", {opset}, "backend-test",
           registry);
  }

  // test_cc_gemm_default_single_elem_vector_bias — Y = A * B + C (1-element 1-D bias).
  {
    NodeProto node = MakeGemmNode(/*has_bias=*/true);
    Tensor a = Tensor::FromFloat("", {3, 7}, Randn<float>({3, 7}, /*seed=*/20));
    Tensor b = Tensor::FromFloat("", {7, 3}, Randn<float>({7, 3}, /*seed=*/21));
    Tensor c = Tensor::FromFloat("", {1}, Randn<float>({1}, /*seed=*/22));
    Tensor y = gemm_kernel(a, b, &c, 1.0f, 1.0f, 0, 0);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_default_single_elem_vector_bias", {opset},
           "backend-test", registry);
  }

  // test_cc_gemm_alpha — Y = alpha * A * B + C (alpha != 1).
  {
    const float alpha = 0.5f;
    NodeProto node = MakeGemmNode(/*has_bias=*/true, alpha);
    Tensor a = Tensor::FromFloat("", {3, 5}, Randn<float>({3, 5}, /*seed=*/23));
    Tensor b = Tensor::FromFloat("", {5, 4}, Randn<float>({5, 4}, /*seed=*/24));
    Tensor c = Tensor::FromFloat("", {1, 4}, std::vector<float>(4, 0.0f));
    Tensor y = gemm_kernel(a, b, &c, alpha, 1.0f, 0, 0);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_alpha", {opset}, "backend-test", registry);
  }

  // test_cc_gemm_beta — Y = A * B + beta * C (beta != 1).
  {
    const float beta = 0.5f;
    NodeProto node = MakeGemmNode(/*has_bias=*/true, /*alpha=*/1.0f, beta);
    Tensor a = Tensor::FromFloat("", {2, 7}, Randn<float>({2, 7}, /*seed=*/25));
    Tensor b = Tensor::FromFloat("", {7, 4}, Randn<float>({7, 4}, /*seed=*/26));
    Tensor c = Tensor::FromFloat("", {1, 4}, Randn<float>({1, 4}, /*seed=*/27));
    Tensor y = gemm_kernel(a, b, &c, 1.0f, beta, 0, 0);
    Expect(node, {a, b, c}, {y}, "test_cc_gemm_beta", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

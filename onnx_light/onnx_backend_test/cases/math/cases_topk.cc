// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Builds a TopK NodeProto for the opset >= 10 form (K is a tensor input).
NodeProto MakeTopKNode(int64_t axis, int64_t largest = 1, int64_t sorted_attr = 1,
                       bool include_largest = false, bool include_sorted = false) {
  NodeProto node;
  node.set_op_type("TopK");
  node.add_input("x");
  node.add_input("k");
  node.add_output("values");
  node.add_output("indices");
  AddAttribute(node, "axis", axis);
  if (include_largest) {
    AddAttribute(node, "largest", largest);
  }
  if (include_sorted) {
    AddAttribute(node, "sorted", sorted_attr);
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// TopK — selects the top-K largest or smallest values along an axis.
// Mirrors the upstream ``onnx.backend.test.case.node.topk`` cases.
// ---------------------------------------------------------------------------
void RegisterTopKCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::TopK topk_kernel{ctx};

  // test_cc_top_k — 3x4 float input, axis=1, k=3, largest=1 (default).
  {
    NodeProto node = MakeTopKNode(/*axis=*/1);
    Tensor x = Tensor::FromFloat("", {3, 4},
                                 {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    Tensor k = Tensor::FromInt64("", {1}, {3});
    auto [values, indices] = topk_kernel(x, 3, /*axis=*/1, /*largest=*/true, /*sorted=*/true);
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k", {opset},
           "backend-test", registry);
  }

  // test_cc_top_k_smallest — largest=0, sorted=1, k=3.
  {
    NodeProto node = MakeTopKNode(/*axis=*/1, /*largest=*/0, /*sorted_attr=*/1,
                                  /*include_largest=*/true, /*include_sorted=*/true);
    Tensor x = Tensor::FromFloat("", {3, 4},
                                 {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 11.f, 10.f, 9.f, 8.f});
    Tensor k = Tensor::FromInt64("", {1}, {3});
    auto [values, indices] = topk_kernel(x, 3, /*axis=*/1, /*largest=*/false, /*sorted=*/true);
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_smallest", {opset},
           "backend-test", registry);
  }

  // test_cc_top_k_negative_axis — axis=-1, k=3.
  {
    NodeProto node = MakeTopKNode(/*axis=*/-1);
    Tensor x = Tensor::FromFloat("", {3, 4},
                                 {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    Tensor k = Tensor::FromInt64("", {1}, {3});
    auto [values, indices] = topk_kernel(x, 3, /*axis=*/-1, /*largest=*/true, /*sorted=*/true);
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_negative_axis",
           {opset}, "backend-test", registry);
  }

  // test_cc_top_k_same_values — 1-D input with duplicates, k=3, default attrs.
  // Tiebreak uses the smaller original index.
  {
    NodeProto node = MakeTopKNode(/*axis=*/0);
    Tensor x = Tensor::FromFloat("", {5}, {1.f, 2.f, 3.f, 3.f, 2.f});
    Tensor k = Tensor::FromInt64("", {1}, {3});
    auto [values, indices] = topk_kernel(x, 3, /*axis=*/0, /*largest=*/true, /*sorted=*/true);
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_same_values",
           {opset}, "backend-test", registry);
  }

  // test_cc_top_k_same_values_2d — 2-D input with ties along axis=1.
  {
    NodeProto node = MakeTopKNode(/*axis=*/1);
    Tensor x = Tensor::FromFloat("", {2, 4}, {1.f, 2.f, 2.f, 3.f, 5.f, 5.f, 4.f, 3.f});
    Tensor k = Tensor::FromInt64("", {1}, {3});
    auto [values, indices] = topk_kernel(x, 3, /*axis=*/1, /*largest=*/true, /*sorted=*/true);
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_same_values_2d",
           {opset}, "backend-test", registry);
  }

  // test_cc_top_k_same_values_largest — explicit largest=1 attribute with ties.
  {
    NodeProto node = MakeTopKNode(/*axis=*/0, /*largest=*/1, /*sorted_attr=*/1,
                                  /*include_largest=*/true, /*include_sorted=*/true);
    Tensor x = Tensor::FromFloat("", {5}, {1.f, 2.f, 3.f, 3.f, 2.f});
    Tensor k = Tensor::FromInt64("", {1}, {3});
    auto [values, indices] = topk_kernel(x, 3, /*axis=*/0, /*largest=*/true, /*sorted=*/true);
    Expect(node, {x, k}, {std::move(values), std::move(indices)},
           "test_cc_top_k_same_values_largest", {opset}, "backend-test", registry);
  }

  // test_cc_top_k_uint64 — uint64 input exercises the all_numeric_types
  // type constraint introduced at opset 11.
  {
    NodeProto node = MakeTopKNode(/*axis=*/1);
    Tensor x = Tensor::FromUint64("", {3, 4}, {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u});
    Tensor k = Tensor::FromInt64("", {1}, {3});
    auto [values, indices] = topk_kernel(x, 3, /*axis=*/1, /*largest=*/true, /*sorted=*/true);
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_uint64", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

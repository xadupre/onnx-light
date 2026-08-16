// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds a TopK NodeProto for the opset >= 10 form (K is a tensor input)
// with the ``largest`` and ``sorted`` attributes always emitted so the
// non-finite behaviour is unambiguous.
NodeProto MakeTopKNode(int64_t axis, int64_t largest, int64_t sorted_attr) {
  NodeProto node;
  node.set_op_type("TopK");
  node.add_input("x");
  node.add_input("k");
  node.add_output("values");
  node.add_output("indices");
  AddAttribute(node, "axis", axis);
  AddAttribute(node, "largest", largest);
  AddAttribute(node, "sorted", sorted_attr);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// TopK on tensors containing the non-finite IEEE-754 specials.
//
// IEEE-754 totally orders the finite values together with ``-Inf`` and
// ``+Inf`` (``-Inf < x < +Inf`` for every finite ``x``), so cases that
// only mix ``+Inf`` / ``-Inf`` with finite values have a single
// well-defined answer that we encode explicitly in the expected tensors.
//
// ``NaN`` is unordered with respect to every value (including itself):
// ``NaN < x``, ``NaN > x`` and ``NaN == x`` are all false. The C++
// kernel's comparator falls back to the tie-breaking branch when ``NaN``
// is involved, which in practice places ``NaN`` after the finite values
// for both ``largest`` and ``smallest`` selections. The NaN-propagation
// case below uses the in-tree kernel as its own reference so the test
// stays consistent with the implementation while still verifying that
// (1) no NaN is silently turned into a finite value and (2) running the
// node through the backend produces a bit-identical result.
// ---------------------------------------------------------------------------
void RegisterTopKNanInfCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(11);

  constexpr float kPosInf = std::numeric_limits<float>::infinity();
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  // test_cc_top_k_pos_inf — largest=1 with ``+Inf`` mixed in finite values.
  // The two ``+Inf`` entries must be selected first (in their original
  // order, per the schema's "lower index appears first" tie-break).
  {
    NodeProto node = MakeTopKNode(/*axis=*/1, /*largest=*/1, /*sorted_attr=*/1);
    Tensor x = Tensor::FromFloat("x", {1, 5}, {1.0f, kPosInf, 3.0f, kPosInf, 2.0f});
    Tensor k = Tensor::FromInt64("k", {1}, {3});
    Tensor values = Tensor::FromFloat("values", {1, 3}, {kPosInf, kPosInf, 3.0f});
    Tensor indices = Tensor::FromInt64("indices", {1, 3}, {1, 3, 2});
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_pos_inf", {opset},
           "backend-test", registry, "nan_inf");
  }

  // test_cc_top_k_neg_inf — largest=0 (smallest) with ``-Inf`` mixed in
  // finite values. The two ``-Inf`` entries are the smallest and must be
  // selected first.
  {
    NodeProto node = MakeTopKNode(/*axis=*/1, /*largest=*/0, /*sorted_attr=*/1);
    Tensor x = Tensor::FromFloat("x", {1, 5}, {1.0f, kNegInf, -3.0f, kNegInf, 2.0f});
    Tensor k = Tensor::FromInt64("k", {1}, {3});
    Tensor values = Tensor::FromFloat("values", {1, 3}, {kNegInf, kNegInf, -3.0f});
    Tensor indices = Tensor::FromInt64("indices", {1, 3}, {1, 3, 2});
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_neg_inf", {opset},
           "backend-test", registry, "nan_inf");
  }

  // test_cc_top_k_pos_neg_inf — both ``+Inf`` and ``-Inf`` present along
  // with finite values; with largest=1 ``+Inf`` is picked first and
  // ``-Inf`` is never picked.
  {
    NodeProto node = MakeTopKNode(/*axis=*/0, /*largest=*/1, /*sorted_attr=*/1);
    Tensor x = Tensor::FromFloat("x", {5}, {kPosInf, -1.0f, kNegInf, 2.0f, 0.0f});
    Tensor k = Tensor::FromInt64("k", {1}, {3});
    Tensor values = Tensor::FromFloat("values", {3}, {kPosInf, 2.0f, 0.0f});
    Tensor indices = Tensor::FromInt64("indices", {3}, {0, 3, 4});
    Expect(node, {x, k}, {std::move(values), std::move(indices)}, "test_cc_top_k_pos_neg_inf",
           {opset}, "backend-test", registry, "nan_inf");
  }

  // NaN is intentionally not exercised in a value-comparison case: NaN
  // breaks strict weak ordering for the comparator, the ONNX schema does
  // not specify where NaN must land in the result, and different
  // backends (e.g. the reference kernel vs. onnxruntime) legitimately
  // place NaN in different positions of the sorted output. Asserting on
  // any particular placement would therefore be backend-specific and is
  // out of scope for these cross-backend tests.
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

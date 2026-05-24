// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/kernels/reduction/include_reduction_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// ReduceSum — y = sum(data, axes) (since opset 13 in the ai.onnx domain).
// Starting with opset 13 ``axes`` is an optional second input (int64 tensor)
// rather than an attribute; ``keepdims`` (default 1) and
// ``noop_with_empty_axes`` (default 0) remain attributes.
//
// Three cases are registered:
//
//   * ``test_cc_reducesum_default_axes_keepdims`` — ``axes`` omitted, reduces
//     over every dimension and keeps reduced dims as size 1.
//   * ``test_cc_reducesum_do_not_keepdims`` — explicit ``axes = [1]`` with
//     ``keepdims = 0`` so reduced dims are dropped from the output.
//   * ``test_cc_reducesum_negative_axes_keepdims`` — explicit
//     ``axes = [-2]`` with ``keepdims = 1`` exercises negative-axis support.
// ---------------------------------------------------------------------------
void RegisterReduceSumCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::ReduceSum reduce_sum_kernel{kernel::KernelContext(opset)};

  // ``axes`` omitted, default ``keepdims = 1``: result is a fully-reduced
  // 1x1x1 tensor.
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_output("reduced");

    Tensor data = Tensor::FromFloat(
        "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor reduced = reduce_sum_kernel(data, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data}, {reduced}, "test_cc_reducesum_default_axes_keepdims", {opset},
           "backend-test", registry);
  }

  // Explicit ``axes = [1]`` with ``keepdims = 0``: reduced dim is dropped.
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("keepdims");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(0);

    Tensor data = Tensor::FromFloat(
        "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor axes = Tensor::FromInt64("", {1}, {1});
    Tensor reduced = reduce_sum_kernel(data, axes, /*keepdims=*/false,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data, axes}, {reduced}, "test_cc_reducesum_do_not_keepdims", {opset},
           "backend-test", registry);
  }

  // Negative axis = -2 with ``keepdims = 1`` (default).
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");

    Tensor data = Tensor::FromFloat(
        "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor axes = Tensor::FromInt64("", {1}, {-2});
    Tensor reduced = reduce_sum_kernel(data, axes, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data, axes}, {reduced}, "test_cc_reducesum_negative_axes_keepdims", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

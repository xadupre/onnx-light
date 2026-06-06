// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// ReduceSum — y = sum(data, axes) (since opset 13 in the ai.onnx domain).
// Starting with opset 13 ``axes`` is an optional second input (int64 tensor)
// rather than an attribute; ``keepdims`` (default 1) and
// ``noop_with_empty_axes`` (default 0) remain attributes.
//
// The cases below mirror every upstream ``test_reduce_sum_*`` node test
// expressible against the reference :ref:`kernel::ReduceSum`. The upstream
// suite registers each scenario twice — once with a small deterministic
// ``[[[1, 2], ...]]`` "example" input and once with a NumPy-seeded
// ``_random`` input — but the random variants only differ in their input
// data and therefore add no behavioural coverage beyond the example one;
// only the deterministic variants are ported here (and the ``_example``
// suffix is dropped to match the ``test_cc_`` naming convention used by
// the rest of this registry).
//
// Cases registered:
//
//   * ``test_cc_reducesum_default_axes_keepdims`` — ``axes`` omitted, reduces
//     over every dimension and keeps reduced dims as size 1.
//   * ``test_cc_reducesum_keepdims`` — explicit ``axes = [1]`` with the
//     default ``keepdims = 1`` so the reduced dim is preserved as size 1.
//   * ``test_cc_reducesum_do_not_keepdims`` — explicit ``axes = [1]`` with
//     ``keepdims = 0`` so reduced dims are dropped from the output.
//   * ``test_cc_reducesum_negative_axes_keepdims`` — explicit
//     ``axes = [-2]`` with ``keepdims = 1`` exercises negative-axis support.
//   * ``test_cc_reducesum_empty_axes_input_noop`` — empty ``axes`` input
//     combined with ``noop_with_empty_axes = 1``: the kernel performs an
//     identity copy instead of reducing over all dimensions.
//   * ``test_cc_reducesum_empty_set`` — reducing over an axis of size 0
//     (data shape ``[2, 0, 4]``, ``axes = [1]``, ``keepdims = 1``) returns
//     a zero-initialized tensor of shape ``[2, 1, 4]``.
//   * ``test_cc_reducesum_empty_set_non_reduced_axis_zero`` — when a
//     non-reduced axis has size 0 (data shape ``[2, 0, 4]``,
//     ``axes = [2]``, ``keepdims = 1``) the result preserves the empty
//     dimension and contains no elements.
// ---------------------------------------------------------------------------
void RegisterReduceSumCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::ReduceSum reduce_sum_kernel{ctx};

  // Upstream ``test_reduce_sum_*`` cases all share the same ``[3, 2, 2]``
  // example input ``[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]``.
  const std::vector<int64_t> example_shape = {3, 2, 2};
  const std::vector<float> example_values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // ``axes`` omitted, default ``keepdims = 1``: result is a fully-reduced
  // 1x1x1 tensor.
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_output("reduced");

    Tensor data = Tensor::FromFloat("", example_shape, example_values);
    Tensor reduced = reduce_sum_kernel(data, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data}, {reduced}, "test_cc_reducesum_default_axes_keepdims", {opset},
           "backend-test", registry);
  }

  // Explicit ``axes = [1]`` with default ``keepdims = 1``: reduced dim
  // is preserved as size 1.
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");

    Tensor data = Tensor::FromFloat("", example_shape, example_values);
    Tensor axes = Tensor::FromInt64("", {1}, {1});
    Tensor reduced = reduce_sum_kernel(data, axes, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data, axes}, {reduced}, "test_cc_reducesum_keepdims", {opset}, "backend-test",
           registry);
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

    Tensor data = Tensor::FromFloat("", example_shape, example_values);
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

    Tensor data = Tensor::FromFloat("", example_shape, example_values);
    Tensor axes = Tensor::FromInt64("", {1}, {-2});
    Tensor reduced = reduce_sum_kernel(data, axes, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data, axes}, {reduced}, "test_cc_reducesum_negative_axes_keepdims", {opset},
           "backend-test", registry);
  }

  // Empty ``axes`` input combined with ``noop_with_empty_axes = 1`` is an
  // identity (the kernel returns a copy of ``data``).
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("noop_with_empty_axes");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(1);

    Tensor data = Tensor::FromFloat("", example_shape, example_values);
    Tensor axes = Tensor::FromInt64("", {0}, {});
    Tensor reduced = reduce_sum_kernel(data, axes, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/true);

    Expect(node, {data, axes}, {reduced}, "test_cc_reducesum_empty_axes_input_noop", {opset},
           "backend-test", registry);
  }

  // Reducing over an axis of size 0: data shape ``[2, 0, 4]``, ``axes = [1]``,
  // ``keepdims = 1`` yields a zero-initialized ``[2, 1, 4]`` tensor.
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");

    Tensor data = Tensor::FromFloat("", {2, 0, 4}, {});
    Tensor axes = Tensor::FromInt64("", {1}, {1});
    Tensor reduced = reduce_sum_kernel(data, axes, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data, axes}, {reduced}, "test_cc_reducesum_empty_set", {opset}, "backend-test",
           registry);
  }

  // Non-reduced axis of size 0: data shape ``[2, 0, 4]``, ``axes = [2]``,
  // ``keepdims = 1`` yields a ``[2, 0, 1]`` tensor (no elements).
  {
    NodeProto node;
    node.set_op_type("ReduceSum");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");

    Tensor data = Tensor::FromFloat("", {2, 0, 4}, {});
    Tensor axes = Tensor::FromInt64("", {1}, {2});
    Tensor reduced = reduce_sum_kernel(data, axes, /*keepdims=*/true,
                                       /*noop_with_empty_axes=*/false);

    Expect(node, {data, axes}, {reduced}, "test_cc_reducesum_empty_set_non_reduced_axis_zero",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

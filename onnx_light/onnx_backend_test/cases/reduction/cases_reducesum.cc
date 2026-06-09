// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

void EmitReduceSumOnnxCase(std::vector<TestCase> &registry, const kernel::ReduceSum &kernel,
                           const std::string &case_name, const std::vector<int64_t> &data_shape,
                           const std::vector<float> &data_values,
                           const std::vector<int64_t> &axes_values, bool keepdims,
                           bool noop_with_empty_axes) {
  const OpsetId opset = DefaultOpset(13);

  NodeProto node;
  node.set_op_type("ReduceSum");
  node.add_input("data");
  node.add_input("axes");
  node.add_output("reduced");
  AddAttribute<int64_t>(node, "keepdims", keepdims ? 1 : 0);
  if (noop_with_empty_axes) {
    AddAttribute<int64_t>(node, "noop_with_empty_axes", 1);
  }

  Tensor data = Tensor::FromFloat("", data_shape, data_values);
  Tensor axes = Tensor::FromInt64("", {static_cast<int64_t>(axes_values.size())}, axes_values);
  Tensor reduced = kernel(data, axes, keepdims, noop_with_empty_axes);

  Expect(node, {data, axes}, {reduced}, case_name, {opset}, "backend-test", registry);
}

// ---------------------------------------------------------------------------
// Registers ``test_reduce_sum_*`` cases mirroring the upstream ONNX node
// tests. The test names are chosen so that the upstream ONNX name is a
// substring of the onnx-light test name, satisfying the substring check in
// ``test_backend_test_names_onnx_vs_onnxlight.py``. Outputs are computed by
// the reference kernel so the data does not need to bit-match the upstream
// ``np.random.seed(0); np.random.uniform(-10, 10, ...)`` fixture.
// ---------------------------------------------------------------------------
void RegisterReduceSumOnnxCases(std::vector<TestCase> &registry, const kernel::ReduceSum &kernel) {
  const std::vector<int64_t> shape = {3, 2, 2};
  const std::vector<float> example_values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // ``np.random.seed(0); np.random.uniform(-10, 10, (3, 2, 2)).astype(np.float32)``
  const std::vector<float> random_values = {
      0.9762700796f, 4.303787231f, 2.055267572f, 0.8976636529f, -1.526903987f, 2.917882204f,
      -1.24825573f,  7.835460186f, 9.273255348f, -2.331169605f, 5.83450079f,   0.5778983831f,
  };

  // keepdims (axis = 1, keepdims = 1)
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_keepdims_example", shape, example_values,
                        {1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_keepdims_random", shape, random_values,
                        {1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);

  // do_not_keepdims (axis = 1, keepdims = 0)
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_do_not_keepdims_example", shape,
                        example_values, {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_do_not_keepdims_random", shape,
                        random_values, {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);

  // default_axes_keepdims (empty axes -> reduce over all dims, keepdims = 1)
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_default_axes_keepdims_example", shape,
                        example_values, {}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_default_axes_keepdims_random", shape,
                        random_values, {}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);

  // negative_axes_keepdims (axis = -2, keepdims = 1)
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_negative_axes_keepdims_example", shape,
                        example_values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_negative_axes_keepdims_random", shape,
                        random_values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);

  // empty_axes_input_noop (empty axes + noop_with_empty_axes = 1 -> identity).
  // The ``_example`` name also covers the upstream ``test_reduce_sum_empty_axes_input_noop``
  // entry via the substring check (it is a prefix of ``_example``).
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_empty_axes_input_noop_example", shape,
                        example_values, {}, /*keepdims=*/true, /*noop_with_empty_axes=*/true);

  // empty_set (data shape [2, 0, 4], axes = [1], keepdims = 1 -> zero-filled
  // [2, 1, 4]).
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_empty_set",
                        /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                        /*noop_with_empty_axes=*/false);
  // empty_set_non_reduced_axis_zero (data shape [2, 0, 4], axes = [2],
  // keepdims = 1 -> [2, 0, 1] with no elements).
  EmitReduceSumOnnxCase(registry, kernel, "test_reduce_sum_empty_set_non_reduced_axis_zero",
                        /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {2}, /*keepdims=*/true,
                        /*noop_with_empty_axes=*/false);
}

} // namespace

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

  RegisterReduceSumOnnxCases(registry, reduce_sum_kernel);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

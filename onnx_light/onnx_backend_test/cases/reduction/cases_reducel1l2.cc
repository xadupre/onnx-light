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

void EmitReduceL1L2Case(std::vector<TestCase> &registry, const std::string &op_type,
                        const kernel::ReduceL1L2 &kernel, const std::string &case_name,
                        const std::vector<int64_t> &data_shape,
                        const std::vector<float> &data_values,
                        const std::vector<int64_t> &axes_values, bool keepdims,
                        bool noop_with_empty_axes) {
  const OpsetId opset = DefaultOpset(18);

  NodeProto node;
  node.set_op_type(op_type);
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

// Emits a case where the optional ``axes`` input is omitted entirely (single
// "data" input). With ``noop_with_empty_axes`` default-false this reduces
// over every dimension of ``data``.
void EmitReduceL1L2DefaultAxesCase(std::vector<TestCase> &registry, const std::string &op_type,
                                   const kernel::ReduceL1L2 &kernel, const std::string &case_name,
                                   const std::vector<int64_t> &data_shape,
                                   const std::vector<float> &data_values, bool keepdims) {
  const OpsetId opset = DefaultOpset(18);

  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("data");
  node.add_output("reduced");
  AddAttribute<int64_t>(node, "keepdims", keepdims ? 1 : 0);

  Tensor data = Tensor::FromFloat("", data_shape, data_values);
  Tensor reduced = kernel(data, keepdims, /*noop_with_empty_axes=*/false);

  Expect(node, {data}, {reduced}, case_name, {opset}, "backend-test", registry);
}

void RegisterReduceL1L2Cases(std::vector<TestCase> &registry, const std::string &op_type,
                             const kernel::ReduceL1L2 &kernel, const std::string &name_prefix) {
  const std::vector<int64_t> shape = {3, 2, 2};
  // Same ``[3, 2, 2]`` ``arange(1, 13)`` payload used by the sibling
  // ``cases_reducesum.cc`` / ``cases_reduceminmax.cc`` files and by the
  // upstream ONNX reference tests for ``ReduceL1`` / ``ReduceL2``.
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  EmitReduceL1L2DefaultAxesCase(registry, op_type, kernel,
                                "test_cc_" + name_prefix + "_default_axes_keepdims", shape, values,
                                /*keepdims=*/true);
  EmitReduceL1L2Case(registry, op_type, kernel, "test_cc_" + name_prefix + "_keepdims", shape,
                     values, {1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceL1L2Case(registry, op_type, kernel, "test_cc_" + name_prefix + "_do_not_keepdims",
                     shape, values, {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceL1L2Case(registry, op_type, kernel,
                     "test_cc_" + name_prefix + "_negative_axes_keepdims", shape, values, {-2},
                     /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceL1L2Case(registry, op_type, kernel, "test_cc_" + name_prefix + "_empty_axes_input_noop",
                     shape, values, {},
                     /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  // Reducing over an axis of size 0: the L1/L2 identity is 0, so the result
  // is a zero-filled tensor.
  EmitReduceL1L2Case(registry, op_type, kernel, "test_cc_" + name_prefix + "_empty_set",
                     /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                     /*noop_with_empty_axes=*/false);
  // Non-reduced axis of size 0: the result preserves the empty dimension.
  EmitReduceL1L2Case(registry, op_type, kernel,
                     "test_cc_" + name_prefix + "_empty_set_non_reduced_axis_zero",
                     /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {2}, /*keepdims=*/true,
                     /*noop_with_empty_axes=*/false);
}

// Registers the nine ONNX reference backend test cases for ReduceL1 / ReduceL2
// (non-expanded variants). The test names are chosen so that the ONNX test
// name is a substring of the onnx-light test name, satisfying the substring
// check in ``test_backend_test_names_onnx_vs_onnxlight.py``.
//
// The upstream tests come in ``_example`` / ``_random`` pairs that share the
// same axes configuration but differ only in input values:
//
//   * ``_example``  — deterministic ``arange(1, 13)`` input.
//   * ``_random``   — ``np.random.seed(0); np.random.uniform(-10, 10, (3,2,2))``
//                     cast to float32 (same seed across all reduce-op random
//                     variants in the upstream ONNX test suite).
void RegisterReduceL1L2OnnxCases(std::vector<TestCase> &registry, const std::string &op_type,
                                 const kernel::ReduceL1L2 &kernel, const std::string &onnx_prefix) {
  const std::vector<int64_t> shape = {3, 2, 2};
  const std::vector<float> example_values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // ``np.random.seed(0); np.random.uniform(-10, 10, (3, 2, 2)).astype(np.float32)``
  const std::vector<float> random_values = {
      0.9762700796f, 4.303787231f, 2.055267572f, 0.8976636529f, -1.526903987f, 2.917882204f,
      -1.24825573f,  7.835460186f, 9.273255348f, -2.331169605f, 5.83450079f,   0.5778983831f,
  };

  // keep_dims (axis = 2, keepdims = 1)
  EmitReduceL1L2Case(registry, op_type, kernel, "test_" + onnx_prefix + "_keep_dims_example", shape,
                     example_values, {2},
                     /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceL1L2Case(registry, op_type, kernel, "test_" + onnx_prefix + "_keep_dims_random", shape,
                     random_values, {2},
                     /*keepdims=*/true, /*noop_with_empty_axes=*/false);

  // do_not_keepdims (axis = 2, keepdims = 0)
  EmitReduceL1L2Case(registry, op_type, kernel, "test_" + onnx_prefix + "_do_not_keepdims_example",
                     shape, example_values, {2}, /*keepdims=*/false,
                     /*noop_with_empty_axes=*/false);
  EmitReduceL1L2Case(registry, op_type, kernel, "test_" + onnx_prefix + "_do_not_keepdims_random",
                     shape, random_values, {2},
                     /*keepdims=*/false, /*noop_with_empty_axes=*/false);

  // default_axes_keepdims (empty axes tensor → reduce over all dims, keepdims = 1)
  EmitReduceL1L2Case(registry, op_type, kernel,
                     "test_" + onnx_prefix + "_default_axes_keepdims_example", shape,
                     example_values, {}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceL1L2Case(registry, op_type, kernel,
                     "test_" + onnx_prefix + "_default_axes_keepdims_random", shape, random_values,
                     {}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);

  // negative_axes_keep_dims (axis = -1, keepdims = 1)
  EmitReduceL1L2Case(registry, op_type, kernel,
                     "test_" + onnx_prefix + "_negative_axes_keep_dims_example", shape,
                     example_values, {-1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceL1L2Case(registry, op_type, kernel,
                     "test_" + onnx_prefix + "_negative_axes_keep_dims_random", shape,
                     random_values, {-1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);

  // empty_set (data shape [2, 0, 4], axis = 1, keepdims = 1)
  EmitReduceL1L2Case(registry, op_type, kernel, "test_" + onnx_prefix + "_empty_set",
                     /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                     /*noop_with_empty_axes=*/false);
}

} // namespace

void RegisterReduceL1Cases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceL1 reduce_l1_kernel{ctx};
  RegisterReduceL1L2Cases(registry, "ReduceL1", reduce_l1_kernel, "reducel1");
  RegisterReduceL1L2OnnxCases(registry, "ReduceL1", reduce_l1_kernel, "reduce_l1");
}

void RegisterReduceL2Cases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceL2 reduce_l2_kernel{ctx};
  RegisterReduceL1L2Cases(registry, "ReduceL2", reduce_l2_kernel, "reducel2");
  RegisterReduceL1L2OnnxCases(registry, "ReduceL2", reduce_l2_kernel, "reduce_l2");
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

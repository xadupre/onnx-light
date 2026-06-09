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

void EmitReduceMinMaxCase(std::vector<TestCase> &registry, const std::string &op_type,
                          const kernel::ReduceMinMax &kernel, const std::string &case_name,
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
void EmitReduceMinMaxDefaultAxesCase(std::vector<TestCase> &registry, const std::string &op_type,
                                     const kernel::ReduceMinMax &kernel,
                                     const std::string &case_name,
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

void RegisterReduceMinMaxCases(std::vector<TestCase> &registry, const std::string &op_type,
                               const kernel::ReduceMinMax &kernel, const std::string &name_prefix) {
  const std::vector<int64_t> shape = {3, 2, 2};
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // ``axes`` omitted, default ``keepdims = 1``: reduces over every dimension
  // and keeps reduced dims as size 1.
  EmitReduceMinMaxDefaultAxesCase(registry, op_type, kernel,
                                  "test_cc_" + name_prefix + "_default_axes_keepdims", shape,
                                  values, /*keepdims=*/true);
  EmitReduceMinMaxCase(registry, op_type, kernel, "test_cc_" + name_prefix + "_keepdims", shape,
                       values, {1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel, "test_cc_" + name_prefix + "_do_not_keepdims",
                       shape, values, {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel,
                       "test_cc_" + name_prefix + "_negative_axes_keepdims", shape, values, {-2},
                       /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel,
                       "test_cc_" + name_prefix + "_empty_axes_input_noop", shape, values, {},
                       /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  // Reducing over a size-0 axis: data shape ``[2, 0, 4]``, axes=[1],
  // ``keepdims=1`` yields a ``[2, 1, 4]`` tensor filled with the identity of
  // the reduction (``-inf`` for ReduceMax, ``+inf`` for ReduceMin).
  EmitReduceMinMaxCase(registry, op_type, kernel, "test_cc_" + name_prefix + "_empty_set",
                       /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
  // Non-reduced axis of size 0: data shape ``[2, 0, 4]``, axes=[2],
  // ``keepdims=1`` yields a ``[2, 0, 1]`` tensor (no elements).
  EmitReduceMinMaxCase(registry, op_type, kernel,
                       "test_cc_" + name_prefix + "_empty_set_non_reduced_axis_zero",
                       /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {2}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
}

// ---------------------------------------------------------------------------
// Registers ``test_reduce_max_*`` / ``test_reduce_min_*`` cases mirroring the
// upstream ONNX node tests. The test names are chosen so that the upstream
// ONNX name is a substring of the onnx-light test name, satisfying the
// substring check in ``test_backend_test_names_onnx_vs_onnxlight.py``.
// Outputs are computed by the reference kernel so the data does not need to
// bit-match the upstream fixture.
// ---------------------------------------------------------------------------
void RegisterReduceMinMaxOnnxCases(std::vector<TestCase> &registry, const std::string &op_type,
                                   const kernel::ReduceMinMax &kernel,
                                   const std::string &onnx_prefix) {
  const std::vector<int64_t> shape = {3, 2, 2};
  const std::vector<float> example_values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // ``np.random.seed(0); np.random.uniform(-10, 10, (3, 2, 2)).astype(np.float32)``
  const std::vector<float> random_values = {
      0.9762700796f, 4.303787231f, 2.055267572f, 0.8976636529f, -1.526903987f, 2.917882204f,
      -1.24825573f,  7.835460186f, 9.273255348f, -2.331169605f, 5.83450079f,   0.5778983831f,
  };

  EmitReduceMinMaxCase(registry, op_type, kernel, "test_" + onnx_prefix + "_keepdims_example",
                       shape, example_values, {1}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel, "test_" + onnx_prefix + "_keepdims_random", shape,
                       random_values, {1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel,
                       "test_" + onnx_prefix + "_do_not_keepdims_example", shape, example_values,
                       {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel, "test_" + onnx_prefix + "_do_not_keepdims_random",
                       shape, random_values, {1},
                       /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxDefaultAxesCase(registry, op_type, kernel,
                                  "test_" + onnx_prefix + "_default_axes_keepdims_random", shape,
                                  random_values, /*keepdims=*/true);
  EmitReduceMinMaxCase(registry, op_type, kernel,
                       "test_" + onnx_prefix + "_negative_axes_keepdims_example", shape,
                       example_values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel,
                       "test_" + onnx_prefix + "_negative_axes_keepdims_random", shape,
                       random_values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMinMaxCase(registry, op_type, kernel, "test_" + onnx_prefix + "_empty_set",
                       /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
}

} // namespace

void RegisterReduceMaxCases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceMax reduce_max_kernel{ctx};
  RegisterReduceMinMaxCases(registry, "ReduceMax", reduce_max_kernel, "reducemax");
  RegisterReduceMinMaxOnnxCases(registry, "ReduceMax", reduce_max_kernel, "reduce_max");
  // Upstream uses a singular "keepdim" typo for the ``_example`` variant of
  // the default-axes case (``test_reduce_max_default_axes_keepdim_example``).
  EmitReduceMinMaxDefaultAxesCase(
      registry, "ReduceMax", reduce_max_kernel, "test_reduce_max_default_axes_keepdim_example",
      /*data_shape=*/{3, 2, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
      /*keepdims=*/true);
}

void RegisterReduceMinCases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceMin reduce_min_kernel{ctx};
  RegisterReduceMinMaxCases(registry, "ReduceMin", reduce_min_kernel, "reducemin");
  RegisterReduceMinMaxOnnxCases(registry, "ReduceMin", reduce_min_kernel, "reduce_min");
  EmitReduceMinMaxDefaultAxesCase(
      registry, "ReduceMin", reduce_min_kernel, "test_reduce_min_default_axes_keepdims_example",
      /*data_shape=*/{3, 2, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
      /*keepdims=*/true);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

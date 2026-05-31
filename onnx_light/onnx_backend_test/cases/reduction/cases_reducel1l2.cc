// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/kernels/reduction/include_reduction_kernels.h"
#include "onnx_backend_test/test_case.h"
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
  // Mix of positive and negative values so the L1 case (sum of |x|) differs
  // from a plain ReduceSum and the L2 case (sqrt of sum of squares) exercises
  // both the squaring and the final sqrt.
  const std::vector<float> values = {1.0f, -2.0f, 3.0f, -4.0f,  5.0f,  -6.0f,
                                     7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f};
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

} // namespace

void RegisterReduceL1Cases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceL1 reduce_l1_kernel{ctx};
  RegisterReduceL1L2Cases(registry, "ReduceL1", reduce_l1_kernel, "reducel1");
}

void RegisterReduceL2Cases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceL2 reduce_l2_kernel{ctx};
  RegisterReduceL1L2Cases(registry, "ReduceL2", reduce_l2_kernel, "reducel2");
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

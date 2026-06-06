// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

void EmitReduceSumSquareCase(std::vector<TestCase> &registry, const kernel::ReduceSumSquare &kernel,
                             const std::string &case_name, const std::vector<int64_t> &data_shape,
                             const std::vector<float> &data_values,
                             const std::vector<int64_t> &axes_values, bool keepdims,
                             bool noop_with_empty_axes) {
  const OpsetId opset = DefaultOpset(18);

  NodeProto node;
  node.set_op_type("ReduceSumSquare");
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
void EmitReduceSumSquareDefaultAxesCase(std::vector<TestCase> &registry,
                                        const kernel::ReduceSumSquare &kernel,
                                        const std::string &case_name,
                                        const std::vector<int64_t> &data_shape,
                                        const std::vector<float> &data_values, bool keepdims) {
  const OpsetId opset = DefaultOpset(18);

  NodeProto node;
  node.set_op_type("ReduceSumSquare");
  node.add_input("data");
  node.add_output("reduced");
  AddAttribute<int64_t>(node, "keepdims", keepdims ? 1 : 0);

  Tensor data = Tensor::FromFloat("", data_shape, data_values);
  Tensor reduced = kernel(data, keepdims, /*noop_with_empty_axes=*/false);

  Expect(node, {data}, {reduced}, case_name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// ReduceSumSquare — y = sum(x * x, axes). Mirrors the upstream
// ``test_reduce_sum_square_*`` node tests; the case list parallels
// ``cases_reducel1l2.cc`` since ReduceSumSquare shares the same attribute /
// input layout (data + optional axes input, ``keepdims`` and
// ``noop_with_empty_axes`` attributes) as the other simple reductions added in
// opset 18.
// ---------------------------------------------------------------------------
void RegisterReduceSumSquareCases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceSumSquare kernel{ctx};

  const std::vector<int64_t> shape = {3, 2, 2};
  // Same ``[3, 2, 2]`` ``arange(1, 13)`` payload used by the sibling
  // ``cases_reducesum.cc`` / ``cases_reducel1l2.cc`` files and by the upstream
  // ONNX reference tests for ``ReduceSumSquare``.
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  EmitReduceSumSquareDefaultAxesCase(registry, kernel,
                                     "test_cc_reducesumsquare_default_axes_keepdims", shape, values,
                                     /*keepdims=*/true);
  EmitReduceSumSquareCase(registry, kernel, "test_cc_reducesumsquare_keepdims", shape, values, {1},
                          /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceSumSquareCase(registry, kernel, "test_cc_reducesumsquare_do_not_keepdims", shape,
                          values, {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceSumSquareCase(registry, kernel, "test_cc_reducesumsquare_negative_axes_keepdims", shape,
                          values, {-2}, /*keepdims=*/true,
                          /*noop_with_empty_axes=*/false);
  EmitReduceSumSquareCase(registry, kernel, "test_cc_reducesumsquare_empty_axes_input_noop", shape,
                          values, {}, /*keepdims=*/true,
                          /*noop_with_empty_axes=*/true);
  // Reducing over an axis of size 0: the sum-square identity is 0, so the
  // result is a zero-filled tensor.
  EmitReduceSumSquareCase(registry, kernel, "test_cc_reducesumsquare_empty_set",
                          /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                          /*noop_with_empty_axes=*/false);
  // Non-reduced axis of size 0: the result preserves the empty dimension.
  EmitReduceSumSquareCase(registry, kernel,
                          "test_cc_reducesumsquare_empty_set_non_reduced_axis_zero",
                          /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {2}, /*keepdims=*/true,
                          /*noop_with_empty_axes=*/false);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

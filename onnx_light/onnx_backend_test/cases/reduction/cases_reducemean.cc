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

void EmitReduceMeanCase(std::vector<TestCase> &registry, const kernel::ReduceMean &kernel,
                        const std::string &case_name, const std::vector<int64_t> &data_shape,
                        const std::vector<float> &data_values,
                        const std::vector<int64_t> &axes_values, bool keepdims,
                        bool noop_with_empty_axes) {
  const OpsetId opset = DefaultOpset(18);

  NodeProto node;
  node.set_op_type("ReduceMean");
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
void EmitReduceMeanDefaultAxesCase(std::vector<TestCase> &registry,
                                   const kernel::ReduceMean &kernel, const std::string &case_name,
                                   const std::vector<int64_t> &data_shape,
                                   const std::vector<float> &data_values, bool keepdims) {
  const OpsetId opset = DefaultOpset(18);

  NodeProto node;
  node.set_op_type("ReduceMean");
  node.add_input("data");
  node.add_output("reduced");
  AddAttribute<int64_t>(node, "keepdims", keepdims ? 1 : 0);

  Tensor data = Tensor::FromFloat("", data_shape, data_values);
  Tensor reduced = kernel(data, keepdims, /*noop_with_empty_axes=*/false);

  Expect(node, {data}, {reduced}, case_name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// ReduceMean — y = mean(x, axes). Mirrors the upstream ``test_reduce_mean_*``
// node tests; the case list parallels ``cases_reduceprod.cc`` since
// ReduceMean shares the same attribute / input layout (data + optional axes
// input, ``keepdims`` and ``noop_with_empty_axes`` attributes) as the other
// simple reductions added in opset 18.
// ---------------------------------------------------------------------------
void RegisterReduceMeanCases(std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{DefaultOpset(18)};
  const kernel::ReduceMean kernel{ctx};

  const std::vector<int64_t> shape = {3, 2, 2};
  // Same ``[3, 2, 2]`` ``arange(1, 13)`` payload used by the sibling
  // ``cases_reducesum.cc`` / ``cases_reduceprod.cc`` files and by the
  // upstream ONNX reference tests for ``ReduceMean``.
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  EmitReduceMeanDefaultAxesCase(registry, kernel, "test_cc_reducemean_default_axes_keepdims", shape,
                                values, /*keepdims=*/true);
  EmitReduceMeanCase(registry, kernel, "test_cc_reducemean_keepdims", shape, values, {1},
                     /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMeanCase(registry, kernel, "test_cc_reducemean_do_not_keepdims", shape, values, {1},
                     /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceMeanCase(registry, kernel, "test_cc_reducemean_negative_axes_keepdims", shape, values,
                     {-2}, /*keepdims=*/true,
                     /*noop_with_empty_axes=*/false);
  EmitReduceMeanCase(registry, kernel, "test_cc_reducemean_empty_axes_input_noop", shape, values,
                     {}, /*keepdims=*/true,
                     /*noop_with_empty_axes=*/true);

  // Upstream ``test_reduce_mean_*`` ONNX node tests.
  const std::vector<float> random_values = {
      0.9762700796f, 4.303787231f, 2.055267572f, 0.8976636529f, -1.526903987f, 2.917882204f,
      -1.24825573f,  7.835460186f, 9.273255348f, -2.331169605f, 5.83450079f,   0.5778983831f,
  };
  EmitReduceMeanCase(registry, kernel, "test_reduce_mean_keepdims_example", shape, values, {1},
                     /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMeanCase(registry, kernel, "test_reduce_mean_keepdims_random", shape, random_values,
                     {1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMeanCase(registry, kernel, "test_reduce_mean_do_not_keepdims_example", shape, values,
                     {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceMeanCase(registry, kernel, "test_reduce_mean_do_not_keepdims_random", shape,
                     random_values, {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceMeanDefaultAxesCase(registry, kernel, "test_reduce_mean_default_axes_keepdims_example",
                                shape, values, /*keepdims=*/true);
  EmitReduceMeanDefaultAxesCase(registry, kernel, "test_reduce_mean_default_axes_keepdims_random",
                                shape, random_values, /*keepdims=*/true);
  EmitReduceMeanCase(registry, kernel, "test_reduce_mean_negative_axes_keepdims_example", shape,
                     values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceMeanCase(registry, kernel, "test_reduce_mean_negative_axes_keepdims_random", shape,
                     random_values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

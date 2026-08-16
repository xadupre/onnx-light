// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

void EmitReduceLogSumCase(std::vector<TestCase> &registry, const std::string &op_type,
                          const onnx_kernels::kernel::ReduceLogSumOp &kernel,
                          const std::string &case_name, const std::vector<int64_t> &data_shape,
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
  Expect(registry, std::move(node), case_name, {opset}, [=]() -> IoData {
    Tensor data = Tensor::FromFloat("", data_shape, data_values);
    Tensor axes = Tensor::FromInt64("", {static_cast<int64_t>(axes_values.size())}, axes_values);
    Tensor reduced = kernel(data, axes, keepdims, noop_with_empty_axes);

    return IoData{{std::move(data), std::move(axes)}, {std::move(reduced)}};
  });
}

// Emits a case where the optional ``axes`` input is omitted entirely (single
// "data" input). With ``noop_with_empty_axes`` default-false this reduces
// over every dimension of ``data``.
void EmitReduceLogSumDefaultAxesCase(std::vector<TestCase> &registry, const std::string &op_type,
                                     const onnx_kernels::kernel::ReduceLogSumOp &kernel,
                                     const std::string &case_name,
                                     const std::vector<int64_t> &data_shape,
                                     const std::vector<float> &data_values, bool keepdims) {
  const OpsetId opset = DefaultOpset(18);

  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("data");
  node.add_output("reduced");
  AddAttribute<int64_t>(node, "keepdims", keepdims ? 1 : 0);
  Expect(registry, std::move(node), case_name, {opset}, [=]() -> IoData {
    Tensor data = Tensor::FromFloat("", data_shape, data_values);
    Tensor reduced = kernel(data, keepdims, /*noop_with_empty_axes=*/false);

    return IoData{{std::move(data)}, {std::move(reduced)}};
  });
}

void RegisterReduceLogSumOpCases(std::vector<TestCase> &registry, const std::string &op_type,
                                 const onnx_kernels::kernel::ReduceLogSumOp &kernel,
                                 const std::string &name_prefix) {
  const std::vector<int64_t> shape = {3, 2, 2};
  // Same ``[3, 2, 2]`` ``arange(1, 13)`` payload used by the sibling
  // ``cases_reducesum.cc`` / ``cases_reducel1l2.cc`` files and by the
  // upstream ONNX reference tests.
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  EmitReduceLogSumDefaultAxesCase(registry, op_type, kernel,
                                  "test_cc_" + name_prefix + "_default_axes_keepdims", shape,
                                  values, /*keepdims=*/true);
  EmitReduceLogSumCase(registry, op_type, kernel, "test_cc_" + name_prefix + "_keepdims", shape,
                       values, {1}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, op_type, kernel, "test_cc_" + name_prefix + "_do_not_keepdims",
                       shape, values, {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, op_type, kernel,
                       "test_cc_" + name_prefix + "_negative_axes_keepdims", shape, values, {-2},
                       /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, op_type, kernel,
                       "test_cc_" + name_prefix + "_empty_axes_input_noop", shape, values, {},
                       /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  // Reducing over an axis of size 0: the empty-set identity is -inf
  // (``log(sum(empty)) == log(0)``).
  EmitReduceLogSumCase(registry, op_type, kernel, "test_cc_" + name_prefix + "_empty_set",
                       /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
  // Non-reduced axis of size 0: the result preserves the empty dimension.
  EmitReduceLogSumCase(registry, op_type, kernel,
                       "test_cc_" + name_prefix + "_empty_set_non_reduced_axis_zero",
                       /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {2}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
}

// ---------------------------------------------------------------------------
// Registers ``test_reduce_log_sum_exp_*`` cases mirroring the upstream
// ``ReduceLogSumExp`` ONNX node tests. The test names are chosen so that the
// upstream ONNX name is a substring of the onnx-light test name, satisfying
// the substring check in ``test_backend_test_names_onnx_vs_onnxlight.py``.
// Outputs are computed by the reference kernel so the data does not need to
// bit-match the upstream fixture.
// ---------------------------------------------------------------------------
void RegisterReduceLogSumExpOnnxCases(std::vector<TestCase> &registry,
                                      const onnx_kernels::kernel::ReduceLogSumOp &kernel) {
  const std::vector<int64_t> shape = {3, 2, 2};
  const std::vector<float> example_values = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // ``np.random.seed(0); np.random.uniform(-10, 10, (3, 2, 2)).astype(np.float32)``
  const std::vector<float> random_values = {
      0.9762700796f, 4.303787231f, 2.055267572f, 0.8976636529f, -1.526903987f, 2.917882204f,
      -1.24825573f,  7.835460186f, 9.273255348f, -2.331169605f, 5.83450079f,   0.5778983831f,
  };

  EmitReduceLogSumCase(registry, "ReduceLogSumExp", kernel,
                       "test_reduce_log_sum_exp_keepdims_example", shape, example_values, {1},
                       /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSumExp", kernel,
                       "test_reduce_log_sum_exp_keepdims_random", shape, random_values, {1},
                       /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSumExp", kernel,
                       "test_reduce_log_sum_exp_do_not_keepdims_example", shape, example_values,
                       {1}, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSumExp", kernel,
                       "test_reduce_log_sum_exp_do_not_keepdims_random", shape, random_values, {1},
                       /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumDefaultAxesCase(registry, "ReduceLogSumExp", kernel,
                                  "test_reduce_log_sum_exp_default_axes_keepdims_example", shape,
                                  example_values, /*keepdims=*/true);
  EmitReduceLogSumDefaultAxesCase(registry, "ReduceLogSumExp", kernel,
                                  "test_reduce_log_sum_exp_default_axes_keepdims_random", shape,
                                  random_values, /*keepdims=*/true);
  EmitReduceLogSumCase(registry, "ReduceLogSumExp", kernel,
                       "test_reduce_log_sum_exp_negative_axes_keepdims_example", shape,
                       example_values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSumExp", kernel,
                       "test_reduce_log_sum_exp_negative_axes_keepdims_random", shape,
                       random_values, {-2}, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSumExp", kernel, "test_reduce_log_sum_exp_empty_set",
                       /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
}

// ---------------------------------------------------------------------------
// Registers ``test_reduce_log_sum_*`` cases (the plain ``ReduceLogSum``, not
// the ``Exp`` variant). Upstream only ships a smaller set: ``_asc_axes``,
// ``_desc_axes``, ``_default`` (axes omitted, keepdims default), and
// ``_negative_axes`` (axes = [-2]). ``_empty_set`` is shared with the other
// reductions. Outputs are computed by the reference kernel.
// ---------------------------------------------------------------------------
void RegisterReduceLogSumOnnxCases(std::vector<TestCase> &registry,
                                   const onnx_kernels::kernel::ReduceLogSumOp &kernel) {
  // Multi-axis cases use the upstream ``[3, 4, 5]`` shape.
  const std::vector<int64_t> shape_3_4_5 = {3, 4, 5};
  std::vector<float> data_3_4_5(3 * 4 * 5);
  for (size_t i = 0; i < data_3_4_5.size(); ++i) {
    data_3_4_5[i] = 0.1f + static_cast<float>(i) * 0.01f;
  }

  EmitReduceLogSumCase(registry, "ReduceLogSum", kernel, "test_reduce_log_sum_asc_axes",
                       shape_3_4_5, data_3_4_5, {0, 1}, /*keepdims=*/false,
                       /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSum", kernel, "test_reduce_log_sum_desc_axes",
                       shape_3_4_5, data_3_4_5, {2, 1}, /*keepdims=*/false,
                       /*noop_with_empty_axes=*/false);
  // ``_default``: empty axes, keepdims = 1 (the default).
  EmitReduceLogSumCase(registry, "ReduceLogSum", kernel, "test_reduce_log_sum_default", shape_3_4_5,
                       data_3_4_5, {}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSum", kernel, "test_reduce_log_sum_negative_axes",
                       shape_3_4_5, data_3_4_5, {-2}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
  EmitReduceLogSumCase(registry, "ReduceLogSum", kernel, "test_reduce_log_sum_empty_set",
                       /*data_shape=*/{2, 0, 4}, /*data_values=*/{}, {1}, /*keepdims=*/true,
                       /*noop_with_empty_axes=*/false);
}

} // namespace

void RegisterReduceLogSumCases(std::vector<TestCase> &registry, TestMode mode) {
  const KernelContext ctx{DefaultOpset(18)};
  const onnx_kernels::kernel::ReduceLogSum kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("ReduceLogSum");
    node.add_input("data");
    node.add_output("reduced");
    AddAttribute<int64_t>(node, "keepdims", 1);

    Expect(registry, std::move(node), "test_cc_reducelogsum_default_axes_keepdims_benchmark",
           {DefaultOpset(18)}, {256 * 256 * 16}, {1}, [kernel]() -> IoData {
             Tensor data = RandnTensor(DataType::FLOAT, {256, 256, 16}, /*seed=*/9701);
             Tensor reduced = kernel(data, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
             return IoData{{std::move(data)}, {std::move(reduced)}};
           });
    return;
  }
  RegisterReduceLogSumOpCases(registry, "ReduceLogSum", kernel, "reducelogsum");
  RegisterReduceLogSumOnnxCases(registry, kernel);

  // DOUBLE
  {
    const OpsetId opset = DefaultOpset(18);
    NodeProto node;
    node.set_op_type("ReduceLogSum");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");
    AddAttribute<int64_t>(node, "keepdims", 1);
    Expect(registry, std::move(node), "test_cc_reducelogsum_double", {opset}, [=]() -> IoData {
      Tensor data = Tensor::FromDouble(
          "", {3, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
      Tensor axes = Tensor::FromInt64("", {1}, {1});
      Tensor reduced = kernel(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
      return IoData{{std::move(data), std::move(axes)}, {std::move(reduced)}};
    });
  }
}

void RegisterReduceLogSumExpCases(std::vector<TestCase> &registry, TestMode mode) {
  const KernelContext ctx{DefaultOpset(18)};
  const onnx_kernels::kernel::ReduceLogSumExp kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("ReduceLogSumExp");
    node.add_input("data");
    node.add_output("reduced");
    AddAttribute<int64_t>(node, "keepdims", 1);

    Expect(registry, std::move(node), "test_cc_reducelogsumexp_default_axes_keepdims_benchmark",
           {DefaultOpset(18)}, {256 * 256 * 16}, {1}, [kernel]() -> IoData {
             Tensor data = RandnTensor(DataType::FLOAT, {256, 256, 16}, /*seed=*/9701);
             Tensor reduced = kernel(data, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
             return IoData{{std::move(data)}, {std::move(reduced)}};
           });
    return;
  }
  RegisterReduceLogSumOpCases(registry, "ReduceLogSumExp", kernel, "reducelogsumexp");
  RegisterReduceLogSumExpOnnxCases(registry, kernel);

  // DOUBLE
  {
    const OpsetId opset = DefaultOpset(18);
    NodeProto node;
    node.set_op_type("ReduceLogSumExp");
    node.add_input("data");
    node.add_input("axes");
    node.add_output("reduced");
    AddAttribute<int64_t>(node, "keepdims", 1);
    Expect(registry, std::move(node), "test_cc_reducelogsumexp_double", {opset}, [=]() -> IoData {
      Tensor data = Tensor::FromDouble(
          "", {3, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
      Tensor axes = Tensor::FromInt64("", {1}, {1});
      Tensor reduced = kernel(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
      return IoData{{std::move(data), std::move(axes)}, {std::move(reduced)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

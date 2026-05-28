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

// Shared helper that materialises a single ``ArgMax``/``ArgMin`` upstream
// case. ``op_type`` is ``"ArgMax"`` or ``"ArgMin"``; ``mode`` selects the
// reference kernel; the remaining parameters mirror the upstream Python
// ``onnx.backend.test.case.node.argmax`` / ``argmin`` definitions.
void EmitArgReduceCase(std::vector<TestCase> &registry, const std::string &op_type,
                       kernel::ArgReduce::Mode mode, const std::string &case_name,
                       const std::vector<int64_t> &data_shape,
                       const std::vector<float> &data_values, bool include_axis, int64_t axis,
                       int64_t keepdims, bool select_last_index) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::ArgReduce arg_kernel{kernel::KernelContext(opset), mode};

  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("data");
  node.add_output("result");
  if (include_axis) {
    AddAttribute<int64_t>(node, "axis", axis);
  }
  AddAttribute<int64_t>(node, "keepdims", keepdims);
  if (select_last_index) {
    AddAttribute<int64_t>(node, "select_last_index", 1);
  }

  Tensor data = Tensor::FromFloat("", data_shape, data_values);
  Tensor result = arg_kernel(data, include_axis ? axis : 0, keepdims != 0, select_last_index);

  Expect(node, {data}, {result}, case_name, {opset}, "backend-test", registry);
}

// Registers the eight upstream ``test_argmax_*`` / ``test_argmin_*`` example
// scenarios for the given operator. The upstream suite registers each
// scenario twice (a deterministic ``_example`` and a NumPy-seeded
// ``_random`` variant). The random variants only differ in their input
// data and add no behavioural coverage, so — following the convention in
// ``cases_reducesum.cc`` — only the deterministic variants are ported
// here and the ``_example`` suffix is dropped to match the ``test_cc_``
// naming scheme.
void RegisterArgReduceCases(std::vector<TestCase> &registry, const std::string &op_type,
                            kernel::ArgReduce::Mode mode, const std::string &name_prefix) {
  // The upstream example input shared across every variant.
  const std::vector<int64_t> shape = {2, 2};
  const std::vector<float> values = {2.0f, 2.0f, 3.0f, 10.0f};

  // axis=1, keepdims=0
  EmitArgReduceCase(registry, op_type, mode, "test_cc_" + name_prefix + "_no_keepdims", shape,
                    values, /*include_axis=*/true, /*axis=*/1, /*keepdims=*/0,
                    /*select_last_index=*/false);

  // axis=1, keepdims=1
  EmitArgReduceCase(registry, op_type, mode, "test_cc_" + name_prefix + "_keepdims", shape, values,
                    /*include_axis=*/true, /*axis=*/1, /*keepdims=*/1,
                    /*select_last_index=*/false);

  // default axis (omitted -> 0), keepdims=1
  EmitArgReduceCase(registry, op_type, mode, "test_cc_" + name_prefix + "_default_axis", shape,
                    values, /*include_axis=*/false, /*axis=*/0, /*keepdims=*/1,
                    /*select_last_index=*/false);

  // axis=-1, keepdims=1
  EmitArgReduceCase(registry, op_type, mode, "test_cc_" + name_prefix + "_negative_axis_keepdims",
                    shape, values,
                    /*include_axis=*/true, /*axis=*/-1, /*keepdims=*/1,
                    /*select_last_index=*/false);

  // select_last_index variants — same axes/keepdims values as above.
  EmitArgReduceCase(registry, op_type, mode,
                    "test_cc_" + name_prefix + "_no_keepdims_select_last_index", shape, values,
                    /*include_axis=*/true, /*axis=*/1, /*keepdims=*/0,
                    /*select_last_index=*/true);

  EmitArgReduceCase(registry, op_type, mode,
                    "test_cc_" + name_prefix + "_keepdims_select_last_index", shape, values,
                    /*include_axis=*/true, /*axis=*/1, /*keepdims=*/1,
                    /*select_last_index=*/true);

  EmitArgReduceCase(registry, op_type, mode,
                    "test_cc_" + name_prefix + "_default_axis_select_last_index", shape, values,
                    /*include_axis=*/false, /*axis=*/0, /*keepdims=*/1,
                    /*select_last_index=*/true);

  EmitArgReduceCase(registry, op_type, mode,
                    "test_cc_" + name_prefix + "_negative_axis_keepdims_select_last_index", shape,
                    values, /*include_axis=*/true, /*axis=*/-1, /*keepdims=*/1,
                    /*select_last_index=*/true);
}

} // namespace

// ---------------------------------------------------------------------------
// ArgMax / ArgMin — return the indices of the extremum values along a
// single ``axis`` (since opset 13 for the type constraints used here).
// Each operator gets the eight upstream example cases:
//
//   * ``no_keepdims`` — ``axis=1``, ``keepdims=0``.
//   * ``keepdims`` — ``axis=1``, ``keepdims=1``.
//   * ``default_axis`` — ``axis`` attribute omitted (defaults to 0),
//     ``keepdims=1``.
//   * ``negative_axis_keepdims`` — ``axis=-1``, ``keepdims=1``.
//   * The same four variants with ``select_last_index=1`` exercise the
//     tie-breaking introduced in opset 12.
// ---------------------------------------------------------------------------
void RegisterArgMaxCases(std::vector<TestCase> &registry) {
  RegisterArgReduceCases(registry, "ArgMax", kernel::ArgReduce::Mode::kMax, "argmax");
}

void RegisterArgMinCases(std::vector<TestCase> &registry) {
  RegisterArgReduceCases(registry, "ArgMin", kernel::ArgReduce::Mode::kMin, "argmin");
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_kernels/random.h"
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
  const kernel::KernelContext ctx{opset};
  const kernel::ArgReduce arg_kernel{ctx, mode};

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
// ``_random`` variant). Both variants are mirrored here so that the
// substring check in ``test_backend_test_names_onnx_vs_onnxlight`` finds a
// matching ``onnx_light`` case for each upstream name. The ``_random``
// variant uses larger seeded data to mirror the upstream
// ``np.random.randn(3, 4, 5, 6)`` input shape; only the name needs to match
// for the substring check, but the larger shape also adds meaningful
// behavioural coverage beyond the 2x2 example.
void RegisterArgReduceCases(std::vector<TestCase> &registry, const std::string &op_type,
                            kernel::ArgReduce::Mode mode, const std::string &name_prefix) {
  // The upstream example input shared across every ``_example`` variant.
  const std::vector<int64_t> example_shape = {2, 2};
  const std::vector<float> example_values = {2.0f, 2.0f, 3.0f, 10.0f};

  // The upstream random input shape used for every ``_random`` variant; data
  // are generated through the seeded ``Randn`` helper to keep the registry
  // deterministic.
  const std::vector<int64_t> random_shape = {3, 4, 5, 6};
  const std::vector<float> random_values = Randn<float>(random_shape, /*seed=*/91);

  struct Scenario {
    std::string name; // suffix appended after ``_example`` / ``_random``
    bool include_axis;
    int64_t axis;
    int64_t keepdims;
  };

  const std::vector<Scenario> scenarios = {
      {"no_keepdims", true, 1, 0},
      {"keepdims", true, 1, 1},
      {"default_axis", false, 0, 1},
      {"negative_axis_keepdims", true, -1, 1},
  };

  for (const Scenario &s : scenarios) {
    for (bool select_last : {false, true}) {
      const std::string select_suffix = select_last ? "_select_last_index" : "";
      // ``_example`` variant.
      EmitArgReduceCase(registry, op_type, mode,
                        "test_cc_" + name_prefix + "_" + s.name + "_example" + select_suffix,
                        example_shape, example_values, s.include_axis, s.axis, s.keepdims,
                        select_last);
      // ``_random`` variant.
      EmitArgReduceCase(registry, op_type, mode,
                        "test_cc_" + name_prefix + "_" + s.name + "_random" + select_suffix,
                        random_shape, random_values, s.include_axis, s.axis, s.keepdims,
                        select_last);
    }
  }
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

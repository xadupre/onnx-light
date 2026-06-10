// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kDefaultIrVersion = 9;

} // namespace

// ---------------------------------------------------------------------------
// ``Concat → Split → Concat → Relu`` — mirrors the ``test_concat_split``
// example from yet-another-onnx-builder (https://github.com/xadupre/
// yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py).
// Exercises Concat / Split shape propagation when the concat axis dims are
// symbolic (``b + c``) and the Split divides the dim by 2.
//
//   xy = Concat([X, Y],  axis=1)            # (a, b+c)
//   S1, S2 = Split(xy,   axis=1, num_outputs=2)  # (a, (b+c+1)//2), (a, b+c - (b+c+1)//2)
//   zs = Concat([S2, S1], axis=1)           # (a, b+c)
//   Z  = Relu(zs)                           # (a, b+c)
//
// Inputs:
//   X : float[a, b]   Y : float[a, c]
// Output:
//   Z : float[a, e]   (symbolic; the runtime shape is (a, b+c))
//
// The reference DataSet uses concrete sizes ``a=3, b=4, c=6`` so the case
// is executable (``b + c = 10`` is even and divisible by 2 so Split produces
// two equal halves).
// ---------------------------------------------------------------------------
void RegisterConcatSplitShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_concat_split";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  NodeProto &xy_node = AddNode(*graph, "Concat", {"X", "Y"}, {"xy"});
  AddAxisAttribute(xy_node, 1);

  NodeProto &split_node = AddNode(*graph, "Split", {"xy"}, {"S1", "S2"});
  AddAxisAttribute(split_node, 1);
  AddAttribute<int64_t>(split_node, "num_outputs", 2);

  NodeProto &zs_node = AddNode(*graph, "Concat", {"S2", "S1"}, {"zs"});
  AddAxisAttribute(zs_node, 1);

  AddNode(*graph, "Relu", {"zs"}, {"Z"});

  // Graph inputs: X = float[a, b], Y = float[a, c].
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"a", "b"});
  AppendValueInfo(*graph->add_input(), "Y", DataType::FLOAT, {"a", "c"});

  // Intermediate value_info — leave the concat axis dim unannotated; shape
  // inference renders it as a fresh symbolic dim (e.g. ``Concat_axis1``).
  AppendValueInfo(*graph->add_value_info(), "xy", DataType::FLOAT, {"a", DimSpec()});
  AppendValueInfo(*graph->add_value_info(), "S1", DataType::FLOAT, {"a", DimSpec()});
  AppendValueInfo(*graph->add_value_info(), "S2", DataType::FLOAT, {"a", DimSpec()});
  AppendValueInfo(*graph->add_value_info(), "zs", DataType::FLOAT, {"a", DimSpec()});

  // Graph output Z — same shape as zs.
  AppendValueInfo(*graph->add_output(), "Z", DataType::FLOAT, {"a", "e"});

  // Build the reference DataSet — concrete a=3, b=4, c=6 tensors so that
  // b + c = 10 is divisible by 2 and Split produces two (3, 5) halves.
  constexpr int64_t kA = 3;
  constexpr int64_t kB = 4;
  constexpr int64_t kC = 6;
  std::vector<float> x_values(static_cast<size_t>(kA * kB));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f - 1.0f;
  }
  std::vector<float> y_values(static_cast<size_t>(kA * kC));
  for (size_t i = 0; i < y_values.size(); ++i) {
    y_values[i] = static_cast<float>(i) * 0.2f - 0.5f;
  }
  Tensor x = Tensor::FromFloat("X", {kA, kB}, x_values);
  Tensor y = Tensor::FromFloat("Y", {kA, kC}, y_values);

  Tensor xy = kernel::Concat(ctx)({x, y}, /*axis=*/1);
  std::vector<Tensor> splits = kernel::Split(ctx)(xy, /*axis=*/1, /*split=*/{}, /*num_outputs=*/2);
  Tensor zs = kernel::Concat(ctx)({splits[1], splits[0]}, /*axis=*/1);
  Tensor z = kernel::Relu(ctx)(zs);
  z.name = "Z";

  AppendDataSet(tc, {std::move(x), std::move(y)}, {std::move(z)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

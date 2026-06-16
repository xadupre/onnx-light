// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
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
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Resize(scales=[0.5, 0.5]) → Tile(repeats=[2, 2])`` — a two-node model
// whose input is a 2-D FLOAT tensor ``X`` with **symbolic** dimensions
// ``H`` (an odd concrete value) and ``W`` (an even concrete value). The
// model exercises:
//
//  1. Shape inference through ``Resize`` when ``scales`` (FLOAT) is a
//     constant initializer: since the FLOAT data-propagation lattice does
//     not track scale values, the output dims of ``Resize`` are inferred
//     as fresh symbolic names ``Resize_dim0`` / ``Resize_dim1``.
//
//  2. Shape inference through ``Tile`` when the ``repeats`` INT64
//     initializer is data-propagated but the input dims are still
//     symbolic: each output dim becomes ``Tile_dim{i}`` because the
//     product ``Resize_dim{i} * repeats[i]`` cannot be resolved to a
//     concrete integer.
//
// The choice of H=5 (odd) and W=6 (even) is deliberate: ``floor(H * 0.5)``
// and ``floor(W * 0.5)`` differ (2 vs 3), making the concrete test data a
// non-trivial regression guard for the dimension-halving arithmetic.
//
// Graph topology::
//
//   X [H, W]
//     → Resize(X, roi="", scales=[0.5, 0.5], mode=nearest, asymmetric)
//     → resized_out [Resize_dim0, Resize_dim1]
//     → Tile(resized_out, repeats=[2, 2])
//     → output [Tile_dim0, Tile_dim1]
//
// Concrete shapes (H=5, W=6)::
//
//   X            float[5, 6]
//   resized_out  float[2, 3]   (floor(5*0.5)=2, floor(6*0.5)=3)
//   output       float[4, 6]   (2*2=4, 3*2=6)
// ---------------------------------------------------------------------------
void RegisterResizeTileShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_resize_tile";

  TestCase tc(name, name, "model", "inference", 1e-7, 1e-3);

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Resize(X, roi=<absent>, scales) → resized_out
  // asymmetric coordinate mode: x_original = x_resized / 0.5 = x_resized * 2.
  NodeProto &resize_node = AddNode(*graph, "Resize", {"X", "", "scales"}, {"resized_out"});
  AddAttribute<std::string>(resize_node, "mode", "nearest");
  AddAttribute<std::string>(resize_node, "coordinate_transformation_mode", "asymmetric");

  // Tile(resized_out, repeats) → output
  AddNode(*graph, "Tile", {"resized_out", "repeats"}, {"output"});

  // Initializers: ``scales`` (FLOAT) and ``repeats`` (INT64) are constant
  // inputs that the graph does not expose as user-overridable graph inputs.
  AddInitializer<float>(*graph, "scales", {2}, {0.5f, 0.5f});
  AddInitializer<int64_t>(*graph, "repeats", {2}, {int64_t{2}, int64_t{2}});

  // Graph input X: float[H, W] with symbolic dim names.
  // The concrete test data uses H=5 (odd) and W=6 (even).
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"H", "W"});

  // Intermediate value_info: resized_out inferred as [Resize_dim0, Resize_dim1].
  // ComputeShapeResize assigns a fresh ``Resize_dim{i}`` symbol for each axis
  // when the scales input is a FLOAT initializer (its values are not tracked
  // by the integer data-propagation lattice).
  AppendValueInfo(*graph->add_value_info(), "resized_out", DataType::FLOAT,
                  {"Resize_dim0", "Resize_dim1"});

  // Graph output: Tile result inferred as [Tile_dim0, Tile_dim1].
  // ComputeShapeTile emits ``Tile_dim{i}`` for each axis when the input dim
  // is symbolic (even if the repeats values are known), because the product
  // ``Resize_dim{i} * repeats[i]`` is not an integer.
  AppendValueInfo(*graph->add_output(), "output", DataType::FLOAT, {"Tile_dim0", "Tile_dim1"});

  // Reference DataSet — concrete H=5 (odd), W=6 (even).
  constexpr int64_t kH = 5; // odd
  constexpr int64_t kW = 6; // even

  std::vector<float> x_values(static_cast<size_t>(kH * kW));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) + 1.0f;
  }
  Tensor x = Tensor::FromFloat("X", {kH, kW}, x_values);

  // Resize with scales=[0.5, 0.5], asymmetric + nearest:
  //   floor(H * 0.5) = 2, floor(W * 0.5) = 3  →  resized_out shape [2, 3].
  const Tensor scales_tensor = Tensor::FromFloat("", {2}, {0.5f, 0.5f});
  kernel::Resize::Attributes resize_attrs;
  resize_attrs.mode = "nearest";
  resize_attrs.coordinate_transformation_mode = "asymmetric";
  Tensor resized_out = kernel::Resize{ctx}(x, scales_tensor, resize_attrs);
  resized_out.name = "resized_out";

  // Tile with repeats=[2, 2]:  [2, 3] → [4, 6].
  const Tensor repeats_tensor = Tensor::FromInt64("", {2}, {int64_t{2}, int64_t{2}});
  Tensor output = kernel::Tile{ctx}(resized_out, repeats_tensor);
  output.name = "output";

  AppendDataSet(tc, {std::move(x)}, {std::move(output)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

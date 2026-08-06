// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Resize(scales=[0.5, 0.5]) → Tile(repeats=[2, 2]) → Max(.., 0)``
//
// Input ``X`` is a 2-D FLOAT tensor with **symbolic** dimensions
// ``H`` (even, concrete value = 10) and ``2*h`` (concrete value = 6).
// The model exercises:
//
//  1. Shape inference through ``Resize`` when ``scales`` (FLOAT) is a
//     constant initializer with a uniform scale of 0.5: the output dims
//     are inferred symbolically as ``H//2`` and ``h`` by applying integer
//     floor-division arithmetic on the input dim expressions.
//
//  2. Shape inference through ``Tile`` when the ``repeats`` INT64
//     initializer is data-propagated: even though the input dims are
//     symbolic (``H//2``, ``h``), the output dims are computed symbolically
//     as ``2*(H//2)`` and ``2*h`` using the expressions library.
//
//  3. Shape inference through ``Max`` with a scalar initializer ``0.0``:
//     broadcasting preserves the tile output shape ``[2*(H//2), 2*h]``.
//
// Graph topology::
//
//   X [H, 2*h]
//     → Resize(X, roi="", scales=[0.5, 0.5], mode=nearest, asymmetric)
//     → resized_out [H//2, h]
//     → Tile(resized_out, repeats=[2, 2])
//     → tile_out [2*(H//2), 2*h]
//     → Max(tile_out, zeros_scalar)
//     → output [2*(H//2), 2*h]
//
// Concrete shapes (H=10, h=3 → W=2*h=6)::
//
//   X            float[10, 6]
//   resized_out  float[5, 3]   (floor(10*0.5)=5, floor(6*0.5)=3)
//   tile_out     float[10, 6]  (5*2=10, 3*2=6)
//   output       float[10, 6]  (Max with 0 is identity for positive values)
// ---------------------------------------------------------------------------
void RegisterResizeTileShapeInferenceCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const onnx_kernels::kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_resize_tile";

  TestCase tc(name, name, "model", "inference", 1e-7, 1e-3);

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Resize(X, roi=<absent>, scales) → resized_out
  // asymmetric coordinate mode: x_original = x_resized / 0.5 = x_resized * 2.
  NodeProto &resize_node = AddNode(*graph, "Resize", {"X", "", "scales"}, {"resized_out"});
  AddAttribute<std::string>(resize_node, "mode", "nearest");
  AddAttribute<std::string>(resize_node, "coordinate_transformation_mode", "asymmetric");

  // Tile(resized_out, repeats) → tile_out
  AddNode(*graph, "Tile", {"resized_out", "repeats"}, {"tile_out"});

  // Max(tile_out, zeros_scalar) → output
  // The scalar ``zeros_scalar`` broadcasts to the shape of ``tile_out`` and
  // the Max is an identity for non-negative inputs.
  AddNode(*graph, "Max", {"tile_out", "zeros_scalar"}, {"output"});

  // Initializers: ``scales`` (FLOAT), ``repeats`` (INT64), and
  // ``zeros_scalar`` (FLOAT scalar 0.0).
  AddInitializer<float>(*graph, "scales", {2}, {0.5f, 0.5f});
  AddInitializer<int64_t>(*graph, "repeats", {2}, {int64_t{2}, int64_t{2}});
  AddInitializer<float>(*graph, "zeros_scalar", {}, {0.0f});

  // Graph input X: float[H, 2*h] with symbolic dim names.
  // The concrete test data uses H=10 (even) and h=3 (so W = 2*h = 6).
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"H", "2*h"});

  // Intermediate value_info: resized_out inferred as [H//2, h].
  // ComputeShapeResize detects the uniform float scale 0.5 (min==max==0.5)
  // and applies dim_div: dim_div("H", 2) → "H//2", dim_div("2*h", 2) → "h".
  AppendValueInfo(*graph->add_value_info(), "resized_out", DataType::FLOAT, {"H//2", "h"});

  // Intermediate value_info: tile_out inferred as [2*(H//2), 2*h].
  // ComputeShapeTile applies dim_mul when the repeat count is known and the
  // input dim is symbolic: dim_mul("H//2", 2) → "2*(H//2)", dim_mul("h", 2) → "2*h".
  AppendValueInfo(*graph->add_value_info(), "tile_out", DataType::FLOAT, {"2*(H//2)", "2*h"});

  // Graph output: Max result with shape preserved from tile_out.
  // ComputeShapeElementWiseBroadcast propagates the shape [2*(H//2), 2*h]
  // when broadcasting with the scalar zeros_scalar.
  AppendValueInfo(*graph->add_output(), "output", DataType::FLOAT, {"2*(H//2)", "2*h"});

  // Reference DataSet — concrete H=10 (even), h=3, W=2*h=6.
  constexpr int64_t kH = 10;     // even, represents symbolic "H"
  constexpr int64_t kh = 3;      // represents symbolic "h"
  constexpr int64_t kW = 2 * kh; // = 6, represents symbolic "2*h"

  std::vector<float> x_values(static_cast<size_t>(kH * kW));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) + 1.0f;
  }
  Tensor x = Tensor::FromFloat("X", {kH, kW}, x_values);

  // Resize with scales=[0.5, 0.5], asymmetric + nearest:
  //   floor(H * 0.5) = 5, floor(W * 0.5) = 3  →  resized_out shape [5, 3].
  const Tensor scales_tensor = Tensor::FromFloat("", {2}, {0.5f, 0.5f});
  onnx_kernels::kernel::Resize::Attributes resize_attrs;
  resize_attrs.mode = "nearest";
  resize_attrs.coordinate_transformation_mode = "asymmetric";
  Tensor resized_out = onnx_kernels::kernel::Resize{ctx}(x, scales_tensor, resize_attrs);
  resized_out.name = "resized_out";

  // Tile with repeats=[2, 2]:  [5, 3] → [10, 6].
  const Tensor repeats_tensor = Tensor::FromInt64("", {2}, {int64_t{2}, int64_t{2}});
  Tensor tile_out = onnx_kernels::kernel::Tile{ctx}(resized_out, repeats_tensor);
  tile_out.name = "tile_out";

  // Max(tile_out, zeros_scalar): all x_values > 0 so Max is identity → [10, 6].
  const Tensor zeros_scalar = Tensor::FromFloat("zeros_scalar", {}, {0.0f});
  const onnx_kernels::kernel::Max max_kernel{ctx};
  Tensor output = max_kernel({tile_out, zeros_scalar});
  output.name = "output";

  AppendDataSet(tc, {std::move(x)}, {std::move(output)});

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"
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
// ``Pad(reflect) → Conv(canny) → Sub(ReduceMean)`` — an image-processing
// pipeline expressed with **symbolic** spatial dimensions so that the
// shape-inference pass has to propagate symbolic dims through every stage.
//
// The single input ``X`` is a grayscale image batch ``float[N, 1, H, W]``
// where ``N``, ``H`` and ``W`` are symbolic. The model:
//
//  1. **Pads** the image by one pixel on every spatial side
//     (``pads = [0, 0, 1, 1, 0, 0, 1, 1]``) using ``reflect`` mode. Because
//     the spatial dims are symbolic, ``ComputeShapePad`` propagates them as
//     symbolic **expressions** ``H+2`` / ``W+2`` (input dim plus the per-axis
//     pad), keeping the unpadded ``N`` and ``1`` dims untouched.
//
//  2. Applies a **Canny-style edge filter** as a 3×3 ``Conv`` with the
//     discrete Laplacian kernel (``[[0,-1,0],[-1,4,-1],[0,-1,0]]``) and no
//     padding. A 3×3 VALID convolution shrinks each spatial dim by 2, so
//     ``ComputeShapeConv`` evaluates the spatial formula symbolically and the
//     ``H+2`` / ``W+2`` expressions collapse back to ``H`` / ``W``; the output
//     channel count ``M = 1`` comes from the (concrete) weight initializer.
//
//  3. **Removes the average** by subtracting the global mean: a ``ReduceMean``
//     over every axis (``keepdims = 1``) yields a ``[1, 1, 1, 1]`` tensor
//     that ``Sub`` broadcasts back to the filtered shape.
//
// Graph topology::
//
//   X [N, 1, H, W]
//     → Pad(X, pads=[0,0,1,1,0,0,1,1], mode="reflect")
//     → padded   [N, 1, H+2, W+2]
//     → Conv(padded, W, kernel_shape=[3,3], pads=[0,0,0,0])
//     → filtered [N, 1, H, W]
//     → ReduceMean(filtered, axes=[0,1,2,3], keepdims=1)
//     → avg      [1, 1, 1, 1]
//     → Sub(filtered, avg)
//     → Y        [N, 1, H, W]
//
// Concrete shapes (N=2, H=5, W=7)::
//
//   X         float[2, 1, 5, 7]
//   padded    float[2, 1, 7, 9]   (5+2, 7+2)
//   filtered  float[2, 1, 5, 7]   (7-2, 9-2)
//   avg       float[1, 1, 1, 1]
//   Y         float[2, 1, 5, 7]
// ---------------------------------------------------------------------------
void RegisterPadCannyAverageShapeInferenceCases(std::vector<TestCase> &registry,
                                                TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const onnx_kernels::kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_pad_canny_average";

  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE, 1e-7, 1e-3);

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Pad(X, pads, mode="reflect") → padded
  NodeProto &pad_node = AddNode(*graph, "Pad", {"X", "pads"}, {"padded"});
  AddAttribute<std::string>(pad_node, "mode", "reflect");

  // Conv(padded, W) → filtered. The Laplacian weight acts as a Canny-style
  // edge filter; the 3×3 kernel without padding shrinks each spatial dim by 2.
  NodeProto &conv_node = AddNode(*graph, "Conv", {"padded", "W"}, {"filtered"});
  AddAttribute<std::vector<int64_t>>(conv_node, "kernel_shape", {int64_t{3}, int64_t{3}});
  AddAttribute<std::vector<int64_t>>(conv_node, "pads",
                                     {int64_t{0}, int64_t{0}, int64_t{0}, int64_t{0}});

  // ReduceMean(filtered, axes=[0,1,2,3], keepdims=1) → avg ([1, 1, 1, 1]).
  NodeProto &mean_node = AddNode(*graph, "ReduceMean", {"filtered", "axes_mean"}, {"avg"});
  AddAttribute<int64_t>(mean_node, "keepdims", 1);

  // Sub(filtered, avg) → Y (broadcasts the scalar mean back to filtered shape).
  AddNode(*graph, "Sub", {"filtered", "avg"}, {"Y"});

  // Initializers: the Laplacian convolution weight ``W`` (M=1, C=1, 3×3), the
  // INT64 ``pads`` and the INT64 ``axes_mean`` data-propagated through shape
  // inference.
  AddInitializer<float>(*graph, "W", {1, 1, 3, 3},
                        {0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f});
  AddInitializer<int64_t>(*graph, "pads", {8},
                          {int64_t{0}, int64_t{0}, int64_t{1}, int64_t{1}, int64_t{0}, int64_t{0},
                           int64_t{1}, int64_t{1}});
  AddInitializer<int64_t>(*graph, "axes_mean", {4},
                          {int64_t{0}, int64_t{1}, int64_t{2}, int64_t{3}});

  // Graph input X: float[N, 1, H, W] with symbolic spatial dims. The channel
  // dim is the concrete grayscale channel (1).
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT,
                  {DimSpec("N"), DimSpec(int64_t{1}), DimSpec("H"), DimSpec("W")});

  // Intermediate value_info entries (ordered alphabetically to match the
  // ordering returned by ``infer_shapes_model``).
  AppendValueInfo(
      *graph->add_value_info(), "avg", DataType::FLOAT,
      {DimSpec(int64_t{1}), DimSpec(int64_t{1}), DimSpec(int64_t{1}), DimSpec(int64_t{1})});
  AppendValueInfo(*graph->add_value_info(), "filtered", DataType::FLOAT,
                  {DimSpec("N"), DimSpec(int64_t{1}), DimSpec("H"), DimSpec("W")});
  AppendValueInfo(*graph->add_value_info(), "padded", DataType::FLOAT,
                  {DimSpec("N"), DimSpec(int64_t{1}), DimSpec("H+2"), DimSpec("W+2")});

  // Graph output Y — same symbolic dims as filtered.
  AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT,
                  {DimSpec("N"), DimSpec(int64_t{1}), DimSpec("H"), DimSpec("W")});

  // Reference DataSet — concrete N=2, H=5, W=7.
  constexpr int64_t kN = 2;
  constexpr int64_t kC = 1;
  constexpr int64_t kH = 5;
  constexpr int64_t kW = 7;

  std::vector<float> x_values(static_cast<size_t>(kN * kC * kH * kW));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) + 1.0f;
  }
  Tensor x_tensor = Tensor::FromFloat("X", {kN, kC, kH, kW}, x_values);

  // Pad(reflect) by one pixel on every spatial side: [2, 1, 5, 7] → [2, 1, 7, 9].
  const Tensor pads_tensor = Tensor::FromInt64("", {8},
                                               {int64_t{0}, int64_t{0}, int64_t{1}, int64_t{1},
                                                int64_t{0}, int64_t{0}, int64_t{1}, int64_t{1}});
  Tensor padded = onnx_kernels::kernel::Pad{ctx}(x_tensor, pads_tensor, /*constant_value=*/nullptr,
                                                 /*axes=*/nullptr, "reflect");
  padded.name = "padded";

  // Conv with the 3×3 Laplacian kernel, no padding: [2, 1, 7, 9] → [2, 1, 5, 7].
  const Tensor w_tensor = Tensor::FromFloat(
      "W", {1, 1, 3, 3}, {0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f});
  const Tensor bias; // empty → optional bias absent
  onnx_kernels::kernel::Conv::Attributes conv_attrs;
  conv_attrs.kernel_shape = {3, 3};
  conv_attrs.pads = {0, 0, 0, 0};
  Tensor filtered = onnx_kernels::kernel::Conv{ctx}(padded, w_tensor, bias, conv_attrs);
  filtered.name = "filtered";

  // ReduceMean over every axis (keepdims): [2, 1, 5, 7] → [1, 1, 1, 1].
  const Tensor axes_tensor =
      Tensor::FromInt64("", {4}, {int64_t{0}, int64_t{1}, int64_t{2}, int64_t{3}});
  Tensor avg = onnx_kernels::kernel::ReduceMean{ctx}(filtered, axes_tensor, /*keepdims=*/true,
                                                     /*noop_with_empty_axes=*/false);
  avg.name = "avg";

  // Sub(filtered, avg): broadcasts the scalar mean back to the filtered shape.
  Tensor y = onnx_kernels::kernel::Sub{ctx}(filtered, avg);
  y.name = "Y";

  AppendDataSet(tc, {std::move(x_tensor)}, {std::move(y)});

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

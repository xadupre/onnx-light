// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
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
// ``Unsqueeze → Unsqueeze → Sub → Mul → ReduceSum → Sqrt → TopK → ReduceMean``
// — computes the pairwise Euclidean distance matrix of an input ``X`` of
// shape ``[N, D]``, keeps the ``k`` largest distances of every row and
// finally averages them. Both the input dims **and** the TopK ``k`` are
// symbolic at shape-inference time:
//
//   * ``X`` is declared with the symbolic dims ``[N, D]`` so the
//     shape-inference pass has to propagate symbols through broadcasting
//     (``Sub``), the reduction (``ReduceSum``) and the element-wise ops.
//   * ``K`` is a **model input** (INT64 ``[1]``), not an initializer, so its
//     *value* is unknown when shapes are inferred. ``TopK`` therefore cannot
//     resolve its output axis and must emit a fresh symbolic dim for it,
//     which ``ReduceMean`` then reduces away.
//
// Graph topology::
//
//   X [N, D]
//     ├── Unsqueeze(axes=[1]) ──► x_rows [N, 1, D]
//     └── Unsqueeze(axes=[0]) ──► x_cols [1, N, D]
//     Sub(x_rows, x_cols)       ──► diff   [N, N, D]
//     Mul(diff, diff)           ──► sq     [N, N, D]
//     ReduceSum(sq, axes=[-1], keepdims=0) ──► sum_sq [N, N]
//     Sqrt(sum_sq)              ──► dist   [N, N]
//     TopK(dist, K, axis=-1)    ──► topk_values [N, k], topk_indices [N, k]
//     ReduceMean(topk_values, axes=[-1], keepdims=0) ──► Y [N]
//
// Concrete shapes (N=3, D=3, k=2)::
//
//   X            float[3, 3]
//   dist         float[3, 3]
//   topk_values  float[3, 2]
//   Y            float[3]
//
// The reference DataSet uses a 3×3 input whose rows lie on the axes of an
// integer right-triangle grid, so the pairwise distances are integers
// (``[[0, 3, 4], [3, 0, 5], [4, 5, 0]]``); keeping the ``k = 2`` largest of
// each row and averaging them yields ``Y = [3.5, 4.0, 4.5]``.
// ---------------------------------------------------------------------------
void RegisterTopKPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);

  const std::string name = "test_cc_shape_inference_topk_pairwise_distance";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Pairwise distance via broadcasting. ``x_rows`` is ``[N, 1, D]`` and
  // ``x_cols`` is ``[1, N, D]``; their difference broadcasts to ``[N, N, D]``.
  AddNode(*graph, "Unsqueeze", {"X", "axes_row"}, {"x_rows"});
  AddNode(*graph, "Unsqueeze", {"X", "axes_col"}, {"x_cols"});
  AddNode(*graph, "Sub", {"x_rows", "x_cols"}, {"diff"});
  AddNode(*graph, "Mul", {"diff", "diff"}, {"sq"});
  NodeProto &reduce_sum = AddNode(*graph, "ReduceSum", {"sq", "reduce_axes"}, {"sum_sq"});
  AddAttribute<int64_t>(reduce_sum, "keepdims", 0);
  AddNode(*graph, "Sqrt", {"sum_sq"}, {"dist"});

  // TopK with ``K`` taken from a model input — its value is unknown to shape
  // inference, so the ``topk_values`` / ``topk_indices`` output axis is
  // symbolic.
  NodeProto &topk = AddNode(*graph, "TopK", {"dist", "K"}, {"topk_values", "topk_indices"});
  AddAxisAttribute(topk, -1);
  AddAttribute<int64_t>(topk, "largest", 1);
  AddAttribute<int64_t>(topk, "sorted", 1);

  // ReduceMean over the (symbolic) TopK axis collapses it away, so ``Y`` is a
  // concrete-rank ``[N]`` vector once the input dims are known.
  NodeProto &reduce_mean = AddNode(*graph, "ReduceMean", {"topk_values", "mean_axes"}, {"Y"});
  AddAttribute<int64_t>(reduce_mean, "keepdims", 0);

  // Initializers — the ``axes`` vectors referenced by ``Unsqueeze`` /
  // ``ReduceSum`` / ``ReduceMean``. ``K`` is deliberately *not* an
  // initializer; it is a graph input below.
  AddInitializerShape(*graph, "axes_row", {1});
  AddInitializerShape(*graph, "axes_col", {0});
  AddInitializerShape(*graph, "reduce_axes", {-1});
  AddInitializerShape(*graph, "mean_axes", {-1});

  // Graph inputs. ``X`` carries symbolic dims ``[N, D]``; ``K`` is the INT64
  // ``[1]`` number of top distances to keep.
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec("N"), DimSpec("D")});
  AppendValueInfo(*graph->add_input(), "K", DataType::INT64, {DimSpec(int64_t{1})});

  // Intermediate value_info entries. ``dist`` is the symbolic ``[N, N]``
  // distance matrix; ``topk_values`` keeps a symbolic ``k`` as its trailing
  // axis because ``K`` is a runtime input.
  AppendValueInfo(*graph->add_value_info(), "dist", DataType::FLOAT, {DimSpec("N"), DimSpec("N")});
  AppendValueInfo(*graph->add_value_info(), "topk_values", DataType::FLOAT,
                  {DimSpec("N"), DimSpec("k")});

  // Output Y — the per-row mean of the ``k`` largest distances, shape ``[N]``.
  AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec("N")});

  // Reference DataSet with a concrete ``[3, 3]`` input and ``k = 2``. Rows
  // lie on the axes of an integer grid so the pairwise distances are exact
  // integers: ``[[0, 3, 4], [3, 0, 5], [4, 5, 0]]``. Keeping the two largest
  // distances of each row and averaging gives ``[3.5, 4.0, 4.5]``.
  Tensor x = Tensor::FromFloat("X", {3, 3},
                               {0.0f, 0.0f, 0.0f, //
                                3.0f, 0.0f, 0.0f, //
                                0.0f, 4.0f, 0.0f});
  Tensor k = Tensor::FromInt64("K", {1}, {int64_t{2}});
  Tensor y = Tensor::FromFloat("Y", {3}, {3.5f, 4.0f, 4.5f});
  AppendDataSet(tc, {std::move(x), std::move(k)}, {std::move(y)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

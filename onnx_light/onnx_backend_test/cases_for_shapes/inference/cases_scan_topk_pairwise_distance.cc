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

// Builds the body subgraph for the pairwise-distance ``Scan`` case.
//
// Inputs : state_X (FLOAT [N, D] — the full matrix carried unchanged as a
//                    Scan state variable)
//          x_row   (FLOAT [D]    — one row of the outer-scope scan input X)
// Outputs: state_X_out (FLOAT [N, D] — state, propagated unchanged)
//          dist_sq     (FLOAT [N]    — squared Euclidean distance from
//                                       ``x_row`` to every row of ``state_X``)
//
// Body topology::
//
//   state_X_out = Identity(state_X)               // [N, D] state pass-through
//   diff        = Sub(state_X, x_row)             // broadcasts to [N, D]
//   sq          = Mul(diff, diff)                 // [N, D]
//   dist_sq     = ReduceSum(sq, axes=[-1], keepdims=0)  // [N]
//
// The ``reduce_axes`` initializer consumed by ``ReduceSum`` is declared in the
// outer graph and referenced from the body via outer-scope lookup. Stacking
// the ``dist_sq`` scan output across the ``N`` scan iterations yields the
// squared pairwise distance matrix of shape ``[N, N]``.
GraphProto BuildPairwiseDistanceScanBody() {
  GraphProto g;
  g.set_name("pairwise_distance_scan_body");

  AppendValueInfo(*g.add_input(), "state_X", TensorProto::DataType::FLOAT,
                  {DimSpec("N"), DimSpec("D")});
  AppendValueInfo(*g.add_input(), "x_row", TensorProto::DataType::FLOAT, {DimSpec("D")});

  AddNode(g, "Identity", {"state_X"}, {"state_X_out"});
  AddNode(g, "Sub", {"state_X", "x_row"}, {"diff"});
  AddNode(g, "Mul", {"diff", "diff"}, {"sq"});
  NodeProto &reduce = AddNode(g, "ReduceSum", {"sq", "reduce_axes"}, {"dist_sq"});
  AddAttribute<int64_t>(reduce, "keepdims", 0);

  AppendValueInfo(*g.add_output(), "state_X_out", TensorProto::DataType::FLOAT,
                  {DimSpec("N"), DimSpec("D")});
  AppendValueInfo(*g.add_output(), "dist_sq", TensorProto::DataType::FLOAT, {DimSpec("N")});
  return g;
}

} // namespace

// ---------------------------------------------------------------------------
// ``Scan(body=pairwise_distance_scan_body) → Sqrt → TopK → ReduceMean`` —
// computes the pairwise Euclidean distance matrix of an input ``X`` of shape
// ``[N, D]`` via a ``Scan`` node that scans ``X`` along axis 0 (``X`` is also
// carried as the Scan state so the body can broadcast every row against the
// full matrix), then keeps the ``k`` largest distances of every row and
// averages them. Both the input dims **and** the TopK ``k`` are symbolic at
// shape-inference time:
//
//   * ``X`` is declared with the symbolic dims ``[N, D]``. The Scan trip count
//     is taken from ``X``'s scan axis (axis 0 = ``N``), so the stacked
//     distance matrix is ``[N, N]``.
//   * ``K`` is a **model input** (INT64 ``[1]``), not an initializer, so
//     ``TopK`` cannot resolve its output axis and must emit a fresh symbolic
//     dim for it, which ``ReduceMean`` then reduces away.
//
// Graph topology::
//
//   X [N, D]
//     Scan(X_state=X, X_scan=X, num_scan_inputs=1, body)
//         ──► state_final [N, D], dist_sq [N, N]
//     Sqrt(dist_sq)            ──► dist [N, N]
//     TopK(dist, K, axis=-1)   ──► topk_values [N, k], topk_indices
//     ReduceMean(topk_values, axes=[-1], keepdims=0) ──► Y [N]
//
// The reference DataSet uses a 3×3 input whose rows lie on the axes of an
// integer right-triangle grid, so the pairwise distances are integers
// (``[[0, 3, 4], [3, 0, 5], [4, 5, 0]]``); keeping the ``k = 2`` largest of
// each row and averaging them yields ``Y = [3.5, 4.0, 4.5]``.
// ---------------------------------------------------------------------------
void RegisterScanTopKPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);

  const std::string name = "test_cc_shape_inference_scan_topk_pairwise_distance";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Scan: 1 state input (X) + 1 scan input (X), num_scan_inputs=1. Outputs:
  // state_final (the unchanged matrix) and the stacked squared-distance matrix.
  NodeProto &scan_node = AddNode(*graph, "Scan", {"X", "X"}, {"state_final", "dist_sq"});
  AddAttribute<int64_t>(scan_node, "num_scan_inputs", 1);
  AttributeProto *body_attr = scan_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildPairwiseDistanceScanBody();

  // Turn the squared distances into Euclidean distances.
  AddNode(*graph, "Sqrt", {"dist_sq"}, {"dist"});

  // TopK with ``K`` taken from a model input — its value is unknown to shape
  // inference, so the ``topk_values`` / ``topk_indices`` output axis is
  // symbolic.
  NodeProto &topk = AddNode(*graph, "TopK", {"dist", "K"}, {"topk_values", "topk_indices"});
  AddAxisAttribute(topk, -1);
  AddAttribute<int64_t>(topk, "largest", 1);
  AddAttribute<int64_t>(topk, "sorted", 1);

  // ReduceMean over the (symbolic) TopK axis collapses it away, so ``Y`` is a
  // rank-1 ``[N]`` vector once the input dims are known.
  NodeProto &reduce_mean = AddNode(*graph, "ReduceMean", {"topk_values", "mean_axes"}, {"Y"});
  AddAttribute<int64_t>(reduce_mean, "keepdims", 0);

  // Initializers — the ``axes`` vectors referenced by ``ReduceSum`` (from the
  // Scan body via outer-scope lookup) and ``ReduceMean``. ``K`` is
  // deliberately *not* an initializer; it is a graph input below.
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

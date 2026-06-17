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

// Returns the bytes of a BOOL scalar with value ``v`` in the layout used by
// :cpp:class:`TensorProto::raw_data` (one byte per element).
std::vector<uint8_t> BoolBytes(bool v) { return std::vector<uint8_t>{v ? uint8_t{1} : uint8_t{0}}; }

// Adds a graph input named ``name`` typed as a scalar tensor of dtype
// ``dtype`` (empty shape).
void AddBodyInputScalar(GraphProto &g, const char *name, TensorProto::DataType dtype) {
  ValueInfoProto *vi = g.add_input();
  vi->set_name(name);
  TypeProto::Tensor *tt = vi->ref_type().mutable_tensor_type();
  tt->set_elem_type(static_cast<int>(dtype));
  tt->mutable_shape();
}

// Adds a graph output named ``name`` typed as a tensor of dtype ``dtype``
// with the symbolic dim names from ``dims`` (an empty string yields an
// unnamed dim).
void AddBodyOutputTyped(GraphProto &g, const char *name, TensorProto::DataType dtype,
                        const std::vector<std::string> &dims) {
  ValueInfoProto *vi = g.add_output();
  vi->set_name(name);
  TypeProto::Tensor *tt = vi->ref_type().mutable_tensor_type();
  tt->set_elem_type(static_cast<int>(dtype));
  TensorShapeProto *shape = tt->mutable_shape();
  for (const std::string &p : dims) {
    if (p.empty()) {
      shape->add_dim();
    } else {
      shape->add_dim()->set_dim_param(utils::String(p));
    }
  }
}

// Builds the body subgraph for the pairwise-distance ``Loop`` case.
//
// Inputs : iter_count (INT64 scalar), cond_in (BOOL scalar)
// Outputs: cond_out (BOOL scalar), scan_out (FLOAT [N])
//
// At iteration ``i`` the body looks up the ``i``-th row of the outer-scope
// input ``X`` (FLOAT ``[N, D]``) and emits the Euclidean distance from that
// row to every row of ``X`` as a FLOAT ``[N]`` scan output. Stacking the
// ``N`` scan outputs across the ``N`` iterations produces the pairwise
// distance matrix of shape ``[N, N]``.
GraphProto BuildPairwiseDistanceBody() {
  GraphProto g;
  g.set_name("pairwise_distance_body");

  AddBodyInputScalar(g, "iter_count", TensorProto::DataType::INT64);
  AddBodyInputScalar(g, "cond_in", TensorProto::DataType::BOOL);

  AddNode(g, "Identity", {"cond_in"}, {"cond_out"});
  // iter_1d = Unsqueeze(iter_count, axes=[0]) — INT64 [1] index for Gather.
  AddNode(g, "Unsqueeze", {"iter_count", "unsqueeze_axes"}, {"iter_1d"});
  // x_i = Gather(X, iter_1d, axis=0) — FLOAT [1, D]. ``X`` is resolved from
  // the outer (main-graph) scope.
  NodeProto &gather = AddNode(g, "Gather", {"X", "iter_1d"}, {"x_i"});
  AddAxisAttribute(gather, 0);
  // diff = Sub(X, x_i) — FLOAT [N, D] via broadcasting on the leading axis.
  AddNode(g, "Sub", {"X", "x_i"}, {"diff"});
  AddNode(g, "Mul", {"diff", "diff"}, {"sq"});
  // sum_sq = ReduceSum(sq, axes=[-1], keepdims=0) — FLOAT [N].
  NodeProto &reduce = AddNode(g, "ReduceSum", {"sq", "reduce_axes"}, {"sum_sq"});
  AddAttribute<int64_t>(reduce, "keepdims", 0);
  AddNode(g, "Sqrt", {"sum_sq"}, {"dist_row"});
  AddNode(g, "Identity", {"dist_row"}, {"scan_out"});

  AddBodyOutputTyped(g, "cond_out", TensorProto::DataType::BOOL, /*dims=*/{});
  AddBodyOutputTyped(g, "scan_out", TensorProto::DataType::FLOAT, /*dims=*/{"N"});
  return g;
}

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Gather → Loop(body=pairwise_distance_body) → TopK → ReduceMean`` —
// computes the pairwise Euclidean distance matrix of an input ``X`` of shape
// ``[N, D]`` via a ``Loop`` node (one row of the ``[N, N]`` matrix per
// iteration), then keeps the ``k`` largest distances of every row and averages
// them. Both the input dims **and** the TopK ``k`` are symbolic at
// shape-inference time:
//
//   * ``X`` is declared with the symbolic dims ``[N, D]``.
//   * The ``Loop`` trip count is taken from ``Shape(X)[0]`` (a runtime INT64
//     value), so the stacked distance matrix has a symbolic leading axis.
//   * ``K`` is a **model input** (INT64 ``[1]``), not an initializer, so
//     ``TopK`` cannot resolve its output axis and must emit a fresh symbolic
//     dim for it, which ``ReduceMean`` then reduces away.
//
// Graph topology::
//
//   X [N, D]
//     ├── Shape ──► shape_X [2]
//     │              └── Gather(axis=0, indices=[0]) ──► trip_count [1]
//     Loop(trip_count, cond_init, body) ──► dist [loop, N]
//     TopK(dist, K, axis=-1)            ──► topk_values [loop, k], topk_indices
//     ReduceMean(topk_values, axes=[-1], keepdims=0) ──► Y [loop]
//
// The reference DataSet uses a 3×3 input whose rows lie on the axes of an
// integer right-triangle grid, so the pairwise distances are integers
// (``[[0, 3, 4], [3, 0, 5], [4, 5, 0]]``); keeping the ``k = 2`` largest of
// each row and averaging them yields ``Y = [3.5, 4.0, 4.5]``.
// ---------------------------------------------------------------------------
void RegisterLoopTopKPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);

  const std::string name = "test_cc_shape_inference_loop_topk_pairwise_distance";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Main-graph nodes: derive the trip count from ``X``'s leading dim.
  AddNode(*graph, "Shape", {"X"}, {"shape_X"});
  NodeProto &gather_trip = AddNode(*graph, "Gather", {"shape_X", "zero_idx"}, {"trip_count"});
  AddAxisAttribute(gather_trip, 0);

  NodeProto &loop_node = AddNode(*graph, "Loop", {"trip_count", "cond_init"}, {"dist"});
  AttributeProto *body_attr = loop_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildPairwiseDistanceBody();

  // TopK with ``K`` taken from a model input — its value is unknown to shape
  // inference, so the ``topk_values`` / ``topk_indices`` output axis is
  // symbolic.
  NodeProto &topk = AddNode(*graph, "TopK", {"dist", "K"}, {"topk_values", "topk_indices"});
  AddAxisAttribute(topk, -1);
  AddAttribute<int64_t>(topk, "largest", 1);
  AddAttribute<int64_t>(topk, "sorted", 1);

  // ReduceMean over the (symbolic) TopK axis collapses it away, so ``Y`` is a
  // rank-1 ``[loop]`` vector.
  NodeProto &reduce_mean = AddNode(*graph, "ReduceMean", {"topk_values", "mean_axes"}, {"Y"});
  AddAttribute<int64_t>(reduce_mean, "keepdims", 0);

  // Initializers — index ``[0]`` for the trip-count Gather, the BOOL ``cond``
  // constant, the axes vectors referenced from the body via outer-scope
  // lookup, and the ``axes`` vector consumed by ``ReduceMean``. ``K`` is
  // deliberately *not* an initializer; it is a graph input below.
  AddInitializerShape(*graph, "zero_idx", {0});
  AddInitializerShape(*graph, "unsqueeze_axes", {0});
  AddInitializerShape(*graph, "reduce_axes", {-1});
  AddInitializerShape(*graph, "mean_axes", {-1});
  TensorProto &cond_tensor = *graph->add_initializer();
  cond_tensor.set_name("cond_init");
  cond_tensor.set_data_type(TensorProto::DataType::BOOL);
  cond_tensor.set_raw_data(utils::ByteSpan(BoolBytes(true)));

  // Graph inputs. ``X`` carries symbolic dims ``[N, D]``; ``K`` is the INT64
  // ``[1]`` number of top distances to keep.
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec("N"), DimSpec("D")});
  AppendValueInfo(*graph->add_input(), "K", DataType::INT64, {DimSpec(int64_t{1})});

  // Intermediate value_info entries. ``dist`` is the stacked distance matrix
  // whose leading axis is the runtime Loop trip count (symbolic ``loop``) and
  // whose trailing axis is the per-iteration ``[N]`` element shape;
  // ``topk_values`` keeps a symbolic ``k`` as its trailing axis because ``K``
  // is a runtime input.
  AppendValueInfo(*graph->add_value_info(), "dist", DataType::FLOAT,
                  {DimSpec("loop"), DimSpec("N")});
  AppendValueInfo(*graph->add_value_info(), "topk_values", DataType::FLOAT,
                  {DimSpec("loop"), DimSpec("k")});

  // Output Y — the per-row mean of the ``k`` largest distances, shape
  // ``[loop]``.
  AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec("loop")});

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

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

// ---------------------------------------------------------------------------
// Pairwise-distance → TopK(k=input) → ReduceMean shape-inference cases. Three
// variants are registered here, differing only in how the pairwise-distance
// matrix is computed:
//
//   * broadcasting  (``RegisterTopKPairwiseDistanceShapeInferenceCases``)
//   * ``Scan``      (``RegisterScanTopKPairwiseDistanceShapeInferenceCases``)
//   * ``Loop``      (``RegisterLoopTopKPairwiseDistanceShapeInferenceCases``)
//
// In every variant ``X`` carries symbolic dims ``[N, D]`` and the TopK ``k`` is
// a runtime model input, so ``TopK`` emits a symbolic output axis that
// ``ReduceMean`` collapses to recover a rank-1 output.
// ---------------------------------------------------------------------------

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
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

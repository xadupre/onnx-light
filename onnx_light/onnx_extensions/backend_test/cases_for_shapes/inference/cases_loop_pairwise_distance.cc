// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
//
// All ``axes`` inputs (``unsqueeze_axes`` for ``Unsqueeze``, ``reduce_axes``
// for ``ReduceSum``) are declared as initializers in the outer graph and
// referenced from the body via outer-scope lookup, which keeps the body
// minimal and exercises the outer-scope propagation in
// :cpp:func:`onnx_shapes::shapes::ComputeShapeLoop`.
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
  AttributeProto *keepdims = reduce.add_attribute();
  keepdims->set_name("keepdims");
  keepdims->set_type(AttributeProto::AttributeType::INT);
  keepdims->set_i(0);
  AddNode(g, "Sqrt", {"sum_sq"}, {"dist"});
  AddNode(g, "Identity", {"dist"}, {"scan_out"});

  AddBodyOutputTyped(g, "cond_out", TensorProto::DataType::BOOL, /*dims=*/{});
  AddBodyOutputTyped(g, "scan_out", TensorProto::DataType::FLOAT, /*dims=*/{"N"});
  return g;
}

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Gather → Constant → Loop(body=pairwise_distance_body)`` —
// computes the pairwise Euclidean distance matrix of an input ``X`` of
// shape ``[N, D]`` via a ``Loop`` node that iterates ``N`` times. Each
// iteration emits one row of the resulting ``[N, N]`` distance matrix as a
// scan output, and the ``Loop`` node stacks the rows into the final
// ``[N, N]`` tensor.
//
// The case exercises the shape-inference path through a non-trivial
// ``Loop`` body — including outer-scope reference to the main-graph input
// ``X`` from inside the body — and ensures that
// :cpp:func:`onnx_shapes::shapes::ComputeShapeLoop` recovers the rank-2
// stacked scan-output shape with the per-iteration trailing dim ``N``
// propagated from the body's ``ReduceSum`` output.
//
// Main graph topology:
//
//   X                (N, D)         FLOAT
//      │
//      ├── Shape ──► shape_X        (2,)            INT64
//      │              │
//      │              └── Gather(axis=0, indices=[0]) ──► trip_count  (1,) INT64
//      │
//      ├── (initializer ``cond_init``  BOOL [1])
//      ├── (initializer ``unsqueeze_axes``  INT64 [1] = [0])
//      └── (initializer ``reduce_axes``     INT64 [1] = [-1])
//
//   Loop(trip_count[0], cond_init, body=pairwise_distance_body) ──► Y  (N, N) FLOAT
//
// The reference DataSet uses a 3×3 input with rows that lie on the axes of
// an integer right-triangle grid, so the expected output is the matrix of
// integer pairwise distances ``[[0, 3, 4], [3, 0, 5], [4, 5, 0]]``.
// ---------------------------------------------------------------------------
void RegisterLoopPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry,
                                                     TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);

  const std::string name = "test_cc_shape_inference_loop_pairwise_distance";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Main-graph nodes.
  AddNode(*graph, "Shape", {"X"}, {"shape_X"});
  NodeProto &gather_trip = AddNode(*graph, "Gather", {"shape_X", "zero_idx"}, {"trip_count"});
  AddAxisAttribute(gather_trip, 0);

  NodeProto &loop_node = AddNode(*graph, "Loop", {"trip_count", "cond_init"}, {"Y_pre_abs"});
  AttributeProto *body_attr = loop_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildPairwiseDistanceBody();

  // Append an ``Abs`` so the model's final output is computed by an Abs of
  // the Loop's stacked scan output. The pairwise distances are non-negative
  // so the recorded reference values are unchanged.
  AddNode(*graph, "Abs", {"Y_pre_abs"}, {"Y"});

  // Initializers — index ``[0]`` for the trip-count Gather, the BOOL
  // ``cond`` constant, and the axes vectors referenced from the body via
  // outer-scope lookup.
  AddInitializerShape(*graph, "zero_idx", {0});
  AddInitializerShape(*graph, "unsqueeze_axes", {0});
  AddInitializerShape(*graph, "reduce_axes", {-1});
  TensorProto &cond_tensor = *graph->add_initializer();
  cond_tensor.set_name("cond_init");
  cond_tensor.set_data_type(TensorProto::DataType::BOOL);
  cond_tensor.set_raw_data(utils::ByteSpan(BoolBytes(true)));

  // Graph input X uses symbolic dims ``[batch, features]`` matching the
  // expected value_info shapes. The dynamic-shape backend test supports
  // symbolic input dims directly, propagating them through shape inference.
  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  AppendValueInfo(*graph->add_input(), "X", kFloat, {DimSpec("batch"), DimSpec("features")});

  // Intermediate value_info entries. ``shape_X`` is the 1-D INT64 shape
  // vector of ``X`` (length 2 = rank of ``X``); ``trip_count`` is the
  // INT64 ``[1]`` slice extracting the leading dim.
  AppendValueInfo(*graph->add_value_info(), "shape_X", DataType::INT64, {DimSpec(2)});
  AppendValueInfo(*graph->add_value_info(), "trip_count", DataType::INT64, {DimSpec(1)});
  AppendValueInfo(*graph->add_value_info(), "Y_pre_abs", DataType::FLOAT,
                  {DimSpec("batch"), DimSpec("batch")});

  // Output Y — the stacked pairwise distance matrix of shape ``[3, 3]``.
  AppendValueInfo(*graph->add_output(), "Y", kFloat, {DimSpec("batch"), DimSpec("batch")});

  // Reference DataSet with concrete ``[3, 4]`` input. Rows lie on the axes
  // of an integer right-triangle grid: (0,0,0,0), (3,0,0,0), (0,4,0,0).
  // Pairwise L2 distances form the 3-4-5 triple, giving the integer distance
  // matrix [[0,3,4],[3,0,5],[4,5,0]].
  Tensor x = Tensor::FromFloat("X", {3, 4},
                               {0.0f, 0.0f, 0.0f, 0.0f, //
                                3.0f, 0.0f, 0.0f, 0.0f, //
                                0.0f, 4.0f, 0.0f, 0.0f});
  Tensor y = Tensor::FromFloat("Y", {3, 3},
                               {0.0f, 3.0f, 4.0f, //
                                3.0f, 0.0f, 5.0f, //
                                4.0f, 5.0f, 0.0f});
  AppendDataSet(tc, {std::move(x)}, {std::move(y)});

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

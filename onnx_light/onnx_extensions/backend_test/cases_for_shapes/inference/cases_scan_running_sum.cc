// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Shape-inference backend test case for a Scan-based running (cumulative) sum.
// Mirrors the structure of the Loop pairwise-distance case
// (cases_loop_pairwise_distance.cc) but exercises the ``Scan`` operator
// instead of ``Loop``.

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kDefaultIrVersion = 10;

// Builds the body subgraph for the running-sum Scan case.
//
// Inputs : acc_in   (FLOAT [D] — running accumulator state)
//          x_t      (FLOAT [D] — one row of the outer-scope scan input X)
// Outputs: acc_out  (FLOAT [D] — updated accumulator = acc_in + x_t)
//          scan_out (FLOAT [D] — per-iteration scan output)
//
// Body topology:
//   acc_out  = Add(acc_in, x_t)
//   scan_out = Identity(acc_out)
GraphProto BuildRunningSumBody() {
  GraphProto g;
  g.set_name("running_sum_body");

  // Body inputs: first N=1 state input, then M=1 scan input.
  AppendValueInfo(*g.add_input(), "acc_in", TensorProto::DataType::FLOAT, {DimSpec("D")});
  AppendValueInfo(*g.add_input(), "x_t", TensorProto::DataType::FLOAT, {DimSpec("D")});

  AddNode(g, "Add", {"acc_in", "x_t"}, {"acc_out"});
  AddNode(g, "Identity", {"acc_out"}, {"scan_out"});

  // Body outputs: first N=1 state output, then K=1 scan output.
  AppendValueInfo(*g.add_output(), "acc_out", TensorProto::DataType::FLOAT, {DimSpec("D")});
  AppendValueInfo(*g.add_output(), "scan_out", TensorProto::DataType::FLOAT, {DimSpec("D")});

  return g;
}

} // namespace

// ---------------------------------------------------------------------------
// ``Scan(body=running_sum_body) → Abs`` — computes the running (cumulative)
// row sum of an input ``X`` of shape ``[T, D]`` via a ``Scan`` node that
// scans ``X`` along axis 0. Each iteration accumulates one row ``x_t`` into
// a running state ``acc`` (initially zeros) and emits ``acc`` as a
// per-iteration scan output. Stacking the T per-iteration outputs produces
// the cumulative-sum matrix ``Y_pre_abs`` of shape ``[T, D]``; the final
// output ``Y = Abs(Y_pre_abs)`` exercises shape propagation through one
// additional node after the ``Scan`` (mirroring the ``Loop`` case).
//
// Main graph topology:
//
//   X              (T, D)    FLOAT  [graph input]
//   zero_acc       (D,)      FLOAT  [initializer: zeros]
//       │
//       ├── Scan(zero_acc, X, body=running_sum_body, num_scan_inputs=1)
//       │       → acc_final  (D,)    FLOAT  [final accumulator state]
//       │       → Y_pre_abs  (T, D)  FLOAT  [stacked cumulative rows]
//       │
//       └── Abs(Y_pre_abs) ──► Y  (T, D)  FLOAT  [graph output]
//
// The reference DataSet uses T=4, D=3 with integer-valued rows so that the
// cumulative sums are exact in float32:
//   X  = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
//   Y  = [[1, 2, 3], [5, 7, 9], [12, 15, 18], [22, 26, 30]]
//
// Shape-inference path exercised:
//   * ``ComputeShapeScan`` propagates the state shape ``[kD]`` (concrete,
//     from the ``zero_acc`` initializer) through the body.  Because
//     ``BroadcastDim([kD], [D-symbolic])`` returns the concrete integer (>1
//     wins), ``acc_out`` and ``scan_out`` carry shape ``[kD]``.
//   * ``acc_final`` is inferred as ``[kD]`` (state shape matches initializer).
//   * The trip count is taken from ``X.shape[0] = T`` (symbolic); the body's
//     per-iteration scan output ``scan_out`` shape ``[kD]`` is stacked to
//     ``[T, kD]`` for ``Y_pre_abs``.
//   * ``Abs`` propagates the ``[T, kD]`` shape to ``Y``.
// ---------------------------------------------------------------------------
void RegisterScanRunningSumShapeInferenceCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);

  const std::string name = "test_cc_shape_inference_scan_running_sum";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Main-graph nodes.
  // Scan: 1 state input (zero_acc), 1 scan input (X), num_scan_inputs=1.
  // Outputs: acc_final (state), Y_pre_abs (stacked scan output).
  NodeProto &scan_node = AddNode(*graph, "Scan", {"zero_acc", "X"}, {"acc_final", "Y_pre_abs"});
  AddAttribute<int64_t>(scan_node, "num_scan_inputs", 1);
  AttributeProto *body_attr = scan_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildRunningSumBody();

  // Abs so shape propagates through one more node (mirrors the Loop case).
  AddNode(*graph, "Abs", {"Y_pre_abs"}, {"Y"});

  // Initializer: zero_acc = [0.0, 0.0, 0.0] — FLOAT [D] with D=3.
  constexpr int64_t kD = 3;
  std::vector<float> zero_vals(static_cast<size_t>(kD), 0.0f);
  AddInitializer<float>(*graph, "zero_acc", {kD}, zero_vals);

  // Graph input X uses symbolic dims (T, D).
  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  AppendValueInfo(*graph->add_input(), "X", kFloat, {DimSpec("T"), DimSpec("D")});

  // Intermediate value_info entries with the shapes that shape inference
  // should recover.
  // Note: zero_acc is a concrete initializer of shape [kD], so shape inference
  // propagates D=kD as a concrete integer through Add/Identity/Scan state.
  // acc_in=[kD] + x_t=[D-symbolic] → BroadcastDim returns [kD] (concrete wins
  // when >1).  Therefore the stacked scan output and final output also carry
  // the concrete kD in dim[1].
  AppendValueInfo(*graph->add_value_info(), "acc_final", DataType::FLOAT, {DimSpec(kD)});
  AppendValueInfo(*graph->add_value_info(), "Y_pre_abs", DataType::FLOAT,
                  {DimSpec("T"), DimSpec(kD)});

  // Graph output Y — T remains symbolic (from X.shape[0]); D is concrete kD.
  AppendValueInfo(*graph->add_output(), "Y", kFloat, {DimSpec("T"), DimSpec(kD)});

  // Reference DataSet: T=4, D=3.
  // X rows are consecutive integers starting at 1 so the cumulative sums
  // are exact in float32.
  //   Step 0: acc = [0+1,  0+2,  0+3]  = [1,  2,  3]   → scan_out = [1,  2,  3]
  //   Step 1: acc = [1+4,  2+5,  3+6]  = [5,  7,  9]   → scan_out = [5,  7,  9]
  //   Step 2: acc = [5+7,  7+8,  9+9]  = [12, 15, 18]  → scan_out = [12, 15, 18]
  //   Step 3: acc = [12+10,15+11,18+12]= [22, 26, 30]  → scan_out = [22, 26, 30]
  //   Y = Abs(stacked scan_out) = [[1,2,3],[5,7,9],[12,15,18],[22,26,30]]
  constexpr int64_t kT = 4;
  Tensor x = Tensor::FromFloat("X", {kT, kD},
                               {1.0f, 2.0f, 3.0f, //
                                4.0f, 5.0f, 6.0f, //
                                7.0f, 8.0f, 9.0f, //
                                10.0f, 11.0f, 12.0f});
  Tensor y = Tensor::FromFloat("Y", {kT, kD},
                               {1.0f, 2.0f, 3.0f,    //
                                5.0f, 7.0f, 9.0f,    //
                                12.0f, 15.0f, 18.0f, //
                                22.0f, 26.0f, 30.0f});
  AppendDataSet(tc, {std::move(x)}, {std::move(y)});

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

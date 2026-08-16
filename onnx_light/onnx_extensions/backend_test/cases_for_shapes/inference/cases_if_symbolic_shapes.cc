// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

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

// Wires a single-node ``Identity`` sub-graph that captures the outer-scope
// tensor ``input_name`` and surfaces it as the sub-graph's only declared
// output ``output_name``. The sub-graph output ``ValueInfoProto`` is
// populated with ``elem_type`` and ``dims`` so the model is well-formed
// for ONNX shape inference (which requires every sub-graph output to
// carry a tensor type).
void BuildIdentityBranch(GraphProto &g, const std::string &graph_name,
                         const std::string &input_name, const std::string &output_name,
                         DataType elem_type, const std::vector<DimSpec> &dims) {
  g.set_name(graph_name);

  NodeProto *node = g.add_node();
  node->set_op_type("Identity");
  node->add_input(input_name);
  node->add_output(output_name);

  AppendValueInfo(*g.add_output(), output_name, elem_type, dims);
}

// Appends a second ``Identity`` node to the sub-graph ``g`` mapping
// ``input_name`` to ``output_name``, and declares the matching sub-graph
// output ``ValueInfoProto``.
void AppendIdentityOutput(GraphProto &g, const std::string &input_name,
                          const std::string &output_name, DataType elem_type,
                          const std::vector<DimSpec> &dims) {
  NodeProto *node = g.add_node();
  node->set_op_type("Identity");
  node->add_input(input_name);
  node->add_output(output_name);

  AppendValueInfo(*g.add_output(), output_name, elem_type, dims);
}

} // namespace

// ---------------------------------------------------------------------------
// ``If`` with two outputs whose branches produce tensors of the same rank
// but **different symbolic shapes** — exercises the branch-merging path of
// the ``If`` shape inference in
// :cpp:func:`onnx_shapes::shapes::controlflow::ComputeShapeIf`. The merge
// keeps an axis only when *both* branches agree on it; any differing axis
// is replaced by a fresh symbolic dim named ``If_<out>_d<i>`` (see
// ``MergeBranchOutputs`` in
// ``onnx_extensions/shapes/shapes/controlflow/shape_controlflow.cc``).
//
// The ``else_branch`` deliberately includes a data-dependent ``Compress``
// node so the differing leading axis of ``out_a`` originates from
// ``ComputeShapeCompress`` (a ``Compress_<out>_count`` symbol) on the else
// side and from a fixed symbolic dim (``B``) propagated through
// ``Identity`` on the then side. The merging step therefore has to
// reconcile two genuinely different symbolic dims of the same rank.
//
// Graph topology::
//
//   inputs:
//     cond    : bool[]
//     a_then  : float[3, 4]
//     a_else  : float[5, 4]
//     c_else  : bool[5]            # Compress condition (else-branch only)
//     b_then  : int64[3]
//     b_else  : int64[5]
//
//   out_a, out_b via:
//     B1, B2 = If(cond,
//                 then_branch = {
//                   out_a_then = Identity(a_then)            # float[3, 4]
//                   out_b_then = Identity(b_then)            # int64[3]
//                 },
//                 else_branch = {
//                   out_a_else = Compress(a_else, c_else,
//                                         axis=0)            # float[?, 4]
//                   out_b_else = Identity(b_else)            # int64[5]
//                 })
//     out_a = Abs(B1)                                        # float[B1, 4]
//     out_b = Neg(B2)                                        # int64[B2]
//
// Expected inferred output shapes (the differing leading axis is recorded
// as a named symbolic dim — ``B1`` for ``out_a`` and ``B2`` for ``out_b`` —
// while the matching trailing axis is preserved verbatim; ``Abs`` / ``Neg``
// propagate the merged shapes)::
//
//     out_a : float[B1, 4]
//     out_b : int64[B2]
//
// Input shapes are declared with concrete ``dim_value``s (not ``dim_param``s)
// so the dynamic shape-inference backend test
// (``test_backend_with_optim_shape_inference_dynamic``) — which remaps each
// concrete input dim to a fresh ``dim_param`` before running inference — can
// exercise this case as well. The reference :cpp:class:`DataSet` selects the
// then-branch (``cond = true``) so the case is executable end-to-end by
// ``BackendTestCaseRunModel``.
// ---------------------------------------------------------------------------
void RegisterIfSymbolicShapesShapeInferenceCases(std::vector<TestCase> &registry,
                                                 TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(13);

  const std::string name = "test_cc_shape_inference_if_symbolic_shapes";

  TestCase tc(name, name, "model", "inference", 1e-7, 1e-3);

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // B1, B2 = If(cond, then_branch=..., else_branch=...)
  NodeProto &if_node =
      AddNode(*graph, "If", {"cond"}, {"I1", "I2"}, /*domain=*/nullptr, "if_two_outputs");

  AttributeProto *then_attr = if_node.add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *then_g = then_attr->add_g();
  // Both then-branch outputs are wired through ``Identity`` so they pick up
  // the outer-scope ValueInfo (and the corresponding shapes) of their
  // captured inputs. ``b_then`` is rank-1, matching the else-branch
  // ``out_b_else`` so the second ``If`` output merges to a rank-1 shape.
  BuildIdentityBranch(*then_g, "then_branch", "a_then", "out_a_then", DataType::FLOAT,
                      {DimSpec("D3"), DimSpec("D4")});
  AppendIdentityOutput(*then_g, "b_then", "out_b_then", DataType::INT64, {DimSpec("D3")});

  AttributeProto *else_attr = if_node.add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *else_g = else_attr->add_g();
  else_g->set_name("else_branch");
  // out_a_else = Compress(a_else, c_else, axis=0). The output's leading
  // axis is data-dependent so shape inference assigns it a fresh
  // ``Compress_out_a_else_count`` symbol; the trailing ``4`` is preserved.
  NodeProto *compress_node = else_g->add_node();
  compress_node->set_op_type("Compress");
  compress_node->add_input("a_else");
  compress_node->add_input("c_else");
  compress_node->add_output("out_a_else");
  AddAttribute<int64_t>(*compress_node, "axis", 0);
  AppendValueInfo(*else_g->add_output(), "out_a_else", DataType::FLOAT,
                  {DimSpec("Compress_0_d0"), DimSpec(int64_t{4})});
  AppendIdentityOutput(*else_g, "b_else", "out_b_else", DataType::INT64, {DimSpec(int64_t{5})});

  // out_a = Abs(B1) and out_b = Neg(B2). These unary nodes sit between the
  // ``If`` outputs (``B1`` / ``B2``) and the graph outputs, so the merged
  // symbolic shapes flow through ``Abs`` / ``Neg`` (both shape-preserving)
  // onto ``out_a`` / ``out_b``.
  AddNode(*graph, "Abs", {"I1"}, {"Y1"}, /*domain=*/nullptr, "abs_out_a");
  AddNode(*graph, "Neg", {"I2"}, {"Y2"}, /*domain=*/nullptr, "neg_out_b");

  // Intermediate value_info for the ``If`` outputs ``B1`` / ``B2``. Recording
  // them keeps the ``TestOptimShapeInferenceNoNewNames`` check happy (shape
  // inference must not introduce value_info names absent from the model). The
  // leading axis is the merged symbolic dim, which shape inference
  // canonicalizes to the graph-output anchor name (``B1`` for ``out_a`` /
  // ``B2`` for ``out_b``); it is treated as ``-1`` / "unknown" by the
  // shape-inference test helpers.
  AppendValueInfo(*graph->add_value_info(), "I1", DataType::FLOAT,
                  {DimSpec("B1"), DimSpec(int64_t{4})});
  AppendValueInfo(*graph->add_value_info(), "I2", DataType::INT64, {DimSpec("B2")});

  // Graph inputs: scalar BOOL cond, two FLOAT[*, 4] inputs, the
  // else-branch Compress condition ``c_else: bool[5]``, and two INT64[*]
  // inputs. Distinct concrete leading dims (``3`` vs ``5``) force the
  // branch-merging path to synthesize a fresh symbolic dim.
  AppendValueInfo(*graph->add_input(), "cond", DataType::BOOL, std::vector<DimSpec>{});
  AppendValueInfo(*graph->add_input(), "a_then", DataType::FLOAT,
                  {DimSpec(int64_t{3}), DimSpec(int64_t{4})});
  AppendValueInfo(*graph->add_input(), "a_else", DataType::FLOAT,
                  {DimSpec(int64_t{5}), DimSpec(int64_t{4})});
  AppendValueInfo(*graph->add_input(), "c_else", DataType::BOOL, {DimSpec(int64_t{5})});
  AppendValueInfo(*graph->add_input(), "b_then", DataType::INT64, {DimSpec(int64_t{3})});
  AppendValueInfo(*graph->add_input(), "b_else", DataType::INT64, {DimSpec(int64_t{5})});

  // Graph outputs: the merged shapes recorded as ground truth for the
  // ``AllCollectedCasesInferOutputShapes`` test. ``CheckValueInfoMatches``
  // / ``CheckOutputs`` only enforce concrete dim equality, so the leading
  // ``B1`` / ``B2`` symbolic axes are recorded as named dims (treated as
  // ``-1`` / "unknown" by the test helpers), while the trailing concrete
  // ``4`` dim of ``out_a`` is checked verbatim.
  AppendValueInfo(*graph->add_output(), "Y1", DataType::FLOAT, {"B1", DimSpec(int64_t{4})});
  AppendValueInfo(*graph->add_output(), "Y2", DataType::INT64, {"B2"});

  // Reference DataSet — concrete ``A=3``, ``B=5``, ``cond=true`` selects the
  // then-branch, so the ``If`` outputs are the then-branch inputs and the
  // trailing ``Abs`` / ``Neg`` nodes yield ``out_a = Abs(a_then)`` and
  // ``out_b = Neg(b_then)``.
  constexpr int64_t kA = 3;
  constexpr int64_t kB = 5;
  constexpr int64_t kK = 4;

  std::vector<float> a_then_values(static_cast<size_t>(kA * kK));
  for (size_t i = 0; i < a_then_values.size(); ++i) {
    a_then_values[i] = static_cast<float>(i) * 0.1f + 1.0f;
  }
  std::vector<float> a_else_values(static_cast<size_t>(kB * kK));
  for (size_t i = 0; i < a_else_values.size(); ++i) {
    a_else_values[i] = static_cast<float>(i) * 0.1f - 1.0f;
  }
  std::vector<int64_t> b_then_values(static_cast<size_t>(kA));
  for (size_t i = 0; i < b_then_values.size(); ++i) {
    b_then_values[i] = static_cast<int64_t>(i) + 1;
  }
  std::vector<int64_t> b_else_values(static_cast<size_t>(kB));
  for (size_t i = 0; i < b_else_values.size(); ++i) {
    b_else_values[i] = -(static_cast<int64_t>(i) + 1);
  }

  Tensor cond_tensor = Tensor::FromBool("cond", {}, {1});
  Tensor a_then = Tensor::FromFloat("a_then", {kA, kK}, a_then_values);
  Tensor a_else = Tensor::FromFloat("a_else", {kB, kK}, a_else_values);
  // ``c_else`` is only consumed by the else-branch ``Compress`` and is
  // therefore unused at runtime (``cond=true`` selects the then-branch);
  // its concrete value is irrelevant but its shape ``[B]`` is recorded so
  // the symbolic-dim propagation test resolves ``c_else.dim[0] == B``.
  Tensor c_else = Tensor::FromBool("c_else", {kB}, {1, 0, 1, 0, 1});
  Tensor b_then = Tensor::FromInt64("b_then", {kA}, b_then_values);
  Tensor b_else = Tensor::FromInt64("b_else", {kB}, b_else_values);

  // ``cond=true`` selects the then-branch, so ``B1 = a_then`` and
  // ``B2 = b_then``. The trailing ``Abs`` / ``Neg`` nodes then produce
  // ``out_a = Abs(a_then)`` (unchanged, as ``a_then`` is all-positive) and
  // ``out_b = Neg(b_then)``.
  Tensor out_a = a_then;
  out_a.name = "out_a";
  std::vector<int64_t> out_b_values(b_then_values.size());
  for (size_t i = 0; i < out_b_values.size(); ++i) {
    out_b_values[i] = -b_then_values[i];
  }
  Tensor out_b = Tensor::FromInt64("out_b", {kA}, out_b_values);

  AppendDataSet(tc,
                {std::move(cond_tensor), std::move(a_then), std::move(a_else), std::move(c_else),
                 std::move(b_then), std::move(b_else)},
                {std::move(out_a), std::move(out_b)});

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

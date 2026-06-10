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
// :cpp:func:`onnx_optim::shapes::controlflow::ComputeShapeIf`. The merge
// keeps an axis only when *both* branches agree on it; any differing axis
// is replaced by a fresh symbolic dim named ``If_<out>_d<i>`` (see
// ``MergeBranchOutputs`` in
// ``onnx_optim/shapes/controlflow/shape_controlflow.cc``).
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
//   out_a, out_b = If(cond,
//                     then_branch = {
//                       out_a = Identity(a_then)              # float[3, 4]
//                       out_b = Identity(b_then)              # int64[3]
//                     },
//                     else_branch = {
//                       out_a = Compress(a_else, c_else,
//                                        axis=0)              # float[?, 4]
//                       out_b = Identity(b_else)              # int64[5]
//                     })
//
// Expected inferred output shapes (the differing leading axis becomes a
// fresh ``If_<out>_d<i>`` symbolic dim while the matching trailing axis is
// preserved verbatim)::
//
//     out_a : float[If_out_a_d0, 4]
//     out_b : int64[If_out_b_d0]
//
// Input shapes are declared with concrete ``dim_value``s (not ``dim_param``s)
// so the dynamic shape-inference backend test
// (``test_backend_with_optim_shape_inference_dynamic``) — which remaps each
// concrete input dim to a fresh ``dim_param`` before running inference — can
// exercise this case as well. The reference :cpp:class:`DataSet` selects the
// then-branch (``cond = true``) so the case is executable end-to-end by
// ``BackendTestCaseRunModel``.
// ---------------------------------------------------------------------------
void RegisterIfSymbolicShapesShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);

  const std::string name = "test_cc_shape_inference_if_symbolic_shapes";

  TestCase tc(name, name, "model", "inference", 1e-3, 1e-7);

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // out_a, out_b = If(cond, then_branch=..., else_branch=...)
  NodeProto &if_node =
      AddNode(*graph, "If", {"cond"}, {"out_a", "out_b"}, /*domain=*/nullptr, "if_two_outputs");

  AttributeProto *then_attr = if_node.add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *then_g = then_attr->add_g();
  // Both then-branch outputs are wired through ``Identity`` so they pick up
  // the outer-scope ValueInfo (and the corresponding shapes) of their
  // captured inputs.
  BuildIdentityBranch(*then_g, "then_branch", "a_then", "out_a_then", DataType::FLOAT,
                      {DimSpec("D3"), DimSpec("D4")});
  AppendIdentityOutput(*then_g, "b_then", "out_b_then", DataType::INT64, {DimSpec(D3)});

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
                  {DimSpec(), DimSpec(int64_t{4})});
  AppendIdentityOutput(*else_g, "b_else", "out_b_else", DataType::INT64, {DimSpec(int64_t{5})});

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
  // / ``CheckOutputs`` only enforce concrete dim equality, so the
  // ``If_<out>_d<i>`` symbolic axes are recorded as named dims (treated as
  // ``-1`` / "unknown" by the test helpers), while the trailing concrete
  // ``4`` dim of ``out_a`` is checked verbatim.
  AppendValueInfo(*graph->add_output(), "out_a", DataType::FLOAT,
                  {"If_out_a_d0", DimSpec(int64_t{4})});
  AppendValueInfo(*graph->add_output(), "out_b", DataType::INT64, {"If_out_b_d0"});

  // Reference DataSet — concrete ``A=3``, ``B=5``, ``cond=true`` selects the
  // then-branch, so the runtime outputs are exactly the then-branch inputs.
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

  Tensor out_a = a_then;
  out_a.name = "out_a";
  Tensor out_b = b_then;
  out_b.name = "out_b";

  AppendDataSet(tc,
                {std::move(cond_tensor), std::move(a_then), std::move(a_else), std::move(c_else),
                 std::move(b_then), std::move(b_else)},
                {std::move(out_a), std::move(out_b)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

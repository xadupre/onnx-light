// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/controlflow/include_controlflow_kernels.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Returns the bytes of a float array as a byte vector.
std::vector<uint8_t> FloatVecBytes(const std::vector<float> &values) {
  std::vector<uint8_t> out(values.size() * sizeof(float));
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

// Builds a single-node subgraph whose only node is a ``Constant`` op
// producing ``output_name`` from the tensor ``value``. The graph has no
// inputs and a single output declared from ``value``'s type and shape.
void BuildConstantBranch(GraphProto &g, const std::string &graph_name,
                         const std::string &output_name, const Tensor &value) {
  g.set_name(graph_name);

  NodeProto *node = g.add_node();
  node->set_op_type("Constant");
  node->add_output(output_name);

  AttributeProto *attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<DataType>(value.data_type));
  for (int64_t d : value.shape) {
    t->add_dims(static_cast<uint64_t>(d));
  }
  // Pack the value bytes as raw_data (row-major little-endian, matching the
  // layout used by ``Tensor::data``).
  t->set_raw_data(utils::ByteSpan(value.data));

  Tensor out = value;
  out.name = output_name;
  FillValueInfo(out, *g.add_output());
}

} // namespace

// ---------------------------------------------------------------------------
// If — selects one of two branches based on a scalar BOOL ``cond`` (since
// opset 1; subgraphs may also yield sequence types since opset 13).
//
// Four cases are registered:
//
//   * ``test_cc_if`` and ``test_cc_if_else`` use a single-output If whose
//     branches each return a 1-D float tensor via a ``Constant`` node;
//     they exercise the two selection outcomes of a scalar BOOL ``cond``.
//   * ``test_cc_if_multi_output`` exercises a single ``If`` whose two
//     branches declare *two* outputs each (a 1-D INT64 tensor and a
//     2-D FLOAT tensor), validating that the branch-aware kernel
//     correctly returns every declared subgraph output.
//   * ``test_cc_if_outer_scope`` builds an ``If`` whose branches consume
//     an outer-scope tensor ``x`` (a graph initializer in the test
//     model) and apply ``Neg`` (then-branch) or ``Identity`` (else-branch)
//     to it, validating that the kernel inherits the caller's outer
//     scope so subgraphs can reference values declared outside the
//     branch.
// ---------------------------------------------------------------------------
void RegisterIfCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const If if_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> big_shape = {512, 512};
    const Tensor then_value = Tensor::FromFloat("", big_shape, Randn<float>(big_shape, 4301));
    const Tensor else_value = Tensor::FromFloat("", big_shape, Randn<float>(big_shape, 4302));

    NodeProto node;
    node.set_op_type("If");
    node.add_input("cond");
    node.add_output("res");

    AttributeProto *then_attr = node.add_attribute();
    then_attr->set_name("then_branch");
    then_attr->set_type(AttributeProto::AttributeType::GRAPH);
    BuildConstantBranch(*then_attr->add_g(), "then_graph", "then_out", then_value);

    AttributeProto *else_attr = node.add_attribute();
    else_attr->set_name("else_branch");
    else_attr->set_type(AttributeProto::AttributeType::GRAPH);
    BuildConstantBranch(*else_attr->add_g(), "else_graph", "else_out", else_value);

    Expect(registry, std::move(node), "test_cc_if_benchmark", {opset}, [=]() -> IoData {
      Tensor cond("", DataType::BOOL, {}, {1});
      Tensor res = if_kernel(cond, then_value, else_value);
      return IoData{{std::move(cond)}, {std::move(res)}};
    });
    return;
  }

  const Tensor then_value = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Tensor else_value = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  auto make_node = [&]() {
    NodeProto node;
    node.set_op_type("If");
    node.add_input("cond");
    node.add_output("res");

    AttributeProto *then_attr = node.add_attribute();
    then_attr->set_name("then_branch");
    then_attr->set_type(AttributeProto::AttributeType::GRAPH);
    BuildConstantBranch(*then_attr->add_g(), "then_graph", "then_out", then_value);

    AttributeProto *else_attr = node.add_attribute();
    else_attr->set_name("else_branch");
    else_attr->set_type(AttributeProto::AttributeType::GRAPH);
    BuildConstantBranch(*else_attr->add_g(), "else_graph", "else_out", else_value);

    return node;
  };

  // cond = true → output is the then-branch value.
  {
    NodeProto node = make_node();
    Expect(registry, std::move(node), "test_cc_if", {opset}, [=]() -> IoData {
      Tensor cond("", DataType::BOOL, {}, {1});
      Tensor res = if_kernel(cond, then_value, else_value);
      return IoData{{std::move(cond)}, {std::move(res)}};
    });
  }

  // cond = false → output is the else-branch value.
  {
    NodeProto node = make_node();
    Expect(registry, std::move(node), "test_cc_if_else", {opset}, [=]() -> IoData {
      Tensor cond("", DataType::BOOL, {}, {0});
      Tensor res = if_kernel(cond, then_value, else_value);
      return IoData{{std::move(cond)}, {std::move(res)}};
    });
  }

  // Multi-output If: each branch declares two outputs. The returning kernel
  // overload now executes the selected branch graph, so the expected
  // outputs are obtained by running the kernel itself.
  {
    const Tensor then_a = Tensor::From<int64_t>("", {3}, {1, 2, 3});
    const Tensor then_b = Tensor::FromFloat("", {2, 2}, {0.1f, 0.2f, 0.3f, 0.4f});
    const Tensor else_a = Tensor::From<int64_t>("", {3}, {-1, -2, -3});
    const Tensor else_b = Tensor::FromFloat("", {2, 2}, {1.1f, 1.2f, 1.3f, 1.4f});

    NodeProto node;
    node.set_op_type("If");
    node.add_input("cond");
    node.add_output("res_a");
    node.add_output("res_b");

    AttributeProto *then_attr = node.add_attribute();
    then_attr->set_name("then_branch");
    then_attr->set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto &then_g = *then_attr->add_g();
    then_g.set_name("then_graph_two_outputs");
    {
      NodeProto *n0 = then_g.add_node();
      n0->set_op_type("Constant");
      n0->add_output("then_a");
      AttributeProto *a = n0->add_attribute();
      a->set_name("value");
      a->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = a->add_t();
      t->set_data_type(static_cast<DataType>(then_a.data_type));
      for (int64_t d : then_a.shape)
        t->add_dims(static_cast<uint64_t>(d));
      t->set_raw_data(utils::ByteSpan(then_a.data));
    }
    {
      NodeProto *n1 = then_g.add_node();
      n1->set_op_type("Constant");
      n1->add_output("then_b");
      AttributeProto *a = n1->add_attribute();
      a->set_name("value");
      a->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = a->add_t();
      t->set_data_type(static_cast<DataType>(then_b.data_type));
      for (int64_t d : then_b.shape)
        t->add_dims(static_cast<uint64_t>(d));
      t->set_raw_data(utils::ByteSpan(then_b.data));
    }
    Tensor a_info = then_a;
    a_info.name = "then_a";
    FillValueInfo(a_info, *then_g.add_output());
    Tensor b_info = then_b;
    b_info.name = "then_b";
    FillValueInfo(b_info, *then_g.add_output());

    AttributeProto *else_attr = node.add_attribute();
    else_attr->set_name("else_branch");
    else_attr->set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto &else_g = *else_attr->add_g();
    else_g.set_name("else_graph_two_outputs");
    {
      NodeProto *n0 = else_g.add_node();
      n0->set_op_type("Constant");
      n0->add_output("else_a");
      AttributeProto *a = n0->add_attribute();
      a->set_name("value");
      a->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = a->add_t();
      t->set_data_type(static_cast<DataType>(else_a.data_type));
      for (int64_t d : else_a.shape)
        t->add_dims(static_cast<uint64_t>(d));
      t->set_raw_data(utils::ByteSpan(else_a.data));
    }
    {
      NodeProto *n1 = else_g.add_node();
      n1->set_op_type("Constant");
      n1->add_output("else_b");
      AttributeProto *a = n1->add_attribute();
      a->set_name("value");
      a->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = a->add_t();
      t->set_data_type(static_cast<DataType>(else_b.data_type));
      for (int64_t d : else_b.shape)
        t->add_dims(static_cast<uint64_t>(d));
      t->set_raw_data(utils::ByteSpan(else_b.data));
    }
    Tensor ea_info = else_a;
    ea_info.name = "else_a";
    FillValueInfo(ea_info, *else_g.add_output());
    Tensor eb_info = else_b;
    eb_info.name = "else_b";
    FillValueInfo(eb_info, *else_g.add_output());

    Tensor cond("", DataType::BOOL, {}, {1});
    // cond = true → outputs come from the then-branch.
    Expect(registry, std::move(node), "test_cc_if_multi_output", {opset},
           [cond, then_a, then_b]() -> IoData { return IoData{{cond}, {then_a, then_b}}; });
  }

  // -------------------------------------------------------------------------
  // test_cc_if_seq — If selecting between two sequence branches.
  // Mirrors ONNX's ``test_if_seq``.
  // cond=true → then-branch returns Sequence([1,2,3,4,5]),
  // else-branch returns Sequence([5,4,3,2,1]).
  // Graph: If(cond) → ConcatFromSequence → FLOAT[1, 5].
  // -------------------------------------------------------------------------
  {
    const std::string name = "test_cc_if_seq";
    const OpsetId opset13 = DefaultOpset(13);

    Tensor cond_in("cond", DataType::BOOL, {}, {1});
    Tensor expected = Tensor::FromFloat("res", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    TestCase tc(name, name);
    ModelProto &model = tc.emplace_model();
    InitModel(model, /*ir_version=*/9, {opset13});
    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // Build then_body: Constant([1,2,3,4,5]) → SequenceConstruct → then_out
    GraphProto then_body;
    then_body.set_name("then_body");
    {
      NodeProto *cn = then_body.add_node();
      cn->set_op_type("Constant");
      cn->add_output("x");
      AttributeProto *a = cn->add_attribute();
      a->set_name("value");
      a->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = a->add_t();
      t->set_data_type(TensorProto::DataType::FLOAT);
      t->add_dims(5);
      t->set_raw_data(utils::ByteSpan(FloatVecBytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f})));
    }
    {
      NodeProto *sc = then_body.add_node();
      sc->set_op_type("SequenceConstruct");
      sc->add_input("x");
      sc->add_output("then_out");
    }
    {
      ValueInfoProto *vi = then_body.add_output();
      vi->set_name("then_out");
      TypeProto *tp = vi->ref_type().add_sequence_type()->add_elem_type();
      TypeProto::Tensor *tt = tp->add_tensor_type();
      tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
      tt->add_shape()->add_dim()->set_dim_value(5);
    }

    // Build else_body: Constant([5,4,3,2,1]) → SequenceConstruct → else_out
    GraphProto else_body;
    else_body.set_name("else_body");
    {
      NodeProto *cn = else_body.add_node();
      cn->set_op_type("Constant");
      cn->add_output("y");
      AttributeProto *a = cn->add_attribute();
      a->set_name("value");
      a->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = a->add_t();
      t->set_data_type(TensorProto::DataType::FLOAT);
      t->add_dims(5);
      t->set_raw_data(utils::ByteSpan(FloatVecBytes({5.0f, 4.0f, 3.0f, 2.0f, 1.0f})));
    }
    {
      NodeProto *sc = else_body.add_node();
      sc->set_op_type("SequenceConstruct");
      sc->add_input("y");
      sc->add_output("else_out");
    }
    {
      ValueInfoProto *vi = else_body.add_output();
      vi->set_name("else_out");
      TypeProto *tp = vi->ref_type().add_sequence_type()->add_elem_type();
      TypeProto::Tensor *tt = tp->add_tensor_type();
      tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
      tt->add_shape()->add_dim()->set_dim_value(5);
    }

    // If(cond) → seq_res
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("If");
      n->add_input("cond");
      n->add_output("seq_res");
      {
        AttributeProto *a = n->add_attribute();
        a->set_name("then_branch");
        a->set_type(AttributeProto::AttributeType::GRAPH);
        *a->add_g() = then_body;
      }
      {
        AttributeProto *a = n->add_attribute();
        a->set_name("else_branch");
        a->set_type(AttributeProto::AttributeType::GRAPH);
        *a->add_g() = else_body;
      }
    }
    // ConcatFromSequence(seq_res, axis=0, new_axis=1) → res
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("ConcatFromSequence");
      n->add_input("seq_res");
      n->add_output("res");
      AddAttribute<int64_t>(*n, "axis", 0);
      AddAttribute<int64_t>(*n, "new_axis", 1);
    }

    FillValueInfo(cond_in, *graph->add_input());
    FillValueInfo(expected, *graph->add_output());

    DataSet ds;
    ds.inputs.push_back(cond_in);
    ds.outputs.push_back(expected);
    tc.data_sets().emplace_back(std::move(ds));
    registry.emplace_back(std::move(tc));
  }

  // -------------------------------------------------------------------------
  // test_cc_if_opt — If with Optional<Sequence<FLOAT[5]>> output.
  // Mirrors ONNX's ``test_if_opt``.
  // cond=false → else-branch: Constant([1,2,3,4,5]) → SequenceConstruct →
  //              Optional → else_opt.
  // then-branch: Optional(empty, type=Seq<FLOAT[5]>) → optional_empty.
  // Graph: If(cond) → OptionalGetElement → ConcatFromSequence → FLOAT[1,5].
  // -------------------------------------------------------------------------
  {
    const std::string name = "test_cc_if_opt";
    const OpsetId opset16 = DefaultOpset(16);

    Tensor cond_in("cond", DataType::BOOL, {}, {0}); // false → else branch
    Tensor expected = Tensor::FromFloat("res", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    TestCase tc(name, name);
    ModelProto &model = tc.emplace_model();
    InitModel(model, /*ir_version=*/9, {opset16});
    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // then_body: Optional(empty, type=Seq<FLOAT[5]>) → optional_empty
    GraphProto then_body;
    then_body.set_name("then_body");
    {
      NodeProto *n = then_body.add_node();
      n->set_op_type("Optional");
      n->add_output("optional_empty");
      // ``type`` attribute specifies the optional element type.
      AttributeProto *a = n->add_attribute();
      a->set_name("type");
      a->set_type(AttributeProto::AttributeType::TYPE_PROTO);
      TypeProto *tp = a->add_tp();
      TypeProto *seq_tp = tp->add_sequence_type()->add_elem_type();
      TypeProto::Tensor *tt = seq_tp->add_tensor_type();
      tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
      tt->add_shape()->add_dim()->set_dim_value(5);
    }
    {
      ValueInfoProto *vi = then_body.add_output();
      vi->set_name("optional_empty");
      TypeProto *opt_tp = vi->ref_type().add_optional_type()->add_elem_type();
      TypeProto *seq_tp = opt_tp->add_sequence_type()->add_elem_type();
      TypeProto::Tensor *tt = seq_tp->add_tensor_type();
      tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
      tt->add_shape()->add_dim()->set_dim_value(5);
    }

    // else_body: Constant([1..5]) → SequenceConstruct → Optional → else_opt
    GraphProto else_body;
    else_body.set_name("else_body");
    {
      NodeProto *cn = else_body.add_node();
      cn->set_op_type("Constant");
      cn->add_output("x");
      AttributeProto *a = cn->add_attribute();
      a->set_name("value");
      a->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = a->add_t();
      t->set_data_type(TensorProto::DataType::FLOAT);
      t->add_dims(5);
      t->set_raw_data(utils::ByteSpan(FloatVecBytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f})));
    }
    {
      NodeProto *sc = else_body.add_node();
      sc->set_op_type("SequenceConstruct");
      sc->add_input("x");
      sc->add_output("else_seq");
    }
    {
      NodeProto *on = else_body.add_node();
      on->set_op_type("Optional");
      on->add_input("else_seq");
      on->add_output("else_opt");
    }
    {
      ValueInfoProto *vi = else_body.add_output();
      vi->set_name("else_opt");
      TypeProto *opt_tp = vi->ref_type().add_optional_type()->add_elem_type();
      TypeProto *seq_tp = opt_tp->add_sequence_type()->add_elem_type();
      TypeProto::Tensor *tt = seq_tp->add_tensor_type();
      tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
      tt->add_shape()->add_dim()->set_dim_value(5);
    }

    // If(cond) → opt_res
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("If");
      n->add_input("cond");
      n->add_output("opt_res");
      {
        AttributeProto *a = n->add_attribute();
        a->set_name("then_branch");
        a->set_type(AttributeProto::AttributeType::GRAPH);
        *a->add_g() = then_body;
      }
      {
        AttributeProto *a = n->add_attribute();
        a->set_name("else_branch");
        a->set_type(AttributeProto::AttributeType::GRAPH);
        *a->add_g() = else_body;
      }
    }
    // OptionalGetElement(opt_res) → seq_out
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("OptionalGetElement");
      n->add_input("opt_res");
      n->add_output("seq_out");
    }
    // ConcatFromSequence(seq_out, axis=0, new_axis=1) → res
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("ConcatFromSequence");
      n->add_input("seq_out");
      n->add_output("res");
      AddAttribute<int64_t>(*n, "axis", 0);
      AddAttribute<int64_t>(*n, "new_axis", 1);
    }

    FillValueInfo(cond_in, *graph->add_input());
    FillValueInfo(expected, *graph->add_output());

    DataSet ds;
    ds.inputs.push_back(cond_in);
    ds.outputs.push_back(expected);
    tc.data_sets().emplace_back(std::move(ds));
    registry.emplace_back(std::move(tc));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

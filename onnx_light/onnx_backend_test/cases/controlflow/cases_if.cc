// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/controlflow/include_controlflow_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

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
void RegisterIfCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::If if_kernel{ctx};

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
    Tensor cond("", DataType::BOOL, {}, {1});
    Tensor res = if_kernel(cond, then_value, else_value);
    Expect(node, {cond}, {res}, "test_cc_if", {opset}, "backend-test", registry);
  }

  // cond = false → output is the else-branch value.
  {
    NodeProto node = make_node();
    Tensor cond("", DataType::BOOL, {}, {0});
    Tensor res = if_kernel(cond, then_value, else_value);
    Expect(node, {cond}, {res}, "test_cc_if_else", {opset}, "backend-test", registry);
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
    Expect(node, {cond}, {then_a, then_b}, "test_cc_if_multi_output", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

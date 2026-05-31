// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/test_case.h"

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
// Both registered cases use the same model topology: an ``If`` node whose
// ``then_branch`` returns a 1-D float tensor ``[1.0, 2.0]`` via a
// ``Constant`` node and whose ``else_branch`` returns ``[3.0, 4.0]`` the
// same way. The two cases differ only in the value of ``cond`` (true vs.
// false) so the same model exercises both selection outcomes.
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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

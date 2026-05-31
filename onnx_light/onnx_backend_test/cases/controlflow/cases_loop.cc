// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Returns the bytes of an INT64 scalar with value ``v`` in little-endian
// order (matching ``Tensor::data`` layout).
std::vector<uint8_t> Int64Bytes(int64_t v) {
  std::vector<uint8_t> out(sizeof(int64_t));
  std::memcpy(out.data(), &v, sizeof(int64_t));
  return out;
}

// Adds a graph input named ``name`` with a tensor type ``dtype`` and a
// scalar shape (empty dims).
void AddGraphInputScalar(GraphProto &g, const std::string &name, DataType dtype) {
  ValueInfoProto *vi = g.add_input();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
  tensor_type->mutable_shape();
}

void AddGraphOutputTensor(GraphProto &g, const std::string &name, DataType dtype) {
  ValueInfoProto *vi = g.add_output();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
  tensor_type->mutable_shape();
}

// Builds a Loop body subgraph for our simple test case:
//
//   inputs : (i [INT64 scalar], cond_in [BOOL scalar])
//   nodes  : identity-on-cond + a Constant node emitting [42] (scan_output)
//   outputs: (cond_out=cond_in, scan_out [INT64 shape=[1]])
//
// The body has no loop-carried dependencies (N == 0) and produces one scan
// output (K == 1). The reference kernel does not execute the body; it is
// included in the model purely so the registered ``TestCase`` is a
// well-formed ONNX model with a valid ``Loop`` node.
GraphProto BuildSimpleLoopBody() {
  GraphProto g;
  g.set_name("loop_body");

  AddGraphInputScalar(g, "i", DataType::INT64);
  AddGraphInputScalar(g, "cond_in", DataType::BOOL);

  // cond_out = Identity(cond_in)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Identity");
    n->add_input("cond_in");
    n->add_output("cond_out");
  }
  // scan_out = Constant(value=int64 [42])
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Constant");
    n->add_output("scan_out");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->add_t();
    t->set_data_type(DataType::INT64);
    t->add_dims(1);
    t->set_raw_data(utils::ByteSpan(Int64Bytes(42)));
  }

  AddGraphOutputTensor(g, "cond_out", DataType::BOOL);
  // scan_out: the Constant node above produces an INT64 tensor of shape [1].
  ValueInfoProto *scan_vi = g.add_output();
  scan_vi->set_name("scan_out");
  TypeProto::Tensor *scan_tt = scan_vi->ref_type().mutable_tensor_type();
  scan_tt->set_elem_type(static_cast<int>(DataType::INT64));
  scan_tt->ref_shape().add_dim()->set_dim_value(1);
  return g;
}

// Builds a ``Loop`` node with the given input/output names and a body
// subgraph attribute.
NodeProto MakeLoopNode(const std::string &m, const std::string &cond,
                       const std::string &scan_output_name) {
  NodeProto node;
  node.set_op_type("Loop");
  node.add_input(m);
  node.add_input(cond);
  // No v_initial inputs (N == 0): the Loop node has 2 inputs.
  node.add_output(scan_output_name);

  AttributeProto *body_attr = node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildSimpleLoopBody();
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Loop — minimal cases exercising the reference kernel's stacking semantics.
//
// Two registered cases use a Loop node with no loop-carried dependencies
// (N=0) and a single scan output (K=1) producing the constant int64 ``[42]``
// per iteration. They differ only in the value of ``M``: 3 (the scan output
// is ``[3, 1]`` containing ``[[42], [42], [42]]``) and 0 (the scan output
// is empty along its leading axis, ``[0, 1]``).
// ---------------------------------------------------------------------------
void RegisterLoopCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Loop loop_kernel{ctx};

  const Tensor per_iter_value("", DataType::INT64, {1}, Int64Bytes(42));
  const Tensor cond_undef; // omitted scalar BOOL cond.

  auto register_case = [&](const std::string &test_name, int64_t m_value) {
    NodeProto node = MakeLoopNode("M", "", "scan_outputs");
    const Tensor m("", DataType::INT64, {}, Int64Bytes(m_value));
    // Always supply at least one per-iteration template tensor so the
    // stacked output keeps its dtype/trailing-shape even when the loop
    // runs zero iterations.
    const std::size_t row_len = std::max<std::size_t>(1, static_cast<std::size_t>(m_value));
    std::vector<Tensor> per_iter(row_len, per_iter_value);
    std::vector<Tensor> out =
        loop_kernel(m, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {per_iter});
    Expect(node, {m}, out, test_name, {opset}, "backend-test", registry);
  };

  // M = 3 → stacked scan output has shape [3, 1] = [[42], [42], [42]].
  register_case("test_cc_loop_basic_trip_count", 3);
  // M = 0 → stacked scan output has shape [0, 1] (empty along axis 0).
  register_case("test_cc_loop_zero_trip_count", 0);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

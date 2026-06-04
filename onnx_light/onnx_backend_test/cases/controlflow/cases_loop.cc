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
void AddGraphInputScalar(GraphProto &g, const std::string &name, TensorProto::DataType dtype) {
  ValueInfoProto *vi = g.add_input();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
  tensor_type->mutable_shape();
}

// Adds a graph input named ``name`` with a tensor type ``dtype`` and the
// given (non-scalar) ``shape``.
void AddGraphInputTensor(GraphProto &g, const std::string &name, TensorProto::DataType dtype,
                         const std::vector<int64_t> &shape) {
  ValueInfoProto *vi = g.add_input();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
  TensorShapeProto *s = tensor_type->mutable_shape();
  for (int64_t d : shape) {
    s->add_dim()->set_dim_value(d);
  }
}

void AddGraphOutputTensor(GraphProto &g, const std::string &name, TensorProto::DataType dtype) {
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

  AddGraphInputScalar(g, "i", TensorProto::DataType::INT64);
  AddGraphInputScalar(g, "cond_in", TensorProto::DataType::BOOL);

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
    t->set_data_type(TensorProto::DataType::INT64);
    t->add_dims(1);
    t->set_raw_data(utils::ByteSpan(Int64Bytes(42)));
  }

  AddGraphOutputTensor(g, "cond_out", TensorProto::DataType::BOOL);
  // scan_out: the Constant node above produces an INT64 tensor of shape [1].
  ValueInfoProto *scan_vi = g.add_output();
  scan_vi->set_name("scan_out");
  TypeProto::Tensor *scan_tt = scan_vi->ref_type().mutable_tensor_type();
  scan_tt->set_elem_type(static_cast<int>(TensorProto::DataType::INT64));
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
// Loop — loop11 mirror.
//
// Mirrors ONNX's ``test_loop11`` node case: trip-count 5 with a single
// FLOAT[1] loop-carried dependency that accumulates a prefix sum of
// ``x = [1, 2, 3, 4, 5]`` starting from ``y = [-2]`` and a single
// FLOAT[1] scan output emitting the running sum at each iteration.
//
//   res_y    = [13]
//   res_scan = [[-1], [1], [4], [8], [13]]
//
// The body subgraph is well-formed and now also numerically correct:
// ``y_out = y_in + x[iter_count]`` with ``x = [1, 2, 3, 4, 5]`` as a body
// Constant, and ``scan_out = y_out``. The ``kernel::Loop`` reference does
// not execute the body, but external backends (e.g. onnxruntime) do — so
// the body must produce the same expected outputs registered below.
// ---------------------------------------------------------------------------
namespace {

// Returns the little-endian bytes of a FLOAT value (matching the layout of
// ``Tensor::data`` for ``DataType::FLOAT``).
std::vector<uint8_t> FloatBytes(float v) {
  std::vector<uint8_t> out(sizeof(float));
  std::memcpy(out.data(), &v, sizeof(float));
  return out;
}

// Returns the little-endian bytes of ``values`` packed as FLOAT.
std::vector<uint8_t> FloatBytes(const std::vector<float> &values) {
  std::vector<uint8_t> out(values.size() * sizeof(float));
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

// Body subgraph for the loop11 case: 3 inputs (iter_count, cond_in, y_in),
// 3 outputs (cond_out, y_out, scan_out). The body computes
// ``y_out = y_in + x[iter_count]`` where ``x = [1, 2, 3, 4, 5]`` is a
// per-body Constant, and emits ``y_out`` as the scan output. With initial
// ``y = [-2]`` this produces the expected prefix-sum sequence
// ``[[-1], [1], [4], [8], [13]]`` and final ``y = [13]``.
GraphProto BuildLoop11Body() {
  GraphProto g;
  g.set_name("loop11_body");

  AddGraphInputScalar(g, "iter_count", TensorProto::DataType::INT64);
  AddGraphInputScalar(g, "cond_in", TensorProto::DataType::BOOL);
  AddGraphInputTensor(g, "y_in", TensorProto::DataType::FLOAT, {1});

  // cond_out = Identity(cond_in)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Identity");
    n->add_input("cond_in");
    n->add_output("cond_out");
  }
  // x = Constant(value=float [1, 2, 3, 4, 5])
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Constant");
    n->add_output("x");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->add_t();
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->add_dims(5);
    t->set_raw_data(utils::ByteSpan(FloatBytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f})));
  }
  // iter_1d = Unsqueeze(iter_count, axes=[0]) — turn the INT64 scalar into
  // an INT64 tensor of shape [1] suitable as Gather indices producing a
  // FLOAT[1] output.
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Unsqueeze");
    n->add_input("iter_count");
    n->add_output("iter_1d");
    AttributeProto *a = n->add_attribute();
    a->set_name("axes");
    a->set_type(AttributeProto::AttributeType::INTS);
    a->add_ints(static_cast<int64_t>(0));
  }
  // x_i = Gather(x, iter_1d, axis=0) — FLOAT[1].
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Gather");
    n->add_input("x");
    n->add_input("iter_1d");
    n->add_output("x_i");
    AttributeProto *a = n->add_attribute();
    a->set_name("axis");
    a->set_type(AttributeProto::AttributeType::INT);
    a->set_i(0);
  }
  // y_out = Add(y_in, x_i)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Add");
    n->add_input("y_in");
    n->add_input("x_i");
    n->add_output("y_out");
  }
  // scan_out = Identity(y_out)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Identity");
    n->add_input("y_out");
    n->add_output("scan_out");
  }

  AddGraphOutputTensor(g, "cond_out", TensorProto::DataType::BOOL);
  // y_out: FLOAT[1]
  {
    ValueInfoProto *vi = g.add_output();
    vi->set_name("y_out");
    TypeProto::Tensor *tt = vi->ref_type().mutable_tensor_type();
    tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
    tt->ref_shape().add_dim()->set_dim_value(1);
  }
  // scan_out: FLOAT[1]
  {
    ValueInfoProto *vi = g.add_output();
    vi->set_name("scan_out");
    TypeProto::Tensor *tt = vi->ref_type().mutable_tensor_type();
    tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
    tt->ref_shape().add_dim()->set_dim_value(1);
  }
  return g;
}

} // namespace

static void RegisterLoop11Case(std::vector<TestCase> &registry) {
  // ONNX's ``test_loop11`` model imports opset 11 — match that here so the
  // case can be located by substring against ONNX's test name.
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::Loop loop_kernel{ctx};

  // Inputs to the Loop node.
  const Tensor trip_count("trip_count", DataType::INT64, {}, Int64Bytes(5));
  const Tensor cond("cond", DataType::BOOL, {}, std::vector<uint8_t>{1});
  const Tensor y("y", DataType::FLOAT, {1}, FloatBytes(-2.0f));

  // Per-iteration scan-output values, matching ONNX's ``test_loop11``
  // expected ``res_scan = [[-1], [1], [4], [8], [13]]``.
  const std::vector<Tensor> per_iter_scan = {
      Tensor("", DataType::FLOAT, {1}, FloatBytes(-1.0f)),
      Tensor("", DataType::FLOAT, {1}, FloatBytes(1.0f)),
      Tensor("", DataType::FLOAT, {1}, FloatBytes(4.0f)),
      Tensor("", DataType::FLOAT, {1}, FloatBytes(8.0f)),
      Tensor("", DataType::FLOAT, {1}, FloatBytes(13.0f)),
  };

  // Final loop-carried state value matches ``res_y = [13]``.
  const Tensor res_y("", DataType::FLOAT, {1}, FloatBytes(13.0f));

  std::vector<Tensor> out =
      loop_kernel(trip_count, cond, /*v_initial=*/{y}, /*final_state=*/{res_y}, {per_iter_scan});

  // Build the Loop node: 3 inputs (M, cond, y) and 2 outputs (res_y, res_scan).
  NodeProto node;
  node.set_op_type("Loop");
  node.add_input("trip_count");
  node.add_input("cond");
  node.add_input("y");
  node.add_output("res_y");
  node.add_output("res_scan");

  AttributeProto *body_attr = node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildLoop11Body();

  Expect(node, {trip_count, cond, y}, out, "test_cc_loop11_carried_state", {opset}, "backend-test",
         registry);
}

namespace {

// ---------------------------------------------------------------------------
// Loop — minimal cases exercising the reference kernel's stacking semantics.
//
// Two registered cases use a Loop node with no loop-carried dependencies
// (N=0) and a single scan output (K=1) producing the constant int64 ``[42]``
// per iteration. They differ only in the value of ``M``: 3 (the scan output
// is ``[3, 1]`` containing ``[[42], [42], [42]]``) and 0 (the scan output
// is empty along its leading axis, ``[0, 1]``).
// ---------------------------------------------------------------------------
void RegisterTripCountVariants(std::vector<TestCase> &registry) {
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

} // namespace

void RegisterLoopCases(std::vector<TestCase> &registry) {
  RegisterTripCountVariants(registry);
  RegisterLoop11Case(registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

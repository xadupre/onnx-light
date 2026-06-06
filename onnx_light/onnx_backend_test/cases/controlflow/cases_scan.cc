// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/controlflow/include_controlflow_kernels.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Returns the byte representation of a packed list of floats in row-major
// order (matching ``Tensor::data`` layout).
std::vector<uint8_t> FloatBytes(const std::vector<float> &values) {
  std::vector<uint8_t> out(values.size() * sizeof(float));
  if (!values.empty()) {
    std::memcpy(out.data(), values.data(), out.size());
  }
  return out;
}

// Adds a graph input named ``name`` with a tensor type ``dtype``. The shape
// field is intentionally left unset so the rank is unspecified — ORT's body
// shape inference will determine the per-iteration rank from the enclosing
// Scan node (declaring an empty shape here would assert rank 0 and conflict
// with the inferred rank).
void AddGraphInputTensor(GraphProto &g, const std::string &name, TensorProto::DataType dtype) {
  ValueInfoProto *vi = g.add_input();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
}

void AddGraphOutputTensor(GraphProto &g, const std::string &name, TensorProto::DataType dtype) {
  ValueInfoProto *vi = g.add_output();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
}

// Builds a Scan body subgraph for our simple test case:
//
//   inputs : (x_elt [FLOAT scalar-row])
//   nodes  : y_elt = Identity(x_elt)
//   outputs: (y_elt [FLOAT scalar-row])
//
// The body has no state variables (N == 0) and produces one scan output
// (K == 1). The reference kernel does not execute the body; it is included
// in the model purely so the registered ``TestCase`` is a well-formed ONNX
// model with a valid ``Scan`` node.
GraphProto BuildSimpleScanBody() {
  GraphProto g;
  g.set_name("scan_body");

  AddGraphInputTensor(g, "x_elt", TensorProto::DataType::FLOAT);

  NodeProto *n = g.add_node();
  n->set_op_type("Identity");
  n->add_input("x_elt");
  n->add_output("y_elt");

  AddGraphOutputTensor(g, "y_elt", TensorProto::DataType::FLOAT);
  return g;
}

// Builds a ``Scan`` node with the given input/output names, a body
// subgraph attribute and ``num_scan_inputs=1``.
NodeProto MakeSimpleScanNode(const std::string &scan_input_name,
                             const std::string &scan_output_name) {
  NodeProto node;
  node.set_op_type("Scan");
  node.add_input(scan_input_name);
  node.add_output(scan_output_name);

  AttributeProto *body_attr = node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildSimpleScanBody();

  AttributeProto *num_attr = node.add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);
  return node;
}

// Builds a Scan body subgraph implementing
//
//   sum_out  = Add(sum_in, next)
//   scan_out = Identity(sum_out)
//
// for tensor element type FLOAT, used by the ``scan_sum`` / ``scan9_sum``
// reference cases. The shape of ``sum_in`` / ``next`` is intentionally left
// unspecified so the rank is inferred from the enclosing Scan node.
GraphProto BuildSumScanBody() {
  GraphProto g;
  g.set_name("scan_body");

  AddGraphInputTensor(g, "sum_in", TensorProto::DataType::FLOAT);
  AddGraphInputTensor(g, "next", TensorProto::DataType::FLOAT);

  NodeProto *add = g.add_node();
  add->set_op_type("Add");
  add->add_input("sum_in");
  add->add_input("next");
  add->add_output("sum_out");

  NodeProto *id = g.add_node();
  id->set_op_type("Identity");
  id->add_input("sum_out");
  id->add_output("scan_out");

  AddGraphOutputTensor(g, "sum_out", TensorProto::DataType::FLOAT);
  AddGraphOutputTensor(g, "scan_out", TensorProto::DataType::FLOAT);
  return g;
}

// Builds a Scan body subgraph implementing
//
//   sum_out  = Add(sum_in, next)
//   prod_out = Mul(prod_in, next)
//   scan_out = Identity(sum_out)
//
// for tensor element type FLOAT, used by the ``scan9_multi_state`` case.
GraphProto BuildMultiStateScanBody() {
  GraphProto g;
  g.set_name("scan_body");

  AddGraphInputTensor(g, "sum_in", TensorProto::DataType::FLOAT);
  AddGraphInputTensor(g, "prod_in", TensorProto::DataType::FLOAT);
  AddGraphInputTensor(g, "next", TensorProto::DataType::FLOAT);

  NodeProto *add = g.add_node();
  add->set_op_type("Add");
  add->add_input("sum_in");
  add->add_input("next");
  add->add_output("sum_out");

  NodeProto *mul = g.add_node();
  mul->set_op_type("Mul");
  mul->add_input("prod_in");
  mul->add_input("next");
  mul->add_output("prod_out");

  NodeProto *id = g.add_node();
  id->set_op_type("Identity");
  id->add_input("sum_out");
  id->add_output("scan_out");

  AddGraphOutputTensor(g, "sum_out", TensorProto::DataType::FLOAT);
  AddGraphOutputTensor(g, "prod_out", TensorProto::DataType::FLOAT);
  AddGraphOutputTensor(g, "scan_out", TensorProto::DataType::FLOAT);
  return g;
}

// Builds a ``Scan`` node from a pre-built body, the input/output names and
// ``num_scan_inputs``.
NodeProto MakeScanNodeWithBody(const std::vector<std::string> &inputs,
                               const std::vector<std::string> &outputs, GraphProto body,
                               int64_t num_scan_inputs) {
  NodeProto node;
  node.set_op_type("Scan");
  for (const auto &n : inputs) {
    node.add_input(n);
  }
  for (const auto &n : outputs) {
    node.add_output(n);
  }

  AttributeProto *body_attr = node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = std::move(body);

  AttributeProto *num_attr = node.add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(num_scan_inputs);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Scan — minimal cases exercising the reference kernel's stacking semantics.
//
// Two registered cases use a Scan node with no state variables (N=0) and a
// single scan input/output (M=K=1). The scan input is a 2-D FLOAT tensor of
// shape [T, 2]; the body identity-forwards each per-iteration slice of
// shape [2]. The first case scans T=3 iterations and stacks the per-
// iteration slices back into the original [3, 2] tensor; the second case
// uses T=0 to validate the empty leading axis (output shape [0, 2]).
// ---------------------------------------------------------------------------
void RegisterScanCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::Scan scan_kernel{ctx};

  auto register_case = [&](const std::string &test_name, int64_t trip_count) {
    NodeProto node = MakeSimpleScanNode("X", "Y");
    // Build per-iteration FLOAT slices of shape [2] = [2*t, 2*t + 1].
    std::vector<Tensor> per_iter;
    per_iter.reserve(static_cast<std::size_t>(trip_count == 0 ? 1 : trip_count));
    const std::size_t row_len = static_cast<std::size_t>(trip_count == 0 ? 1 : trip_count);
    for (std::size_t t = 0; t < row_len; ++t) {
      per_iter.push_back(
          Tensor("", DataType::FLOAT, {2},
                 FloatBytes({static_cast<float>(2 * t), static_cast<float>(2 * t + 1)})));
    }
    // Build the X input by stacking the per-iter slices along axis 0.
    std::vector<float> x_values;
    for (int64_t t = 0; t < trip_count; ++t) {
      x_values.push_back(static_cast<float>(2 * t));
      x_values.push_back(static_cast<float>(2 * t + 1));
    }
    const Tensor x("", DataType::FLOAT, {trip_count, 2}, FloatBytes(x_values));

    std::vector<Tensor> out =
        scan_kernel(trip_count, /*initial_state=*/{}, /*final_state=*/{}, {per_iter});
    Expect(node, {x}, out, test_name, {opset}, "backend-test", registry);
  };

  // T = 3 → stacked scan output has shape [3, 2] = [[0, 1], [2, 3], [4, 5]].
  register_case("test_cc_scan_basic_trip_count", 3);
  // T = 0 → stacked scan output has shape [0, 2] (empty along axis 0).
  register_case("test_cc_scan_zero_trip_count", 0);

  // -------------------------------------------------------------------------
  // Mirror of upstream onnx PR #7964 backend cases (originally Python-only):
  //   - test_scan_sum            (opset 8, batched form)
  //   - test_scan9_sum           (opset 9)
  //   - test_scan9_multi_state   (opset 9, two state variables)
  //   - test_scan9_scalar        (opset 9, scalar state and scan output)
  //
  // The reference kernel only stacks per-iteration scan outputs; we compute
  // the expected per-iteration ``scan_out`` slices and the final state
  // externally so the registered ``TestCase`` exposes a well-formed model
  // with deterministic expected outputs.
  // -------------------------------------------------------------------------

  // test_scan_sum (opset 8): outer batch dim of size 1.
  // initial=[[0,0]] [1,2], x=[[[1,2],[3,4],[5,6]]] [1,3,2]
  // y=[[9,12]] [1,2], z=[[[1,2],[4,6],[9,12]]] [1,3,2]
  {
    const OpsetId opset8 = DefaultOpset(8);
    NodeProto node = MakeScanNodeWithBody({/*sequence_lens*/ "", "initial", "x"}, {"y", "z"},
                                          BuildSumScanBody(), /*num_scan_inputs=*/1);
    const Tensor initial("", DataType::FLOAT, {1, 2}, FloatBytes({0.f, 0.f}));
    const Tensor x("", DataType::FLOAT, {1, 3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
    const Tensor y("", DataType::FLOAT, {1, 2}, FloatBytes({9.f, 12.f}));
    const Tensor z("", DataType::FLOAT, {1, 3, 2}, FloatBytes({1.f, 2.f, 4.f, 6.f, 9.f, 12.f}));
    Expect(node, {initial, x}, {y, z}, "test_scan_sum", {opset8}, "backend-test", registry);
  }

  // test_scan9_sum (opset 9): no batch dim.
  // initial=[0,0] [2], x=[[1,2],[3,4],[5,6]] [3,2]
  // y=[9,12] [2], z=[[1,2],[4,6],[9,12]] [3,2]
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    const Tensor initial("", DataType::FLOAT, {2}, FloatBytes({0.f, 0.f}));
    const Tensor x("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
    const Tensor y("", DataType::FLOAT, {2}, FloatBytes({9.f, 12.f}));
    const Tensor z("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 4.f, 6.f, 9.f, 12.f}));
    Expect(node, {initial, x}, {y, z}, "test_scan9_sum", {opset9}, "backend-test", registry);
  }

  // test_scan9_multi_state (opset 9): two state variables (sum, prod).
  // initial_sum=[0,0], initial_prod=[1,1], x=[[1,2],[3,4],[5,6]] [3,2]
  // y_sum=[9,12], y_prod=[15,48], z=[[1,2],[4,6],[9,12]] [3,2]
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial_sum", "initial_prod", "x"},
                                          {"y_sum", "y_prod", "z"}, BuildMultiStateScanBody(),
                                          /*num_scan_inputs=*/1);
    const Tensor initial_sum("", DataType::FLOAT, {2}, FloatBytes({0.f, 0.f}));
    const Tensor initial_prod("", DataType::FLOAT, {2}, FloatBytes({1.f, 1.f}));
    const Tensor x("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
    const Tensor y_sum("", DataType::FLOAT, {2}, FloatBytes({9.f, 12.f}));
    const Tensor y_prod("", DataType::FLOAT, {2}, FloatBytes({15.f, 48.f}));
    const Tensor z("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 4.f, 6.f, 9.f, 12.f}));
    Expect(node, {initial_sum, initial_prod, x}, {y_sum, y_prod, z}, "test_scan9_multi_state",
           {opset9}, "backend-test", registry);
  }

  // test_scan9_scalar (opset 9): scalar state and scalar scan element.
  // initial=0.0 [], x=[1,2,3,4,5] [5]
  // y=15.0 [], z=[1,3,6,10,15] [5]
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    const Tensor initial("", DataType::FLOAT, {}, FloatBytes({0.f}));
    const Tensor x("", DataType::FLOAT, {5}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f}));
    const Tensor y("", DataType::FLOAT, {}, FloatBytes({15.f}));
    const Tensor z("", DataType::FLOAT, {5}, FloatBytes({1.f, 3.f, 6.f, 10.f, 15.f}));
    Expect(node, {initial, x}, {y, z}, "test_scan9_scalar", {opset9}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

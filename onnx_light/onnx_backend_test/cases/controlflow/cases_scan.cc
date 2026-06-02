// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/test_case.h"

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

// Adds a graph input named ``name`` with a tensor type ``dtype`` and no
// declared shape (rank/shape will be inferred or left unspecified).
void AddGraphInputTensor(GraphProto &g, const std::string &name, TensorProto::DataType dtype) {
  ValueInfoProto *vi = g.add_input();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
  tensor_type->mutable_shape();
}

void AddGraphOutputTensor(GraphProto &g, const std::string &name, TensorProto::DataType dtype) {
  ValueInfoProto *vi = g.add_output();
  vi->set_name(name);
  TypeProto::Tensor *tensor_type = vi->ref_type().mutable_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(dtype));
  tensor_type->mutable_shape();
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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/controlflow/include_controlflow_cases.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
// (K == 1). The body is now executed end-to-end by ``RunScanNode`` (which
// delegates to ``Scan``'s body-aware overload); we still keep the
// pre-computed per-iteration outputs in the test registration so the
// stacking-only overload can be exercised independently to derive the
// expected output without depending on a runtime.
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

// Builds a Scan body subgraph computing the squared Euclidean distance
// between a fixed ``[N, D]`` matrix ``state_X`` and a per-iteration ``[D]``
// row ``x_row``:
//
//   state_X_out = Identity(state_X)            // state, propagated unchanged
//   diff        = Sub(state_X, x_row)          // broadcasts to [N, D]
//   sq          = Mul(diff, diff)              // [N, D]
//   dist        = ReduceSum(sq, axes=[1], keepdims=0)  // [N]
//
// ``state_X`` is carried as a Scan state variable (its shape is left
// unspecified so the body inherits it from the enclosing Scan node). The
// trailing axis used by ``ReduceSum`` is fixed to 1, matching a rank-2
// ``state_X`` shape. The body is opset-11-compatible (``axes`` is an
// attribute on ``ReduceSum``).
GraphProto BuildPairwiseDistanceScanBody() {
  GraphProto g;
  g.set_name("scan_body");

  AddGraphInputTensor(g, "state_X", TensorProto::DataType::FLOAT);
  AddGraphInputTensor(g, "x_row", TensorProto::DataType::FLOAT);

  NodeProto *id = g.add_node();
  id->set_op_type("Identity");
  id->add_input("state_X");
  id->add_output("state_X_out");

  NodeProto *sub = g.add_node();
  sub->set_op_type("Sub");
  sub->add_input("state_X");
  sub->add_input("x_row");
  sub->add_output("diff");

  NodeProto *mul = g.add_node();
  mul->set_op_type("Mul");
  mul->add_input("diff");
  mul->add_input("diff");
  mul->add_output("sq");

  NodeProto *red = g.add_node();
  red->set_op_type("ReduceSum");
  red->add_input("sq");
  red->add_output("dist");
  AttributeProto *axes_attr = red->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(AttributeProto::AttributeType::INTS);
  axes_attr->add_ints(1);
  AttributeProto *keepdims_attr = red->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(AttributeProto::AttributeType::INT);
  keepdims_attr->set_i(0);

  AddGraphOutputTensor(g, "state_X_out", TensorProto::DataType::FLOAT);
  AddGraphOutputTensor(g, "dist", TensorProto::DataType::FLOAT);
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
void RegisterScanCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);

  if (mode == TestMode::BENCHMARK) {
    const int64_t trip_count = 512;
    const int64_t row_len = 512;
    NodeProto node = MakeSimpleScanNode("X", "Y");
    Expect(registry, std::move(node), "test_cc_scan_benchmark", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext scan_kernel_ctx{opset};
      const Scan scan_kernel{scan_kernel_ctx};

      std::vector<float> x_values = Randn<float>({trip_count, row_len}, 4501);
      std::vector<Tensor> per_iter;
      per_iter.reserve(static_cast<std::size_t>(trip_count));
      for (int64_t t = 0; t < trip_count; ++t) {
        std::vector<float> row(x_values.begin() + t * row_len,
                               x_values.begin() + (t + 1) * row_len);
        per_iter.push_back(Tensor("", DataType::FLOAT, {row_len}, FloatBytes(row)));
      }
      const Tensor x("", DataType::FLOAT, {trip_count, row_len}, FloatBytes(x_values));
      std::vector<Tensor> out =
          scan_kernel(trip_count, /*initial_state=*/{}, /*final_state=*/{}, {per_iter});
      return IoData{{std::move(x)}, std::move(out)};
    });
    return;
  }

  auto register_case = [&registry, &opset](const std::string &test_name, int64_t trip_count) {
    NodeProto node = MakeSimpleScanNode("X", "Y");
    Expect(registry, std::move(node), test_name, {opset}, [trip_count]() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext scan_kernel_ctx{opset};
      const Scan scan_kernel{scan_kernel_ctx};

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
      return IoData{{std::move(x)}, std::move(out)};
    });
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
  // ``Scan``'s body-aware overload now executes the body
  // end-to-end inside the kernel, so the registered ``TestCase`` is a
  // well-formed model with deterministic expected outputs computed
  // externally for cross-checking.
  // -------------------------------------------------------------------------

  // test_scan_sum (opset 8): outer batch dim of size 1.
  // initial=[[0,0]] [1,2], x=[[[1,2],[3,4],[5,6]]] [1,3,2]
  // y=[[9,12]] [1,2], z=[[[1,2],[4,6],[9,12]]] [1,3,2]
  {
    const OpsetId opset8 = DefaultOpset(8);
    NodeProto node = MakeScanNodeWithBody({/*sequence_lens*/ "", "initial", "x"}, {"y", "z"},
                                          BuildSumScanBody(), /*num_scan_inputs=*/1);
    Expect(registry, std::move(node), "test_scan_sum", {opset8}, []() -> IoData {
      Tensor initial("", DataType::FLOAT, {1, 2}, FloatBytes({0.f, 0.f}));
      Tensor x("", DataType::FLOAT, {1, 3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
      Tensor y("", DataType::FLOAT, {1, 2}, FloatBytes({9.f, 12.f}));
      Tensor z("", DataType::FLOAT, {1, 3, 2}, FloatBytes({1.f, 2.f, 4.f, 6.f, 9.f, 12.f}));
      return IoData{{std::move(initial), std::move(x)}, {std::move(y), std::move(z)}};
    });
  }

  // test_scan9_sum (opset 9): no batch dim.
  // initial=[0,0] [2], x=[[1,2],[3,4],[5,6]] [3,2]
  // y=[9,12] [2], z=[[1,2],[4,6],[9,12]] [3,2]
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    Expect(registry, std::move(node), "test_scan9_sum", {opset9}, []() -> IoData {
      Tensor initial("", DataType::FLOAT, {2}, FloatBytes({0.f, 0.f}));
      Tensor x("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
      Tensor y("", DataType::FLOAT, {2}, FloatBytes({9.f, 12.f}));
      Tensor z("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 4.f, 6.f, 9.f, 12.f}));
      return IoData{{std::move(initial), std::move(x)}, {std::move(y), std::move(z)}};
    });
  }

  // test_scan9_multi_state (opset 9): two state variables (sum, prod).
  // initial_sum=[0,0], initial_prod=[1,1], x=[[1,2],[3,4],[5,6]] [3,2]
  // y_sum=[9,12], y_prod=[15,48], z=[[1,2],[4,6],[9,12]] [3,2]
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial_sum", "initial_prod", "x"},
                                          {"y_sum", "y_prod", "z"}, BuildMultiStateScanBody(),
                                          /*num_scan_inputs=*/1);
    Expect(registry, std::move(node), "test_scan9_multi_state", {opset9}, []() -> IoData {
      Tensor initial_sum("", DataType::FLOAT, {2}, FloatBytes({0.f, 0.f}));
      Tensor initial_prod("", DataType::FLOAT, {2}, FloatBytes({1.f, 1.f}));
      Tensor x("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
      Tensor y_sum("", DataType::FLOAT, {2}, FloatBytes({9.f, 12.f}));
      Tensor y_prod("", DataType::FLOAT, {2}, FloatBytes({15.f, 48.f}));
      Tensor z("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 4.f, 6.f, 9.f, 12.f}));
      return IoData{{std::move(initial_sum), std::move(initial_prod), std::move(x)},
                    {std::move(y_sum), std::move(y_prod), std::move(z)}};
    });
  }

  // test_scan9_scalar (opset 9): scalar state and scalar scan element.
  // initial=0.0 [], x=[1,2,3,4,5] [5]
  // y=15.0 [], z=[1,3,6,10,15] [5]
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    Expect(registry, std::move(node), "test_scan9_scalar", {opset9}, []() -> IoData {
      Tensor initial("", DataType::FLOAT, {}, FloatBytes({0.f}));
      Tensor x("", DataType::FLOAT, {5}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f}));
      Tensor y("", DataType::FLOAT, {}, FloatBytes({15.f}));
      Tensor z("", DataType::FLOAT, {5}, FloatBytes({1.f, 3.f, 6.f, 10.f, 15.f}));
      return IoData{{std::move(initial), std::move(x)}, {std::move(y), std::move(z)}};
    });
  }

  // -------------------------------------------------------------------------
  // Additional opset-9 cases exercising the optional Scan attributes that
  // upstream onnx does not cover in its node tests but are part of the
  // operator's spec (and now end-to-end driven by ``Scan``'s
  // body-aware overload).
  // -------------------------------------------------------------------------

  // Helper that injects a list-of-ints attribute into a NodeProto.
  auto add_ints_attr = [](NodeProto &node, const std::string &name,
                          const std::vector<int64_t> &values) {
    AttributeProto *a = node.add_attribute();
    a->set_name(name);
    a->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : values) {
      a->add_ints(v);
    }
  };

  // test_cc_scan9_input_reverse (opset 9): scan_input_directions=[1] iterates
  // x in reverse. With initial=0 and x=[1,2,3,4,5] the per-iter sums become
  // [5,9,12,14,15] and the final state stays 15.
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    add_ints_attr(node, "scan_input_directions", {1});
    Expect(registry, std::move(node), "test_cc_scan9_input_reverse", {opset9}, []() -> IoData {
      Tensor initial("", DataType::FLOAT, {}, FloatBytes({0.f}));
      Tensor x("", DataType::FLOAT, {5}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f}));
      Tensor y("", DataType::FLOAT, {}, FloatBytes({15.f}));
      Tensor z("", DataType::FLOAT, {5}, FloatBytes({5.f, 9.f, 12.f, 14.f, 15.f}));
      return IoData{{std::move(initial), std::move(x)}, {std::move(y), std::move(z)}};
    });
  }

  // test_cc_scan9_output_reverse (opset 9): scan_output_directions=[1] reverses
  // the stacked scan output. Per-iter sums [1,3,6,10,15] become [15,10,6,3,1].
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    add_ints_attr(node, "scan_output_directions", {1});
    Expect(registry, std::move(node), "test_cc_scan9_output_reverse", {opset9}, []() -> IoData {
      Tensor initial("", DataType::FLOAT, {}, FloatBytes({0.f}));
      Tensor x("", DataType::FLOAT, {5}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f}));
      Tensor y("", DataType::FLOAT, {}, FloatBytes({15.f}));
      Tensor z("", DataType::FLOAT, {5}, FloatBytes({15.f, 10.f, 6.f, 3.f, 1.f}));
      return IoData{{std::move(initial), std::move(x)}, {std::move(y), std::move(z)}};
    });
  }

  // test_cc_scan9_output_axis1 (opset 9): scan_output_axes=[1] places the
  // trip-count axis as the *trailing* axis. initial=[0,0] [2],
  // x=[[1,2],[3,4],[5,6]] [3,2] gives per-iter sums [[1,2],[4,6],[9,12]]
  // which, stacked along axis 1 of the rank-2 per-iter element [2], yields
  // shape [2, 3] = [[1, 4, 9], [2, 6, 12]] in row-major order.
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    add_ints_attr(node, "scan_output_axes", {1});
    Expect(registry, std::move(node), "test_cc_scan9_output_axis1", {opset9}, []() -> IoData {
      Tensor initial("", DataType::FLOAT, {2}, FloatBytes({0.f, 0.f}));
      Tensor x("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
      Tensor y("", DataType::FLOAT, {2}, FloatBytes({9.f, 12.f}));
      Tensor z("", DataType::FLOAT, {2, 3}, FloatBytes({1.f, 4.f, 9.f, 2.f, 6.f, 12.f}));
      return IoData{{std::move(initial), std::move(x)}, {std::move(y), std::move(z)}};
    });
  }

  // test_cc_scan9_input_axis_negative (opset 9): scan_input_axes=[-1] uses
  // the trailing dim as the scan axis. With x shape [2, 3] and the trailing
  // axis being the scan axis, the per-iter slices are columns of x:
  // [1,4], [2,5], [3,6]. Sum-accumulated, the final state is [6, 15] and
  // the per-iter sums stacked along axis 0 give shape [3, 2] =
  // [[1,4],[3,9],[6,15]].
  {
    const OpsetId opset9 = DefaultOpset(9);
    NodeProto node = MakeScanNodeWithBody({"initial", "x"}, {"y", "z"}, BuildSumScanBody(),
                                          /*num_scan_inputs=*/1);
    add_ints_attr(node, "scan_input_axes", {-1});
    Expect(registry, std::move(node), "test_cc_scan9_input_axis_negative", {opset9},
           []() -> IoData {
             Tensor initial("", DataType::FLOAT, {2}, FloatBytes({0.f, 0.f}));
             Tensor x("", DataType::FLOAT, {2, 3}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
             Tensor y("", DataType::FLOAT, {2}, FloatBytes({6.f, 15.f}));
             Tensor z("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 4.f, 3.f, 9.f, 6.f, 15.f}));
             return IoData{{std::move(initial), std::move(x)}, {std::move(y), std::move(z)}};
           });
  }

  // test_cc_scan_pairwise_distance (opset 11): computes the squared Euclidean
  // pairwise distance matrix between the rows of a fixed ``[N, D]`` matrix
  // ``X`` using a single ``Scan`` whose state carries ``X`` unchanged and
  // whose per-iteration row drives a body that returns the squared distance
  // vector to every row of ``X``.
  //
  //   N = 3, D = 2, X = [[1, 2], [3, 4], [5, 6]] (used as initial state and
  //   as the scan input — the same FLOAT [3, 2] tensor is referenced twice
  //   in the Scan node's inputs).
  //
  // Per iteration ``t`` the body computes ``dist[i] = sum_d (X[i,d] -
  // X[t,d])**2``:
  //   t=0 (x_row=[1,2]) -> dist = [0,  8, 32]
  //   t=1 (x_row=[3,4]) -> dist = [8,  0,  8]
  //   t=2 (x_row=[5,6]) -> dist = [32, 8,  0]
  //
  // Stacking ``dist`` along the leading axis yields the symmetric pairwise
  // squared-distance matrix ``dists`` of shape ``[N, N] = [3, 3]``. The
  // state output ``state_X_final`` keeps the original ``[N, D] = [3, 2]``
  // shape.
  //
  // This is the canonical Scan body that exercises shape inference through
  // broadcasting (``Sub``) and a reduction (``ReduceSum`` with ``axes=[1]``);
  // it is used by ``BackendTestCaseShapeInference.OnnxOptimInfersShapePairwiseDistanceScan``
  // to validate that the optim shape-inference pipeline correctly recovers
  // both the preserved state shape ``[N, D]`` and the stacked scan output
  // shape ``[N, N]`` from the body subgraph.
  {
    NodeProto node = MakeScanNodeWithBody({"X_state", "X_scan"}, {"state_X_final", "dists"},
                                          BuildPairwiseDistanceScanBody(),
                                          /*num_scan_inputs=*/1);
    Expect(registry, std::move(node), "test_cc_scan_pairwise_distance", {opset}, []() -> IoData {
      Tensor x_state("X_state", DataType::FLOAT, {3, 2},
                     FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
      Tensor x_scan("X_scan", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
      Tensor state_X_final("", DataType::FLOAT, {3, 2}, FloatBytes({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
      Tensor dists("", DataType::FLOAT, {3, 3},
                   FloatBytes({0.f, 8.f, 32.f, 8.f, 0.f, 8.f, 32.f, 8.f, 0.f}));
      return IoData{{std::move(x_state), std::move(x_scan)},
                    {std::move(state_X_final), std::move(dists)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

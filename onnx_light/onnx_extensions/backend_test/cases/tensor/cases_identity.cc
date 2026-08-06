// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeIdentityNode() {
  NodeProto node;
  node.set_op_type("Identity");
  node.add_input("x");
  node.add_output("y");
  return node;
}

// Returns a copy of ``t`` with a new name; lets us rename kernel outputs to
// match the ``y`` output name in :func:`MakeIdentityNode`.
Tensor Rename(Tensor t, const std::string &name) {
  t.name = name;
  return t;
}

} // namespace

void RegisterIdentityCases(std::vector<TestCase> &registry, TestMode mode) {
  // Identity has been available since opset 1; opset 14 broadened the
  // ``V`` type constraint to also cover sequence types and opset 16 added
  // optional types. The default opset chosen here matches the most common
  // tensor-only usage exercised by these reference cases.
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Identity identity_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeIdentityNode();
    Expect(registry, std::move(node), "test_cc_identity_benchmark", {opset},
           {kBenchmarkElementwiseSize}, {kBenchmarkElementwiseSize}, [identity_kernel]() -> IoData {
             Tensor x = Tensor::FromFloat("x", {kBenchmarkElementwiseSize},
                                          Randn<float>({kBenchmarkElementwiseSize}, 2001));
             Tensor y = Rename(identity_kernel(x), "y");
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // test_cc_identity — 4-D float tensor with non-trivial shape; mirrors the
  // ONNX upstream ``test_identity`` case shape (1, 3, 2, 2).
  {
    Expect(registry, MakeIdentityNode(), "test_cc_identity", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat(
          "x", {1, 3, 2, 2},
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
      const Tensor y = Rename(identity_kernel(x), "y");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_identity_scalar — 0-D input is propagated verbatim.
  {
    Expect(registry, MakeIdentityNode(), "test_cc_identity_scalar", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat("x", {}, {42.0f});
      const Tensor y = Rename(identity_kernel(x), "y");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_identity_int64 — integer dtype is preserved.
  {
    Expect(registry, MakeIdentityNode(), "test_cc_identity_int64", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromInt64("x", {2, 3}, {1, 2, 3, 4, 5, 6});
      const Tensor y = Rename(identity_kernel(x), "y");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // -------------------------------------------------------------------------
  // test_cc_identity_sequence — Identity on a Sequence<Tensor<FLOAT, [1,2,2]>>.
  // Mirrors ONNX's ``test_identity_sequence``.
  // Graph: SequenceConstruct(a, b) → Identity(seq) → ConcatFromSequence →
  // output tensor FLOAT[2, 1, 2, 2].
  // -------------------------------------------------------------------------
  {
    const std::string name = "test_cc_identity_sequence";
    const OpsetId opset14 = DefaultOpset(14);

    Tensor a = Tensor::FromFloat("a", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = Tensor::FromFloat("b", {1, 2, 2}, {2.0f, 3.0f, 1.0f, 5.0f});
    Tensor stacked("res", DataType::FLOAT, {2, 1, 2, 2}, {}); // fill below
    {
      // Expected: stack [a, b] along new axis 0 → [2, 1, 2, 2].
      std::vector<uint8_t> data;
      data.insert(data.end(), a.data.begin(), a.data.end());
      data.insert(data.end(), b.data.begin(), b.data.end());
      stacked.data = std::move(data);
    }

    TestCase tc(name, name);
    ModelProto &model = tc.emplace_model();
    InitModel(model, /*ir_version=*/9, {opset14});
    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // SequenceConstruct(a, b) → seq
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("SequenceConstruct");
      n->add_input("a");
      n->add_input("b");
      n->add_output("seq");
    }
    // Identity(seq) → seq_out
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("Identity");
      n->add_input("seq");
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

    FillValueInfo(a, *graph->add_input());
    FillValueInfo(b, *graph->add_input());
    FillValueInfo(stacked, *graph->add_output());

    DataSet ds;
    ds.inputs.push_back(a);
    ds.inputs.push_back(b);
    ds.outputs.push_back(stacked);
    tc.data_sets().emplace_back(std::move(ds));
    registry.emplace_back(std::move(tc));
  }

  // -------------------------------------------------------------------------
  // test_cc_identity_opt — Identity on Optional<Sequence<FLOAT, [5]>>.
  // Mirrors ONNX's ``test_identity_opt``.
  // Graph: SequenceConstruct(x) → Optional(seq) → Identity(opt) →
  //        OptionalGetElement → ConcatFromSequence → output FLOAT[1, 5].
  // -------------------------------------------------------------------------
  {
    const std::string name = "test_cc_identity_opt";
    const OpsetId opset16 = DefaultOpset(16);

    Tensor x = Tensor::FromFloat("x", {5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor stacked = Tensor::FromFloat("res", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    TestCase tc(name, name);
    ModelProto &model = tc.emplace_model();
    InitModel(model, /*ir_version=*/9, {opset16});
    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // SequenceConstruct(x) → seq
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("SequenceConstruct");
      n->add_input("x");
      n->add_output("seq");
    }
    // Optional(seq) → opt
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("Optional");
      n->add_input("seq");
      n->add_output("opt");
    }
    // Identity(opt) → opt_out
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("Identity");
      n->add_input("opt");
      n->add_output("opt_out");
    }
    // OptionalGetElement(opt_out) → seq_out
    {
      NodeProto *n = graph->add_node();
      n->set_op_type("OptionalGetElement");
      n->add_input("opt_out");
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

    FillValueInfo(x, *graph->add_input());
    FillValueInfo(stacked, *graph->add_output());

    DataSet ds;
    ds.inputs.push_back(x);
    ds.outputs.push_back(stacked);
    tc.data_sets().emplace_back(std::move(ds));
    registry.emplace_back(std::move(tc));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

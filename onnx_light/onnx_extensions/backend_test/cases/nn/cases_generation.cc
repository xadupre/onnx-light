// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
namespace {

void Declare(ValueInfoProto &info, const std::string &name, int32_t dtype, const Shape &shape,
             const std::string &symbol = "sequence") {
  info.set_name(name);
  auto *tensor = info.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(dtype);
  for (auto size : shape) {
    auto *dimension = tensor->mutable_shape()->add_dim();
    if (size >= 0)
      dimension->set_dim_value(size);
    else
      dimension->set_dim_param(symbol);
  }
}

NodeProto &Node(GraphProto &graph, const std::string &op, const std::vector<std::string> &inputs,
                const std::vector<std::string> &outputs) {
  auto &node = *graph.add_node();
  node.set_op_type(op);
  for (const auto &name : inputs)
    node.add_input(name);
  for (const auto &name : outputs)
    node.add_output(name);
  return node;
}

} // namespace

void RegisterGenerationCases(std::vector<TestCase> &registry) {
  for (bool persistent : {false, true}) {
    const std::string name = persistent ? "test_cc_generation_attention_persistent"
                                        : "test_cc_generation_attention_plain";
    TestCase test(name, name, TestCaseKind::MODEL);
    test.build = [persistent, name](bool) {
      BuiltCase result;
      auto &model = result.model;
      model.set_ir_version(10);
      model.set_producer_name("backend-test");
      model.add_opset_import()->set_version(23);
      auto &graph = *model.mutable_graph();
      graph.set_name(name);
      Declare(*graph.add_input(), "input_ids", TensorProto::INT64, {1, -1});
      Declare(*graph.add_output(), "logits", TensorProto::FLOAT, {1, -1, 2});
      Declare(*graph.add_value_info(), "embedded", TensorProto::FLOAT, {1, -1, 2});
      Declare(*graph.add_value_info(), "V", TensorProto::FLOAT, {1, 1, -1, 2});
      Declare(*graph.add_value_info(), "QK", TensorProto::FLOAT, {1, 1, -1, 2});
      Declare(*graph.add_value_info(), "Y", TensorProto::FLOAT, {1, 1, -1, 2});
      *graph.add_initializer() = MakeInitializer<float>("embedding", {2, 2}, {2, 0, 0, 4});
      *graph.add_initializer() = MakeInitializer<float>("zero", {}, {0});
      *graph.add_initializer() = MakeInitializer<int64_t>("axis", {1}, {1});
      Node(graph, "Gather", {"embedding", "input_ids"}, {"embedded"});
      Node(graph, "Unsqueeze", {"embedded", "axis"}, {"V"});
      Node(graph, "Mul", {"V", "zero"}, {"QK"});
      auto &attention =
          Node(graph, "Attention",
               persistent ? std::vector<std::string>{"QK", "QK", "V", "", "past_key", "past_value"}
                          : std::vector<std::string>{"QK", "QK", "V"},
               persistent ? std::vector<std::string>{"Y", "present_key", "present_value"}
                          : std::vector<std::string>{"Y"});
      AddAttribute<int64_t>(attention, "is_causal", 1);
      Node(graph, "Squeeze", {"Y", "axis"}, {"logits"});
      DataSet data;
      data.inputs.push_back(Tensor::FromInt64("input_ids", {1, 2}, {0, 1}));
      // Zero queries and keys give uniform causal attention over the embeddings.
      data.outputs.push_back(Tensor::FromFloat("logits", {1, 2, 2}, {2, 0, 1, 2}));
      if (persistent) {
        for (const auto &suffix : {"key", "value"}) {
          const std::string input = std::string("past_") + suffix;
          const std::string output = std::string("present_") + suffix;
          Declare(*graph.add_input(), input, TensorProto::FLOAT, {1, 1, -1, 2}, "past_sequence");
          Declare(*graph.add_output(), output, TensorProto::FLOAT, {1, 1, -1, 2},
                  "present_sequence");
          auto *binding = graph.add_persistent_bindings();
          binding->set_input_name(input);
          binding->set_output_name(output);
          data.inputs.push_back(Tensor::FromFloat(input, {1, 1, 0, 2}, {}));
          data.outputs.push_back(Tensor::FromFloat(output, {1, 1, 2, 2},
                                                   std::string(suffix) == "key"
                                                       ? std::vector<float>{0, 0, 0, 0}
                                                       : std::vector<float>{2, 0, 0, 4}));
        }
      }
      result.data_sets.push_back(std::move(data));
      return result;
    };
    registry.push_back(std::move(test));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

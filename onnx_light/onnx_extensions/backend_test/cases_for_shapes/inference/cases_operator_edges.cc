// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

// Covers the symbolic-shape regressions from microsoft/onnxruntime#33157
// through microsoft/onnxruntime#33161.

GraphProto &InitCase(TestCase &tc, const std::string &name, int64_t opset_version) {
  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {DefaultOpset(opset_version)});
  GraphProto *graph = model.add_graph();
  graph->set_name(name);
  return *graph;
}

void AddInt64Initializer(GraphProto &graph, const std::string &name,
                         const std::vector<int64_t> &dims, const std::vector<int64_t> &values) {
  TensorProto *tensor = graph.add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(TensorProto::DataType::INT64);
  for (int64_t dim : dims) {
    tensor->add_dims(dim);
  }
  for (int64_t value : values) {
    tensor->ref_int64_data().push_back(value);
  }
}

void RegisterShapeStartEnd(std::vector<TestCase> &registry) {
  const std::string name = "test_cc_shape_inference_shape_start_end";
  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
  GraphProto &graph = InitCase(tc, name, 15);

  NodeProto &tail = AddNode(graph, "Shape", {"input"}, {"tail"});
  AddAttribute<int64_t>(tail, "start", 1);
  NodeProto &head = AddNode(graph, "Shape", {"input"}, {"head"});
  AddAttribute<int64_t>(head, "end", -1);
  AddNode(graph, "ConstantOfShape", {"tail"}, {"tail_output"});
  AddNode(graph, "ConstantOfShape", {"head"}, {"head_output"});

  AppendValueInfo(*graph.add_input(), "input", DataType::FLOAT, {"b", "s", 8});
  AppendValueInfo(*graph.add_output(), "tail_output", DataType::FLOAT, {"s", 8});
  AppendValueInfo(*graph.add_output(), "head_output", DataType::FLOAT, {"b", "s"});
  registry.emplace_back(std::move(tc));
}

void RegisterUnevenSplit(std::vector<TestCase> &registry) {
  const std::string name = "test_cc_shape_inference_split_num_outputs_uneven";
  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
  GraphProto &graph = InitCase(tc, name, 18);

  NodeProto &split = AddNode(graph, "Split", {"input"}, {"out0", "out1", "out2"});
  AddAttribute<int64_t>(split, "axis", 1);
  AddAttribute<int64_t>(split, "num_outputs", 3);

  AppendValueInfo(*graph.add_input(), "input", DataType::FLOAT, {"b", 5});
  AppendValueInfo(*graph.add_output(), "out0", DataType::FLOAT, {"b", 2});
  AppendValueInfo(*graph.add_output(), "out1", DataType::FLOAT, {"b", 2});
  AppendValueInfo(*graph.add_output(), "out2", DataType::FLOAT, {"b", 1});
  registry.emplace_back(std::move(tc));
}

void RegisterSliceBounds(std::vector<TestCase> &registry) {
  struct SliceCase {
    int64_t dim;
    int64_t start;
    int64_t end;
    int64_t step;
    int64_t expected;
  };
  const std::vector<SliceCase> cases = {
      {3, -1, -1000, -1, 3},
      {4, 4, 1, -2, 1},
      {5, 2, -6, -1, 3},
      {5, 0, -7, 1, 0},
      {5, 3, -(int64_t{1} << 40), -1, 4},
      {5, 0, -(int64_t{1} << 40), 1, 0},
      {5, -1, int64_t{1} << 40, -1, 0},
      {0, -1, -(int64_t{1} << 40), -1, 0},
  };

  for (size_t i = 0; i < cases.size(); ++i) {
    const SliceCase &slice_case = cases[i];
    const std::string name = "test_cc_shape_inference_slice_out_of_range_" + std::to_string(i);
    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
    GraphProto &graph = InitCase(tc, name, 13);

    AddNode(graph, "Slice", {"input", "starts", "ends", "axes", "steps"}, {"output"});
    AddInt64Initializer(graph, "starts", {1}, {slice_case.start});
    AddInt64Initializer(graph, "ends", {1}, {slice_case.end});
    AddInt64Initializer(graph, "axes", {1}, {1});
    AddInt64Initializer(graph, "steps", {1}, {slice_case.step});

    AppendValueInfo(*graph.add_input(), "input", DataType::FLOAT, {"B", slice_case.dim});
    AppendValueInfo(*graph.add_output(), "output", DataType::FLOAT, {"B", slice_case.expected});
    registry.emplace_back(std::move(tc));
  }
}

void AddSequenceAt(GraphProto &graph) {
  AddInt64Initializer(graph, "position", {}, {0});
  AddNode(graph, "SequenceAt", {"sequence", "position"}, {"output"});
}

void RegisterSplitToSequenceCase(std::vector<TestCase> &registry, const std::string &suffix,
                                 int64_t axis, int64_t keepdims,
                                 const std::vector<int64_t> &split_dims,
                                 const std::vector<int64_t> &split_values,
                                 const std::vector<DimSpec> &expected) {
  const std::string name = "test_cc_shape_inference_split_to_sequence_" + suffix;
  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
  GraphProto &graph = InitCase(tc, name, 13);

  std::vector<std::string> inputs = {"input"};
  if (!split_values.empty()) {
    AddInt64Initializer(graph, "split", split_dims, split_values);
    inputs.push_back("split");
  }
  NodeProto &split = AddNode(graph, "SplitToSequence", inputs, {"sequence"});
  AddAttribute<int64_t>(split, "axis", axis);
  AddAttribute<int64_t>(split, "keepdims", keepdims);
  AddSequenceAt(graph);

  AppendValueInfo(*graph.add_input(), "input", DataType::FLOAT, {"b", 3, 5});
  AppendValueInfo(*graph.add_output(), "output", DataType::FLOAT, expected);
  registry.emplace_back(std::move(tc));
}

void RegisterSplitToSequence(std::vector<TestCase> &registry) {
  RegisterSplitToSequenceCase(registry, "squeeze", 1, 0, {}, {}, {"b", 5});
  RegisterSplitToSequenceCase(registry, "keepdims", 1, 1, {}, {}, {"b", 1, 5});
  RegisterSplitToSequenceCase(registry, "uneven_vector", -1, 1, {2}, {2, 3},
                              {"b", 3, "SequenceAt_output_dim2"});
  RegisterSplitToSequenceCase(registry, "equal_vector", 1, 1, {3}, {1, 1, 1}, {"b", 1, 5});
  RegisterSplitToSequenceCase(registry, "scalar_even", 1, 1, {}, {3}, {"b", 3, 5});
  RegisterSplitToSequenceCase(registry, "scalar_remainder", -1, 1, {}, {2},
                              {"b", 3, "SequenceAt_output_dim2"});

  const std::string name = "test_cc_shape_inference_split_to_sequence_symbolic_split";
  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
  GraphProto &graph = InitCase(tc, name, 13);
  AddNode(graph, "Shape", {"input"}, {"input_shape"});
  AddNode(graph, "Slice", {"input_shape", "starts", "ends"}, {"split"});
  AddInt64Initializer(graph, "starts", {1}, {0});
  AddInt64Initializer(graph, "ends", {1}, {1});
  NodeProto &split = AddNode(graph, "SplitToSequence", {"input", "split"}, {"sequence"});
  AddAttribute<int64_t>(split, "axis", 0);
  AddSequenceAt(graph);
  AppendValueInfo(*graph.add_input(), "input", DataType::FLOAT, {"b", 3, 5});
  AppendValueInfo(*graph.add_output(), "output", DataType::FLOAT, {"b", 3, 5});
  registry.emplace_back(std::move(tc));
}

void RegisterEinsumCase(std::vector<TestCase> &registry, const std::string &suffix,
                        const std::string &equation,
                        const std::vector<std::vector<DimSpec>> &input_shapes,
                        const std::vector<DimSpec> &output_shape) {
  const std::string name = "test_cc_shape_inference_einsum_" + suffix;
  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
  GraphProto &graph = InitCase(tc, name, 13);

  std::vector<std::string> inputs;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    inputs.push_back("input_" + std::to_string(i));
  }
  NodeProto &einsum = AddNode(graph, "Einsum", inputs, {"output"});
  AddAttribute<std::string>(einsum, "equation", equation);
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    AppendValueInfo(*graph.add_input(), inputs[i], DataType::FLOAT, input_shapes[i]);
  }
  AppendValueInfo(*graph.add_output(), "output", DataType::FLOAT, output_shape);
  registry.emplace_back(std::move(tc));
}

void RegisterEinsum(std::vector<TestCase> &registry) {
  RegisterEinsumCase(registry, "implicit_sorted", "ji", {{2, 3}}, {3, 2});
  RegisterEinsumCase(registry, "ellipsis", "...j,jk->...k", {{5, 2, 3}, {3, 4}}, {5, 2, 4});
  RegisterEinsumCase(registry, "ellipsis_broadcast", "...ij,...jk", {{5, 2, 3}, {1, 3, 4}},
                     {5, 2, 4});
  RegisterEinsumCase(registry, "letters_before_ellipsis", "b...ij->b...ji", {{2, 3, 4, 5, 6}},
                     {2, 3, 4, 6, 5});
  RegisterEinsumCase(registry, "symbolic_ellipsis_broadcast", "...i,...i,...i->...i",
                     {{"A", 4}, {"B", 4}, {"C", 4}}, {"Einsum_output_ellipsis0", 4});
}

} // namespace

void RegisterOperatorEdgeShapeInferenceCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  RegisterShapeStartEnd(registry);
  RegisterUnevenSplit(registry);
  RegisterSliceBounds(registry);
  RegisterSplitToSequence(registry);
  RegisterEinsum(registry);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

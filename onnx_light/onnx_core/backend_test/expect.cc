// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/backend_test/io_data.h"
#include "onnx_core/backend_test/test_case.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

namespace {

// Backend models are consumed by third-party runtimes, so keep their IR at the
// latest broadly supported version rather than the development IR version.
constexpr int64_t kDefaultIrVersion = 13;

// Filters node.input/node.output, keeping only entries with a non-empty name.
std::vector<std::string> NonEmpty(const utils::RepeatedField<utils::String> &names) {
  std::vector<std::string> out;
  out.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const auto &s = names[i];
    if (s.size() != 0) {
      out.emplace_back(s.data(), s.size());
    }
  }
  return out;
}

} // namespace

BuiltCase BuildSingleNodeCase(const NodeProto &node, Tensors inputs, Tensors outputs,
                              const std::string &name, const std::vector<OpsetId> &opset_imports,
                              const std::string &producer_name,
                              const std::vector<TypeSpec> &output_types, std::vector<Map> maps) {
  const auto present_inputs = NonEmpty(node.ref_input());
  const auto present_outputs = NonEmpty(node.ref_output());
  EXT_ENFORCE_INVALID(
      present_inputs.size() == inputs.size() + maps.size(),
      "BuildSingleNodeCase: number of input tensors does not match the non-empty inputs.");
  EXT_ENFORCE_INVALID(
      present_outputs.size() == outputs.size() || outputs.empty(),
      "BuildSingleNodeCase: number of output tensors does not match the non-empty outputs.");
  EXT_ENFORCE_INVALID(
      output_types.empty() || output_types.size() == present_outputs.size(),
      "BuildSingleNodeCase: output_types, when provided, must have one entry per output.");
  EXT_ENFORCE_INVALID(!outputs.empty() || present_outputs.empty() || !output_types.empty(),
                      "BuildSingleNodeCase: input-only cases require output_types.");

  // Build a lookup table from map name to its index in ``maps`` so the input
  // loop below can decide whether each present_input is a Map or a Tensor.
  // Also validate that every map name is actually a non-empty node input.
  std::unordered_set<std::string> present_input_set(present_inputs.begin(), present_inputs.end());
  std::unordered_map<std::string, size_t> map_index;
  for (size_t i = 0; i < maps.size(); ++i) {
    EXT_ENFORCE_INVALID(present_input_set.count(maps[i].name) != 0, "BuildSingleNodeCase: map '",
                        maps[i].name, "' is not a non-empty node input.");
    map_index[maps[i].name] = i;
  }

  BuiltCase built;
  InitModel(built.model, kDefaultIrVersion, opset_imports, producer_name);

  GraphProto *graph = built.model.add_graph();
  graph->set_name(name);
  graph->add_node(node);

  size_t tensor_idx = 0;
  for (const std::string &inp_name : present_inputs) {
    auto it = map_index.find(inp_name);
    if (it != map_index.end()) {
      // Map-typed input: declare it with map(key_type, value_type) TypeProto.
      const Map &m = maps[it->second];
      AppendValueInfo(*graph->add_input(), inp_name,
                      MapTypeSpec(m.key_type, TensorTypeSpec(m.value_type)));
    } else {
      inputs[tensor_idx].name = inp_name;
      FillValueInfo(inputs[tensor_idx], *graph->add_input());
      ++tensor_idx;
    }
  }
  for (size_t i = 0; i < present_outputs.size(); ++i) {
    if (output_types.empty()) {
      outputs[i].name = present_outputs[i];
      FillValueInfo(outputs[i], *graph->add_output());
    } else {
      AppendValueInfo(*graph->add_output(), present_outputs[i], output_types[i]);
    }
  }

  DataSet ds;
  ds.inputs = std::move(inputs);
  ds.outputs = std::move(outputs);
  ds.expected_outputs_generated = !ds.outputs.empty() || present_outputs.empty();
  ds.maps = std::move(maps);
  built.data_sets.emplace_back(std::move(ds));
  return built;
}

namespace {

// Copyable state shared by a lazy case builder. Holds the move-only
// ``NodeProto`` (which therefore cannot be captured directly by a copyable
// ``std::function``) plus everything else needed to build the model. Either
// ``make_io`` is set (the inputs/outputs are generated on demand, e.g. for
// benchmark cases) or the concrete ``inputs``/``outputs`` are stored directly.
struct LazyCaseState {
  NodeProto node;
  std::string name;
  std::vector<OpsetId> opset_imports;
  std::string producer_name;
  std::vector<TypeSpec> output_types;
  std::function<IoData(bool)> make_io;
  Tensors inputs;
  Tensors outputs;
};

// Builds the ``TestCase::build`` closure for a lazy case backed by ``state``.
std::function<BuiltCase(bool)> MakeLazyBuild(std::shared_ptr<LazyCaseState> state) {
  return [state](bool generate_outputs) -> BuiltCase {
    if (state->make_io) {
      IoData io = state->make_io(generate_outputs);
      EXT_ENFORCE_INVALID(!generate_outputs || io.expected_outputs_generated,
                          "Expect: callback did not generate requested expected outputs.");
      BuiltCase built = BuildSingleNodeCase(
          state->node, std::move(io.inputs), std::move(io.outputs), state->name,
          state->opset_imports, state->producer_name, state->output_types, std::move(io.maps));
      for (DataSet &ds : built.data_sets) {
        ds.expected_outputs_generated = io.expected_outputs_generated;
        if (!io.expected_outputs_generated) {
          ds.outputs.clear();
        }
      }
      return built;
    }
    return BuildSingleNodeCase(state->node, state->inputs, state->outputs, state->name,
                               state->opset_imports, state->producer_name, state->output_types);
  };
}

// Resolves the grouping tag for a node the same way :func:`Expect` does.
TestCaseTag ResolveTag(const NodeProto &node, TestCaseTag tag) {
  if (tag != TestCaseTag::NONE) {
    return tag;
  }
  const std::string node_domain = node.domain();
  if (node_domain == "ai.onnx.ml") {
    return TestCaseTag::AI_ONNX_ML;
  }
  if (node_domain == "ai.onnx.preview") {
    return TestCaseTag::AI_ONNX_PREVIEW;
  }
  if (node_domain == "ai.onnx.preview.training") {
    return TestCaseTag::AI_ONNX_PREVIEW_TRAINING;
  }
  if (node_domain == "ai.rt") {
    return TestCaseTag::AI_RT;
  }
  if (!node_domain.empty() && node_domain != "ai.onnx") {
    throw std::invalid_argument("Unsupported backend test node domain '" + node_domain + "'.");
  }
  return TestCaseTag::NONE;
}

} // namespace

void Expect(const NodeProto &node, const Tensors &inputs, const Tensors &outputs,
            const std::string &name, const std::vector<OpsetId> &opset_imports,
            const std::string &producer_name, std::vector<TestCase> &registry, TestCaseTag tag,
            const std::vector<TypeSpec> &output_types) {
  const TestCaseTag resolved_tag = ResolveTag(node, tag);

  // Validate arity eagerly so callers still get an immediate error at
  // registration time even though the model/data set are built lazily.
  const auto present_inputs = NonEmpty(node.ref_input());
  const auto present_outputs = NonEmpty(node.ref_output());
  EXT_ENFORCE_INVALID(present_inputs.size() == inputs.size(),
                      "Expect: number of input tensors does not match the non-empty inputs.");
  EXT_ENFORCE_INVALID(present_outputs.size() == outputs.size(),
                      "Expect: number of output tensors does not match the non-empty outputs.");
  EXT_ENFORCE_INVALID(
      output_types.empty() || output_types.size() == outputs.size(),
      "Expect: output_types, when provided, must have one entry per output tensor.");

  TestCase tc(name, name, TestCaseKind::NODE, resolved_tag);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;
  for (const auto &t : inputs) {
    tc.declared_input_element_counts.push_back(t.element_count());
  }
  for (const auto &t : outputs) {
    tc.declared_output_element_counts.push_back(t.element_count());
  }

  auto state = std::make_shared<LazyCaseState>();
  state->node.CopyFrom(node);
  state->name = name;
  state->opset_imports = opset_imports;
  state->producer_name = producer_name;
  state->output_types = output_types;
  state->inputs = inputs;
  state->outputs = outputs;
  tc.build = MakeLazyBuild(std::move(state));

  registry.emplace_back(std::move(tc));
}

void Expect(std::vector<TestCase> &registry, NodeProto node, std::string name,
            std::vector<OpsetId> opset_imports, std::vector<int64_t> in_counts,
            std::vector<int64_t> out_counts, std::function<IoData()> make_io,
            std::string producer_name, TestCaseTag tag, std::vector<TypeSpec> output_types) {
  const TestCaseTag resolved_tag = ResolveTag(node, tag);

  auto state = std::make_shared<LazyCaseState>();
  state->node = std::move(node);
  state->name = name;
  state->opset_imports = std::move(opset_imports);
  state->producer_name = std::move(producer_name);
  state->output_types = std::move(output_types);
  state->make_io = [make_io = std::move(make_io)](bool) mutable { return make_io(); };

  TestCase tc(name, name, TestCaseKind::NODE, resolved_tag);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;
  tc.declared_input_element_counts = std::move(in_counts);
  tc.declared_output_element_counts = std::move(out_counts);
  tc.build = MakeLazyBuild(std::move(state));

  registry.emplace_back(std::move(tc));
}

void Expect(std::vector<TestCase> &registry, NodeProto node, std::string name,
            std::vector<OpsetId> opset_imports, std::vector<int64_t> in_counts,
            std::vector<int64_t> out_counts, std::function<IoData(bool)> make_io,
            std::string producer_name, TestCaseTag tag, std::vector<TypeSpec> output_types) {
  const TestCaseTag resolved_tag = ResolveTag(node, tag);

  auto state = std::make_shared<LazyCaseState>();
  state->node = std::move(node);
  state->name = name;
  state->opset_imports = std::move(opset_imports);
  state->producer_name = std::move(producer_name);
  state->output_types = std::move(output_types);
  state->make_io = std::move(make_io);

  TestCase tc(name, name, TestCaseKind::NODE, resolved_tag);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;
  tc.declared_input_element_counts = std::move(in_counts);
  tc.declared_output_element_counts = std::move(out_counts);
  tc.build = MakeLazyBuild(std::move(state));

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"

#include <regex>
#include <stdexcept>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace backend_test {

namespace {

// Default IR version stamped on test models. Matches ``Version::IR_VERSION``
// in ``onnx_lib/onnx-data.pb.h`` but is duplicated here so this library does
// not need to depend on ``lib_onnx_lib``.
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

// Recursively builds the TypeProto ``tp`` from ``spec``.
void BuildTypeProto(const TypeSpec &spec, TypeProto &tp) {
  switch (spec.kind) {
  case TypeSpec::Kind::kTensor: {
    TypeProto::Tensor *tt = tp.add_tensor_type();
    tt->set_elem_type(spec.elem_type);
    if (spec.has_shape) {
      TensorShapeProto *sh = tt->add_shape();
      for (int64_t d : spec.shape) {
        sh->add_dim()->set_dim_value(d);
      }
    }
    break;
  }
  case TypeSpec::Kind::kSequence: {
    TypeProto::Sequence *seq = tp.add_sequence_type();
    BuildTypeProto(spec.children.front(), *seq->add_elem_type());
    break;
  }
  case TypeSpec::Kind::kMap: {
    TypeProto::Map *mp = tp.add_map_type();
    mp->set_key_type(spec.elem_type);
    BuildTypeProto(spec.children.front(), *mp->add_value_type());
    break;
  }
  }
}

} // namespace

TypeSpec TensorTypeSpec(int32_t elem_type) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kTensor;
  spec.elem_type = elem_type;
  spec.has_shape = false;
  return spec;
}

TypeSpec TensorTypeSpec(int32_t elem_type, std::vector<int64_t> shape) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kTensor;
  spec.elem_type = elem_type;
  spec.has_shape = true;
  spec.shape = std::move(shape);
  return spec;
}

TypeSpec SequenceTypeSpec(TypeSpec elem) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kSequence;
  spec.children.push_back(std::move(elem));
  return spec;
}

TypeSpec MapTypeSpec(int32_t key_type, TypeSpec value) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kMap;
  spec.elem_type = key_type;
  spec.children.push_back(std::move(value));
  return spec;
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, const TypeSpec &spec) {
  vi.set_name(name);
  BuildTypeProto(spec, *vi.add_type());
}

void InitModel(ModelProto &model, int64_t ir_version, const std::vector<OpsetId> &opset_imports,
               const std::string &producer_name) {
  model.set_ir_version(ir_version);
  model.set_producer_name(producer_name);
  for (const auto &osid : opset_imports) {
    OperatorSetIdProto proto;
    proto.set_domain(osid.domain);
    proto.set_version(osid.version);
    model.add_opset_import(proto);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<int64_t> &shape) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (int64_t d : shape) {
    sh->add_dim()->set_dim_value(d);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<DimSpec> &dims) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sh->add_dim();
    if (d.value >= 0) {
      dim->set_dim_value(d.value);
    } else if (!d.param.empty()) {
      dim->set_dim_param(d.param);
    }
    // else: leave the dim unannotated (no dim_value, no dim_param).
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, TensorProto::DataType elem_type,
                     const std::vector<DimSpec> &dims) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sh->add_dim();
    if (d.value >= 0) {
      dim->set_dim_value(d.value);
    } else if (!d.param.empty()) {
      dim->set_dim_param(d.param);
    }
    // else: leave the dim unannotated (no dim_value, no dim_param).
  }
}

void AppendDataSet(TestCase &tc, std::vector<Tensor> inputs, std::vector<Tensor> outputs) {
  DataSet ds;
  ds.inputs = std::move(inputs);
  ds.outputs = std::move(outputs);
  tc.data_sets().emplace_back(std::move(ds));
}

BuiltCase BuildSingleNodeCase(const NodeProto &node, std::vector<Tensor> inputs,
                              std::vector<Tensor> outputs, const std::string &name,
                              const std::vector<OpsetId> &opset_imports,
                              const std::string &producer_name,
                              const std::vector<TypeSpec> &output_types, std::vector<Map> maps) {
  const auto present_inputs = NonEmpty(node.ref_input());
  const auto present_outputs = NonEmpty(node.ref_output());
  EXT_ENFORCE_INVALID(
      present_inputs.size() == inputs.size() + maps.size(),
      "BuildSingleNodeCase: number of input tensors does not match the non-empty inputs.");
  EXT_ENFORCE_INVALID(
      present_outputs.size() == outputs.size(),
      "BuildSingleNodeCase: number of output tensors does not match the non-empty outputs.");
  EXT_ENFORCE_INVALID(
      output_types.empty() || output_types.size() == outputs.size(),
      "BuildSingleNodeCase: output_types, when provided, must have one entry per output tensor.");

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
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].name = present_outputs[i];
    if (output_types.empty()) {
      FillValueInfo(outputs[i], *graph->add_output());
    } else {
      AppendValueInfo(*graph->add_output(), present_outputs[i], output_types[i]);
    }
  }

  DataSet ds;
  ds.inputs = std::move(inputs);
  ds.outputs = std::move(outputs);
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
  std::function<IoData()> make_io;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
};

// Builds the ``TestCase::build`` closure for a lazy case backed by ``state``.
std::function<BuiltCase()> MakeLazyBuild(std::shared_ptr<LazyCaseState> state) {
  return [state]() -> BuiltCase {
    if (state->make_io) {
      IoData io = state->make_io();
      return BuildSingleNodeCase(state->node, std::move(io.inputs), std::move(io.outputs),
                                 state->name, state->opset_imports, state->producer_name,
                                 state->output_types, std::move(io.maps));
    }
    return BuildSingleNodeCase(state->node, state->inputs, state->outputs, state->name,
                               state->opset_imports, state->producer_name, state->output_types);
  };
}

// Resolves the grouping tag for a node the same way :func:`Expect` does.
std::string ResolveTag(const NodeProto &node, const std::string &tag) {
  if (!tag.empty()) {
    return tag;
  }
  const std::string node_domain = node.domain();
  if (!node_domain.empty() && node_domain != "ai.onnx") {
    return node_domain;
  }
  return "";
}

} // namespace

void Expect(const NodeProto &node, const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &outputs, const std::string &name,
            const std::vector<OpsetId> &opset_imports, const std::string &producer_name,
            std::vector<TestCase> &registry, const std::string &tag,
            const std::vector<TypeSpec> &output_types) {
  const std::string resolved_tag = ResolveTag(node, tag);

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

  TestCase tc(name, name, "node", resolved_tag);
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
            std::string producer_name, std::string tag, std::vector<TypeSpec> output_types) {
  const std::string resolved_tag = ResolveTag(node, tag);

  auto state = std::make_shared<LazyCaseState>();
  state->node = std::move(node);
  state->name = name;
  state->opset_imports = std::move(opset_imports);
  state->producer_name = std::move(producer_name);
  state->output_types = std::move(output_types);
  state->make_io = std::move(make_io);

  TestCase tc(name, name, "node", resolved_tag);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;
  tc.declared_input_element_counts = std::move(in_counts);
  tc.declared_output_element_counts = std::move(out_counts);
  tc.build = MakeLazyBuild(std::move(state));

  registry.emplace_back(std::move(tc));
}

void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterMap &entries) {
  if (op_type.empty()) {
    for (const auto &entry : entries) {
      entry.second(registry);
    }
    return;
  }
  auto it = entries.find(op_type);
  if (it != entries.end()) {
    it->second(registry);
  }
}

void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterModeMap &entries, TestMode mode) {
  if (op_type.empty()) {
    for (const auto &entry : entries) {
      entry.second(registry, mode);
    }
    return;
  }
  auto it = entries.find(op_type);
  if (it != entries.end()) {
    it->second(registry, mode);
  }
}

} // namespace backend_test
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE

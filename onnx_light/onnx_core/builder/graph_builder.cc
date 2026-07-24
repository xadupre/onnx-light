// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_proto/onnx_alias.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace builder {

using ::onnx_light::core::shapes::kUnknownOpsetVersion;
using ::onnx_light::core::symbolic::SymTensorFromTensorProto;
using ::onnx_light::core::symbolic::SymTensorToValueInfo;
using ::onnx_light::core::symbolic::TensorTypeToDataType;

namespace {

// Canonical name of the default ONNX operator domain.
constexpr const char *kOnnxDomain = "ai.onnx";

// Normalises a domain string so the empty (default) domain compares equal to
// the canonical ``ai.onnx`` domain used by the operator schemas.
std::string NormaliseDomain(const std::string &domain) {
  return domain.empty() ? std::string(kOnnxDomain) : domain;
}

} // namespace

GraphBuilder::GraphBuilder(std::string name, SchemaLookupFn schema_lookup)
    : name_(std::move(name)), schema_lookup_(std::move(schema_lookup)) {
  graph_.set_name(name_);
}

// ── Opset management ───────────────────────────────────────────────────

void GraphBuilder::SetOpsetVersion(const std::string &domain, int version) {
  const std::string key = NormaliseDomain(domain);
  opsets_[key] = version;
  user_opsets_.insert(key);
  compute_.Shapes().SetOpsetVersion(key, version);
}

int GraphBuilder::OpsetVersion(const std::string &domain) const {
  auto it = opsets_.find(NormaliseDomain(domain));
  return it == opsets_.end() ? kUnknownOpsetVersion : it->second;
}

// ── Name management ────────────────────────────────────────────────────

bool GraphBuilder::HasName(const std::string &name) const noexcept {
  return names_.find(name) != names_.end();
}

const std::string &GraphBuilder::ReserveName(const std::string &name) {
  if (name.empty()) {
    throw BuilderError("GraphBuilder: cannot reserve an empty name.");
  }
  const auto inserted = names_.insert(name);
  if (!inserted.second) {
    throw BuilderError("GraphBuilder: the name '" + name +
                       "' is already used; a name can never be reused.");
  }
  return *inserted.first;
}

std::string GraphBuilder::UniqueName(const std::string &prefix) {
  const std::string base = prefix.empty() ? std::string("n") : prefix;
  std::string candidate;
  do {
    candidate = base + "_" + std::to_string(auto_counter_++);
  } while (HasName(candidate));
  names_.insert(candidate);
  return candidate;
}

// ── Initializers ───────────────────────────────────────────────────────

void GraphBuilder::SeedShape(const std::string &name, SymTensor tensor) {
  compute_.Shapes().Set(name, std::move(tensor));
}

const std::string &GraphBuilder::MakeInitializer(const TensorProto &tensor) {
  const std::string &name = ReserveName(tensor.name());
  TensorProto *added = graph_.add_initializer(tensor);
  SymTensor descriptor;
  if (SymTensorFromTensorProto(*added, descriptor)) {
    SeedShape(name, std::move(descriptor));
  }
  // ``name`` references the entry stored in ``names_`` and remains valid.
  return name;
}

const std::string &GraphBuilder::MakeExternalInitializer(const std::string &name, TensorType dtype,
                                                         const std::vector<int64_t> &dims,
                                                         const std::string &location,
                                                         int64_t offset, int64_t length) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(static_cast<int>(TensorTypeToDataType(dtype)));
  for (int64_t d : dims) {
    tensor.add_dims(d);
  }
  tensor.ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  StringStringEntryProto *loc = tensor.add_external_data();
  loc->set_key("location");
  loc->set_value(location);
  StringStringEntryProto *off = tensor.add_external_data();
  off->set_key("offset");
  off->set_value(std::to_string(offset));
  StringStringEntryProto *len = tensor.add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(length));
  return MakeInitializer(tensor);
}

// ── Inputs / outputs ───────────────────────────────────────────────────

const std::string &GraphBuilder::MakeInput(const std::string &name, const SymTensor &type) {
  const std::string &reserved = ReserveName(name);
  ValueInfoProto *vi = graph_.add_input();
  vi->set_name(name);
  SymTensorToValueInfo(type, *vi);
  SeedShape(reserved, type);
  return reserved;
}

const std::string &GraphBuilder::MakeInput(const std::string &name, TensorType dtype,
                                           const SymShape &shape) {
  return MakeInput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const std::string &name, const SymTensor &type) {
  ValueInfoProto *vi = graph_.add_output();
  vi->set_name(name);
  SymTensorToValueInfo(type, *vi);
}

void GraphBuilder::MakeOutput(const std::string &name, TensorType dtype, const SymShape &shape) {
  MakeOutput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const std::string &name) {
  ValueInfoProto *vi = graph_.add_output();
  vi->set_name(name);
}

// ── Nodes ──────────────────────────────────────────────────────────────

int GraphBuilder::ResolveNodeOpset(const std::string &domain,
                                   const std::vector<const OpSchemaInfo *> &schemas) {
  const std::string key = NormaliseDomain(domain);
  auto it = opsets_.find(key);
  if (user_opsets_.find(key) != user_opsets_.end()) {
    return it->second;
  }
  if (schemas.empty()) {
    if (it != opsets_.end()) {
      return it->second;
    }
    throw BuilderError("GraphBuilder: cannot resolve the opset version for domain '" + key +
                       "'; no operator schema is available, call set_opset_version() first.");
  }
  int op_latest = std::numeric_limits<int>::min();
  for (const OpSchemaInfo *schema : schemas) {
    op_latest = std::max(op_latest, schema->since_version);
  }
  const int target = it != opsets_.end() ? std::max(it->second, op_latest) : op_latest;
  opsets_[key] = target;
  compute_.Shapes().SetOpsetVersion(key, target);
  return target;
}

bool GraphBuilder::ShapeFunctionAvailable(const NodeProto &node) const {
  const std::string domain = node.domain().empty() ? std::string() : node.domain().value();
  const std::string key = NormaliseDomain(domain) + ":" + node.op_type().value();
  const auto &table = core::shapes::DispatchTable();
  return table.find(key) != table.end();
}

std::vector<std::string> GraphBuilder::MakeNode(const std::string &op_type,
                                                const std::vector<std::string> &inputs,
                                                const std::vector<std::string> &outputs,
                                                const std::string &domain, const std::string &name,
                                                const std::vector<AttributeProto> &attributes) {
  // Every non-empty input must reference a value that already exists. Empty
  // strings denote skipped optional inputs and are allowed.
  for (const std::string &input : inputs) {
    if (!input.empty() && !HasName(input)) {
      throw BuilderError("GraphBuilder: node '" + op_type + "' reads the unknown value '" + input +
                         "'.");
    }
  }

  // Resolve the opset version and the matching schema (if any).
  std::vector<OpSchemaInfo> history =
      schema_lookup_ ? schema_lookup_(op_type) : std::vector<OpSchemaInfo>();
  std::vector<const OpSchemaInfo *> domain_schemas;
  const std::string normalised_domain = NormaliseDomain(domain);
  for (const OpSchemaInfo &schema : history) {
    if (NormaliseDomain(schema.domain) == normalised_domain) {
      domain_schemas.push_back(&schema);
    }
  }
  const int opset = ResolveNodeOpset(domain, domain_schemas);
  const OpSchemaInfo *schema = nullptr;
  for (const OpSchemaInfo *candidate : domain_schemas) {
    if (candidate->since_version <= opset &&
        (schema == nullptr || candidate->since_version > schema->since_version)) {
      schema = candidate;
    }
  }

  // Determine how many outputs the node produces and validate against the
  // schema when one is available.
  std::size_t num_outputs = outputs.size();
  if (schema != nullptr) {
    if (!outputs.empty()) {
      const std::size_t min_output = static_cast<std::size_t>(std::max(0, schema->min_output));
      const std::size_t max_output = static_cast<std::size_t>(std::max(0, schema->max_output));
      if (outputs.size() < min_output || outputs.size() > max_output) {
        throw BuilderError("GraphBuilder: node '" + op_type + "' was given " +
                           std::to_string(outputs.size()) + " outputs but the schema at opset " +
                           std::to_string(opset) + " expects between " +
                           std::to_string(min_output) + " and " + std::to_string(max_output) + ".");
      }
    } else {
      num_outputs = static_cast<std::size_t>(std::max(1, schema->min_output));
    }
  } else if (outputs.empty()) {
    num_outputs = 1;
  }

  // Assign the final output names, generating fresh names where needed.
  std::vector<std::string> resolved_outputs;
  resolved_outputs.reserve(num_outputs);
  for (std::size_t i = 0; i < num_outputs; ++i) {
    if (i < outputs.size() && !outputs[i].empty()) {
      resolved_outputs.push_back(ReserveName(outputs[i]));
    } else {
      resolved_outputs.push_back(UniqueName(op_type));
    }
  }

  // Build the node, attach its attributes and append it to the graph.
  NodeProto node;
  node.set_op_type(op_type);
  if (!domain.empty()) {
    node.set_domain(domain);
  }
  if (!name.empty()) {
    node.set_name(name);
  }
  for (const std::string &input : inputs) {
    node.add_input(input);
  }
  for (const std::string &output : resolved_outputs) {
    node.add_output(output);
  }
  for (const AttributeProto &attribute : attributes) {
    node.add_attribute(attribute);
  }
  NodeProto *stored = graph_.add_node(node);

  // Run incremental shape inference for the new node when a shape function is
  // registered for its operator; unregistered operators simply leave their
  // outputs without an inferred descriptor.
  if (ShapeFunctionAvailable(*stored)) {
    compute_.Shapes().ComputeShapeNode(*stored);
  }

  return resolved_outputs;
}

// ── Queries ────────────────────────────────────────────────────────────

bool GraphBuilder::HasShape(const std::string &name) const { return compute_.Shapes().Has(name); }

const SymTensor &GraphBuilder::GetShape(const std::string &name) const {
  return compute_.Shapes().Get(name);
}

// ── Finalization ───────────────────────────────────────────────────────

void GraphBuilder::Finalize(GraphProto &graph) {
  const auto tags = compute_.ComputeValueAndNodeTags(graph);
  compute_.ComputeInPlaceReuseGraph(graph, compute_.Shapes(), /*allow_input_overwrite=*/false,
                                    tags.first);
  compute_.ComputePeakMemory(graph, device_);
  // Writes inferred shapes (value_info), in-place / release-after / shape-tag
  // metadata and per-node peak memory.
  compute_.WriteToGraph(graph);
  // Additionally records the per-node and per-value tags.
  core::compute::WriteValueAndNodeTagsToMetadata(graph);
}

GraphProto GraphBuilder::ToGraph() {
  GraphProto graph = graph_;
  Finalize(graph);
  return graph;
}

ModelProto GraphBuilder::ToModel(int64_t ir_version) {
  ModelProto model;
  model.set_ir_version(ir_version > 0 ? ir_version : static_cast<int64_t>(IR_VERSION));
  model.set_producer_name("onnx-light");
  for (const auto &entry : opsets_) {
    model.add_opset(entry.first, entry.second);
  }
  *model.mutable_graph() = ToGraph();
  return model;
}

FunctionProto GraphBuilder::ToFunction(const std::string &domain) {
  if (graph_.initializer().size() > 0) {
    throw BuilderError("GraphBuilder: a FunctionProto cannot carry initializers; remove them or "
                       "produce a model / graph instead.");
  }
  GraphProto graph = ToGraph();
  FunctionProto function;
  function.set_name(name_);
  if (!domain.empty()) {
    function.set_domain(domain);
  }
  for (int i = 0; i < graph.input().size(); ++i) {
    function.add_input(graph.input(static_cast<std::size_t>(i)).name().value());
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    function.add_output(graph.output(static_cast<std::size_t>(i)).name().value());
  }
  for (const auto &entry : opsets_) {
    function.add_opset(entry.first, entry.second);
  }
  for (int i = 0; i < graph.node().size(); ++i) {
    function.add_node(graph.node(static_cast<std::size_t>(i)));
  }
  return function;
}

} // namespace builder
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE

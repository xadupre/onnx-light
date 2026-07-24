// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_proto/onnx_alias.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace builder {

using ::onnx_light::core::shapes::kUnknownOpsetVersion;
using ::onnx_light::core::shapes::ShapesContext;
using ::onnx_light::core::symbolic::SymTensorFromTensorProto;
using ::onnx_light::core::symbolic::SymTensorFromValueInfo;
using ::onnx_light::core::symbolic::SymTensorToValueInfo;
using ::onnx_light::core::symbolic::TensorTypeToDataType;

GraphBuilder::GraphBuilder(std::string name, SchemaLookupFn schema_lookup)
    : name_(std::move(name)), schema_lookup_(std::move(schema_lookup)) {}

GraphBuilder::~GraphBuilder() = default;
GraphBuilder::GraphBuilder(GraphBuilder &&) noexcept = default;
GraphBuilder &GraphBuilder::operator=(GraphBuilder &&) noexcept = default;

// ── Opset management ───────────────────────────────────────────────────

void GraphBuilder::SetOpsetVersion(const std::string &domain, int version) {
  const std::string key = ShapesContext::NormaliseDomain(domain);
  opsets_[key] = version;
  user_opsets_.insert(key);
  compute_.Shapes().SetOpsetVersion(key, version);
}

int GraphBuilder::OpsetVersion(const std::string &domain) const {
  auto it = opsets_.find(ShapesContext::NormaliseDomain(domain));
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
  const std::string tensor_name = tensor.name().value();
  const std::string &reserved = ReserveName(tensor_name);
  TensorProto &added = initializers_.Set(tensor_name, tensor);
  SymTensor descriptor;
  if (SymTensorFromTensorProto(added, descriptor)) {
    SeedShape(reserved, std::move(descriptor));
  }
  // ``reserved`` references the entry stored in ``names_`` and remains valid.
  return reserved;
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

const std::string &GraphBuilder::MakeInput(const ValueInfoProto &value_info) {
  // Reserve the name first so a duplicate is rejected before ``inputs_`` is
  // mutated, leaving the builder unchanged on error.
  const std::string &reserved = ReserveName(value_info.name().value());
  inputs_.push_back(value_info);
  SymTensor descriptor;
  if (SymTensorFromValueInfo(value_info, descriptor)) {
    SeedShape(reserved, std::move(descriptor));
  }
  return reserved;
}

const std::string &GraphBuilder::MakeInput(const std::string &name, const SymTensor &type) {
  ValueInfoProto vi;
  vi.set_name(name);
  SymTensorToValueInfo(type, vi);
  const std::string &reserved = ReserveName(name);
  inputs_.push_back(std::move(vi));
  SeedShape(reserved, type);
  return reserved;
}

const std::string &GraphBuilder::MakeInput(const std::string &name, TensorType dtype,
                                           const SymShape &shape) {
  return MakeInput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const ValueInfoProto &value_info) { outputs_.push_back(value_info); }

void GraphBuilder::MakeOutput(const std::string &name, const SymTensor &type) {
  ValueInfoProto vi;
  vi.set_name(name);
  SymTensorToValueInfo(type, vi);
  outputs_.push_back(std::move(vi));
}

void GraphBuilder::MakeOutput(const std::string &name, TensorType dtype, const SymShape &shape) {
  MakeOutput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const std::string &name) {
  ValueInfoProto vi;
  vi.set_name(name);
  outputs_.push_back(std::move(vi));
}

// ── Nodes ──────────────────────────────────────────────────────────────

int GraphBuilder::ResolveNodeOpset(const std::string &domain,
                                   const std::vector<const LightOpSchema *> &schemas) {
  const std::string key = ShapesContext::NormaliseDomain(domain);
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
  for (const LightOpSchema *schema : schemas) {
    op_latest = std::max(op_latest, schema->since_version());
  }
  const int target = it != opsets_.end() ? std::max(it->second, op_latest) : op_latest;
  opsets_[key] = target;
  compute_.Shapes().SetOpsetVersion(key, target);
  return target;
}

bool GraphBuilder::ShapeFunctionAvailable(const NodeProto &node) const {
  const std::string domain = node.domain().empty() ? std::string() : node.domain().value();
  const std::string key = ShapesContext::NormaliseDomain(domain) + ":" + node.op_type().value();
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
  std::vector<LightOpSchema> history =
      schema_lookup_ ? schema_lookup_(op_type) : std::vector<LightOpSchema>();
  std::vector<const LightOpSchema *> domain_schemas;
  const std::string normalised_domain = ShapesContext::NormaliseDomain(domain);
  for (const LightOpSchema &schema : history) {
    if (ShapesContext::NormaliseDomain(schema.domain()) == normalised_domain) {
      domain_schemas.push_back(&schema);
    }
  }
  const int opset = ResolveNodeOpset(domain, domain_schemas);
  const LightOpSchema *schema = nullptr;
  for (const LightOpSchema *candidate : domain_schemas) {
    if (candidate->since_version() <= opset &&
        (schema == nullptr || candidate->since_version() > schema->since_version())) {
      schema = candidate;
    }
  }

  // Determine how many outputs the node produces and validate against the
  // schema when one is available.
  std::size_t num_outputs = outputs.size();
  if (schema != nullptr) {
    if (!outputs.empty()) {
      const std::size_t min_output = static_cast<std::size_t>(std::max(0, schema->min_output()));
      const std::size_t max_output = static_cast<std::size_t>(std::max(0, schema->max_output()));
      if (outputs.size() < min_output || outputs.size() > max_output) {
        throw BuilderError("GraphBuilder: node '" + op_type + "' was given " +
                           std::to_string(outputs.size()) + " outputs but the schema at opset " +
                           std::to_string(opset) + " expects between " +
                           std::to_string(min_output) + " and " + std::to_string(max_output) + ".");
      }
    } else {
      num_outputs = static_cast<std::size_t>(std::max(1, schema->min_output()));
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

  // Build the node, attach its attributes and append it to the node list.
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
  nodes_.push_back(std::move(node));
  const NodeProto &stored = nodes_.back();

  // Run incremental shape inference for the new node when a shape function is
  // registered for its operator; unregistered operators simply leave their
  // outputs without an inferred descriptor.
  if (ShapeFunctionAvailable(stored)) {
    compute_.Shapes().ComputeShapeNode(stored);
  }

  return resolved_outputs;
}

// ── Local functions / subgraphs ─────────────────────────────────────────

GraphBuilder &GraphBuilder::MakeLocalFunction(const std::string &name, const std::string &domain) {
  if (local_functions_.Contains(name)) {
    throw BuilderError("GraphBuilder: a local function named '" + name + "' already exists.");
  }
  ReserveName(name);
  auto child = std::make_unique<GraphBuilder>(name, schema_lookup_);
  if (!domain.empty()) {
    // Register the function domain on both the parent (so nodes that call the
    // function resolve their opset and the model imports the domain) and the
    // nested builder (so its emitted FunctionProto imports it too).
    SetOpsetVersion(domain, 1);
    child->SetOpsetVersion(domain, 1);
  }
  GraphBuilder &ref = *child;
  local_functions_.Set(name, std::move(child));
  return ref;
}

GraphBuilder &GraphBuilder::MakeSubgraph(const std::string &name) {
  if (subgraphs_.Contains(name)) {
    throw BuilderError("GraphBuilder: a subgraph named '" + name + "' already exists.");
  }
  ReserveName(name);
  auto child = std::make_unique<GraphBuilder>(name, schema_lookup_);
  GraphBuilder &ref = *child;
  subgraphs_.Set(name, std::move(child));
  return ref;
}

// ── Queries ────────────────────────────────────────────────────────────

bool GraphBuilder::HasShape(const std::string &name) const { return compute_.Shapes().Has(name); }

const SymTensor &GraphBuilder::GetShape(const std::string &name) const {
  return compute_.Shapes().Get(name);
}

GraphProto GraphBuilder::BuildGraph() const {
  GraphProto graph;
  graph.set_name(name_);
  for (const ValueInfoProto &input : inputs_) {
    graph.add_input(input);
  }
  for (const auto &entry : initializers_) {
    graph.add_initializer(entry.second);
  }
  for (const NodeProto &node : nodes_) {
    graph.add_node(node);
  }
  for (const ValueInfoProto &output : outputs_) {
    graph.add_output(output);
  }
  return graph;
}

// ── Description ─────────────────────────────────────────────────────────

std::string GraphBuilder::ToString() const {
  std::ostringstream os;
  os << "GraphBuilder(name=" << name_ << ")\n";

  os << "  opsets:";
  if (opsets_.empty()) {
    os << " <none>";
  } else {
    for (const auto &entry : opsets_) {
      os << " " << entry.first << "=" << entry.second;
    }
  }
  os << "\n";

  os << "  inputs (" << inputs_.size() << "):\n";
  for (const ValueInfoProto &input : inputs_) {
    const std::string &value_name = input.name().value();
    os << "    " << value_name;
    SymTensor descriptor;
    if (SymTensorFromValueInfo(input, descriptor)) {
      os << ": " << descriptor.ToString();
    }
    os << "\n";
  }

  os << "  initializers (" << initializers_.Size() << "):\n";
  for (const auto &entry : initializers_) {
    const TensorProto &tensor = entry.second;
    os << "    " << entry.first << ": dtype=" << static_cast<int>(tensor.data_type())
       << ", shape=[";
    for (int i = 0; i < tensor.dims().size(); ++i) {
      if (i != 0) {
        os << ",";
      }
      os << tensor.dims()[i];
    }
    os << "]";
    if (tensor.data_location() == TensorProto::DataLocation::EXTERNAL) {
      os << " (external)";
    }
    os << "\n";
  }

  os << "  nodes (" << nodes_.size() << "):\n";
  for (const NodeProto &node : nodes_) {
    os << "    ";
    const std::string node_domain = node.domain().empty() ? std::string() : node.domain().value();
    if (!node_domain.empty()) {
      os << node_domain << ".";
    }
    os << node.op_type().value() << "(";
    for (int i = 0; i < node.input().size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << node.input(static_cast<std::size_t>(i));
    }
    os << ") -> ";
    for (int i = 0; i < node.output().size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << node.output(static_cast<std::size_t>(i));
    }
    if (!node.name().empty()) {
      os << "  [name=" << node.name().value() << "]";
    }
    os << "\n";
  }

  os << "  outputs (" << outputs_.size() << "):\n";
  for (const ValueInfoProto &output : outputs_) {
    const std::string &value_name = output.name().value();
    os << "    " << value_name;
    SymTensor descriptor;
    if (SymTensorFromValueInfo(output, descriptor)) {
      os << ": " << descriptor.ToString();
    }
    os << "\n";
  }

  if (local_functions_.Size() != 0) {
    os << "  local functions (" << local_functions_.Size() << "):\n";
    for (const auto &entry : local_functions_) {
      os << "    " << entry.first << "\n";
    }
  }
  if (subgraphs_.Size() != 0) {
    os << "  subgraphs (" << subgraphs_.Size() << "):\n";
    for (const auto &entry : subgraphs_) {
      os << "    " << entry.first << "\n";
    }
  }

  return os.str();
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
  GraphProto graph = BuildGraph();
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
  for (const auto &entry : local_functions_) {
    model.add_function(entry.second->ToFunction(entry.second->name()));
  }
  return model;
}

FunctionProto GraphBuilder::ToFunction(const std::string &domain) {
  if (initializers_.Size() != 0) {
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

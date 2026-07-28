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
#include "onnx_proto/onnx_helper.h"

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

GraphBuilder::GraphBuilder(const ModelProto &model, SchemaLookupFn schema_lookup)
    : name_(model.graph().name().empty() ? std::string("graph") : model.graph().name().value()),
      schema_lookup_(std::move(schema_lookup)) {
  for (const auto &opset : model.opset_import()) {
    SetOpsetVersion(opset.domain().empty() ? std::string() : opset.domain().value(),
                    static_cast<int>(opset.version()));
  }
  for (const auto &function : model.functions()) {
    const std::string function_name = function.name().value();
    const std::string function_domain =
        function.domain().empty() ? std::string() : function.domain().value();
    GraphBuilder &local = MakeLocalFunction(function_name, function_domain);
    for (const auto &entry : opsets_) {
      local.SetOpsetVersion(entry.first, entry.second);
    }
    local.ImportFunction(function);
  }
  ImportGraph(model.graph());
}

GraphBuilder::~GraphBuilder() = default;
GraphBuilder::GraphBuilder(GraphBuilder &&) noexcept = default;
GraphBuilder &GraphBuilder::operator=(GraphBuilder &&) noexcept = default;

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
  const std::string tensor_name = tensor.name().value();
  const std::string &reserved = ReserveName(tensor_name);
  initializers_.push_back(tensor);
  TensorProto &added = initializers_.back();
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
  } else {
    SeedShape(reserved, SymTensor());
  }
  return reserved;
}

const std::string &GraphBuilder::MakeInput(const std::string &name, const SymTensor &type) {
  ValueInfoProto vi;
  vi.set_name(name);
  SymTensorToValueInfo(type, vi);
  const std::string &reserved = ReserveName(name);
  inputs_.add() = std::move(vi);
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
  outputs_.add() = std::move(vi);
}

void GraphBuilder::MakeOutput(const std::string &name, TensorType dtype, const SymShape &shape) {
  MakeOutput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const std::string &name) {
  ValueInfoProto vi;
  vi.set_name(name);
  outputs_.add() = std::move(vi);
}

bool GraphBuilder::HasGraphReferenceSuffix(const std::string &name) {
  static constexpr const char *kSuffix = "_ref";
  const std::size_t suffix_len = 4;
  return name.size() >= suffix_len &&
         name.compare(name.size() - suffix_len, suffix_len, kSuffix) == 0;
}

utils::RepeatedProtoField<AttributeProto> GraphBuilder::ImportAttributes(const NodeProto &node) {
  const auto has_graph_content = [](const GraphProto &graph) {
    return !graph.name().empty() || graph.input().size() > 0 || graph.output().size() > 0 ||
           graph.node().size() > 0 || graph.initializer().size() > 0 ||
           graph.value_info().size() > 0;
  };
  utils::RepeatedProtoField<AttributeProto> imported;
  imported.reserve(node.attribute().size());
  for (const auto &attribute : node.attribute()) {
    const bool has_single_graph_payload =
        (attribute.has_g() && has_graph_content(attribute.g())) ||
        (attribute.graphs().size() > 0 &&
         has_graph_content(attribute.graphs()[static_cast<std::size_t>(0)]));
    if (attribute.type() == AttributeProto::AttributeType::GRAPH && has_single_graph_payload) {
      const GraphProto &graph =
          attribute.has_g() ? attribute.g() : attribute.graphs()[static_cast<std::size_t>(0)];
      std::string base = graph.name().empty() ? attribute.name().value() : graph.name().value();
      if (base.empty()) {
        base = "subgraph";
      }
      std::string subgraph_name = base;
      int suffix = 0;
      while (HasSubgraph(subgraph_name) || HasName(subgraph_name)) {
        subgraph_name = base + "_" + std::to_string(suffix++);
      }
      GraphBuilder &subgraph = MakeSubgraph(subgraph_name);
      subgraph.ImportGraph(graph);
      AttributeProto ref;
      ref.set_name(attribute.name().value() + "_ref");
      ref.set_type(AttributeProto::AttributeType::STRING);
      ref.set_s(subgraph_name);
      imported.push_back(std::move(ref));
      continue;
    }
    if (attribute.type() == AttributeProto::AttributeType::GRAPHS &&
        attribute.graphs().size() > 0) {
      AttributeProto refs;
      refs.set_name(attribute.name().value() + "_ref");
      refs.set_type(AttributeProto::AttributeType::STRINGS);
      const auto &graphs = attribute.graphs();
      for (int i = 0; i < graphs.size(); ++i) {
        const GraphProto &graph = graphs[static_cast<std::size_t>(i)];
        std::string base = graph.name().empty() ? attribute.name().value() : graph.name().value();
        if (base.empty()) {
          base = "subgraph";
        }
        std::string subgraph_name = base;
        int suffix = 0;
        while (HasSubgraph(subgraph_name) || HasName(subgraph_name)) {
          subgraph_name = base + "_" + std::to_string(suffix++);
        }
        GraphBuilder &subgraph = MakeSubgraph(subgraph_name);
        subgraph.ImportGraph(graph);
        refs.add_strings(subgraph_name);
      }
      imported.push_back(std::move(refs));
      continue;
    }
    imported.push_back(attribute);
  }
  return imported;
}

void GraphBuilder::ImportGraph(const GraphProto &graph) {
  for (const auto &input : graph.input()) {
    MakeInput(input);
  }
  for (const auto &initializer : graph.initializer()) {
    MakeInitializer(initializer);
  }
  for (const auto &node : graph.node()) {
    std::vector<std::string> inputs;
    inputs.reserve(node.input().size());
    for (int i = 0; i < node.input().size(); ++i) {
      inputs.push_back(node.input(static_cast<std::size_t>(i)));
    }
    std::vector<std::string> outputs;
    outputs.reserve(node.output().size());
    for (int i = 0; i < node.output().size(); ++i) {
      outputs.push_back(node.output(static_cast<std::size_t>(i)));
    }
    MakeNode(node.op_type().value(), inputs, outputs,
             node.domain().empty() ? std::string() : node.domain().value(),
             node.name().empty() ? std::string() : node.name().value(), ImportAttributes(node));
  }
  for (const auto &output : graph.output()) {
    MakeOutput(output);
  }
}

void GraphBuilder::ImportFunction(const FunctionProto &function) {
  for (const auto &opset : function.opset_import()) {
    SetOpsetVersion(opset.domain().empty() ? std::string() : opset.domain().value(),
                    static_cast<int>(opset.version()));
  }
  for (int i = 0; i < function.input().size(); ++i) {
    ValueInfoProto value_info;
    value_info.set_name(function.input(static_cast<std::size_t>(i)));
    MakeInput(value_info);
  }
  for (const auto &node : function.node()) {
    std::vector<std::string> inputs;
    inputs.reserve(node.input().size());
    for (int i = 0; i < node.input().size(); ++i) {
      inputs.push_back(node.input(static_cast<std::size_t>(i)));
    }
    std::vector<std::string> outputs;
    outputs.reserve(node.output().size());
    for (int i = 0; i < node.output().size(); ++i) {
      outputs.push_back(node.output(static_cast<std::size_t>(i)));
    }
    MakeNode(node.op_type().value(), inputs, outputs,
             node.domain().empty() ? std::string() : node.domain().value(),
             node.name().empty() ? std::string() : node.name().value(), ImportAttributes(node));
  }
  for (int i = 0; i < function.output().size(); ++i) {
    MakeOutput(function.output(static_cast<std::size_t>(i)));
  }
}

void GraphBuilder::MaterializeGraphReferences(NodeProto &node) const {
  for (auto &attribute : node.ref_attribute()) {
    if (!HasGraphReferenceSuffix(attribute.name().value())) {
      continue;
    }
    const std::string base_name =
        attribute.name().value().substr(0, attribute.name().value().size() - 4);
    if (attribute.type() == AttributeProto::AttributeType::STRING && attribute.has_s()) {
      const std::string ref_name = attribute.s().value();
      if (HasSubgraph(ref_name)) {
        AttributeProto graph_attribute;
        graph_attribute.set_name(base_name);
        graph_attribute.set_type(AttributeProto::AttributeType::GRAPH);
        *graph_attribute.mutable_g() = Subgraph(ref_name).BuildGraph();
        attribute = std::move(graph_attribute);
      }
    } else if (attribute.type() == AttributeProto::AttributeType::STRINGS &&
               attribute.strings().size() > 0) {
      AttributeProto graphs_attribute;
      graphs_attribute.set_name(base_name);
      graphs_attribute.set_type(AttributeProto::AttributeType::GRAPHS);
      const auto &refs = attribute.strings();
      for (int i = 0; i < refs.size(); ++i) {
        const std::string ref_name = refs[static_cast<std::size_t>(i)];
        if (HasSubgraph(ref_name)) {
          *graphs_attribute.add_graphs() = Subgraph(ref_name).BuildGraph();
        }
      }
      attribute = std::move(graphs_attribute);
    }
  }
}

// ── Nodes ──────────────────────────────────────────────────────────────

int GraphBuilder::ResolveNodeOpset(const std::string &domain,
                                   const std::vector<LightOpSchema> &schemas) {
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
  for (const LightOpSchema &schema : schemas) {
    op_latest = std::max(op_latest, schema.since_version());
  }
  const int target = it != opsets_.end() ? std::max(it->second, op_latest) : op_latest;
  opsets_[key] = target;
  compute_.Shapes().SetOpsetVersion(key, target);
  return target;
}

const std::vector<LightOpSchema> &
GraphBuilder::DomainSchemas(const std::string &op_type, const std::string &normalised_domain) {
  static const std::vector<LightOpSchema> kEmpty;
  auto table_it = schema_table_.find(op_type);
  if (table_it == schema_table_.end()) {
    std::unordered_map<std::string, std::vector<LightOpSchema>> by_domain;
    if (schema_lookup_) {
      for (LightOpSchema &schema : schema_lookup_(op_type)) {
        by_domain[NormaliseDomain(schema.domain())].push_back(std::move(schema));
      }
    }
    table_it = schema_table_.emplace(op_type, std::move(by_domain)).first;
  }
  auto domain_it = table_it->second.find(normalised_domain);
  return domain_it == table_it->second.end() ? kEmpty : domain_it->second;
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
                                                const utils::RepeatedProtoField<AttributeProto>
                                                    &attributes) {
  // Every non-empty input must reference a value that already exists. Empty
  // strings denote skipped optional inputs and are allowed.
  for (const std::string &input : inputs) {
    if (!input.empty() && !HasName(input)) {
      throw BuilderError("GraphBuilder: node '" + op_type + "' reads the unknown value '" + input +
                         "'.");
    }
  }

  // Resolve the opset version and the matching schema (if any) via the cached
  // per-operator, per-domain lookup table.
  const std::string normalised_domain = NormaliseDomain(domain);
  const std::vector<LightOpSchema> &domain_schemas = DomainSchemas(op_type, normalised_domain);
  const int opset = ResolveNodeOpset(domain, domain_schemas);
  const LightOpSchema *schema = nullptr;
  for (const LightOpSchema &candidate : domain_schemas) {
    if (candidate.since_version() <= opset &&
        (schema == nullptr || candidate.since_version() > schema->since_version())) {
      schema = &candidate;
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
  NodeProto attribute_source;
  for (const AttributeProto &attribute : attributes) {
    attribute_source.add_attribute(attribute);
  }
  const utils::RepeatedProtoField<AttributeProto> normalized_attributes =
      ImportAttributes(attribute_source);
  for (const AttributeProto &attribute : normalized_attributes) {
    node.add_attribute(attribute);
  }
  NodeProto &stored = nodes_.add();
  stored = std::move(node);

  // Run incremental shape inference for the new node when a shape function is
  // registered for its operator; unregistered operators simply leave their
  // outputs without an inferred descriptor.
  bool has_graph_reference_attribute = false;
  bool has_unbound_graph_attribute = false;
  for (const auto &attribute : stored.attribute()) {
    if (HasGraphReferenceSuffix(attribute.name().value())) {
      has_graph_reference_attribute = true;
      break;
    }
    if ((attribute.type() == AttributeProto::AttributeType::GRAPH && !attribute.has_g()) ||
        (attribute.type() == AttributeProto::AttributeType::GRAPHS &&
         attribute.graphs().size() == 0)) {
      has_unbound_graph_attribute = true;
    }
  }
  if (!has_graph_reference_attribute && !has_unbound_graph_attribute &&
      ShapeFunctionAvailable(stored)) {
    compute_.Shapes().ComputeShapeNode(stored);
  }

  // Keep the semantic value / node tags ("shape_tag") and the in-place buffer
  // reuse up to date with the graph built so far. Release-after lifetime is
  // deferred to the finalizers (see Finalize).
  RefreshAnnotations();

  return resolved_outputs;
}

// ── Local functions / subgraphs ─────────────────────────────────────────

GraphBuilder *
GraphBuilder::FindNamedBuilder(const std::vector<std::unique_ptr<GraphBuilder>> &builders,
                               const std::string &name) {
  // Linear scan: local functions and subgraphs are few, so a vector keeps the
  // declaration order without the overhead of a separate name -> index map.
  for (const auto &builder : builders) {
    if (builder->name() == name) {
      return builder.get();
    }
  }
  return nullptr;
}

GraphBuilder &
GraphBuilder::NamedBuilderOrThrow(const std::vector<std::unique_ptr<GraphBuilder>> &builders,
                                  const std::string &name, const char *kind) {
  GraphBuilder *found = FindNamedBuilder(builders, name);
  if (found == nullptr) {
    throw BuilderError("GraphBuilder: no " + std::string(kind) + " named '" + name + "'.");
  }
  return *found;
}

GraphBuilder &GraphBuilder::MakeLocalFunction(const std::string &name, const std::string &domain) {
  if (FindNamedBuilder(local_functions_, name) != nullptr) {
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
  child->function_domain_ = domain;
  GraphBuilder &ref = *child;
  local_functions_.push_back(std::move(child));
  return ref;
}

GraphBuilder &GraphBuilder::MakeSubgraph(const std::string &name) {
  if (FindNamedBuilder(subgraphs_, name) != nullptr) {
    throw BuilderError("GraphBuilder: a subgraph named '" + name + "' already exists.");
  }
  ReserveName(name);
  auto child = std::make_unique<GraphBuilder>(name, schema_lookup_);
  for (const auto &entry : opsets_) {
    child->SetOpsetVersion(entry.first, entry.second);
  }
  GraphBuilder &ref = *child;
  subgraphs_.push_back(std::move(child));
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
  for (const TensorProto &initializer : initializers_) {
    graph.add_initializer(initializer);
  }
  for (const NodeProto &node : nodes_) {
    NodeProto materialized = node;
    MaterializeGraphReferences(materialized);
    graph.add_node(materialized);
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

  os << "  initializers (" << initializers_.size() << "):\n";
  for (const TensorProto &tensor : initializers_) {
    os << "    " << tensor.name().value() << ": dtype=" << static_cast<int>(tensor.data_type())
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

  if (!local_functions_.empty()) {
    os << "  local functions (" << local_functions_.size() << "):\n";
    for (const auto &function : local_functions_) {
      os << "    " << function->name() << "\n";
    }
  }
  if (!subgraphs_.empty()) {
    os << "  subgraphs (" << subgraphs_.size() << "):\n";
    for (const auto &subgraph : subgraphs_) {
      os << "    " << subgraph->name() << "\n";
    }
  }

  return os.str();
}

// ── Finalization ───────────────────────────────────────────────────────

void GraphBuilder::RefreshAnnotations() {
  // Value / node tags and in-place reuse both need the whole graph built so
  // far, so recompute them over the current content. This keeps the builder's
  // live analysis state (queried through Compute()) consistent as nodes are
  // added; the finalizers recompute the same information (plus the release-after
  // lifetime and peak memory) over the complete graph before writing it out.
  const GraphProto graph = BuildGraph();
  const auto tags = compute_.ComputeValueAndNodeTags(graph);
  compute_.ComputeInPlaceReuseGraph(graph, compute_.Shapes(), /*allow_input_overwrite=*/false,
                                    tags.first);
}

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
  for (const auto &function : local_functions_) {
    model.add_function(function->ToFunction(function->function_domain_));
  }
  return model;
}

FunctionProto GraphBuilder::ToFunction(const std::string &domain) {
  if (!initializers_.empty()) {
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

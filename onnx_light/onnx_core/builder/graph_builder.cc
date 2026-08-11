// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

#include "onnx_core/compute/constant_info.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_proto/onnx_alias.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

using ::onnx_light::core::shapes::kUnknownOpsetVersion;
using ::onnx_light::core::shapes::ShapesContext;
using ::onnx_light::core::symbolic::SymTensorFromTensorProto;
using ::onnx_light::core::symbolic::SymTensorFromValueInfo;
using ::onnx_light::core::symbolic::SymTensorToValueInfo;
using ::onnx_light::core::symbolic::TensorTypeToDataType;

namespace {

// Compares two packed payload fields byte-for-byte without copying.
template <typename T>
bool SamePackedData(const utils::RepeatedField<T> &lhs, const utils::RepeatedField<T> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return lhs.size() == 0 || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0;
}

// Compares the external_data key/value entries of two tensors.
bool SameExternalData(const TensorProto &lhs, const TensorProto &rhs) {
  const auto &left = lhs.external_data();
  const auto &right = rhs.external_data();
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (!(left[i].key() == right[i].key()) || !(left[i].value() == right[i].value())) {
      return false;
    }
  }
  return true;
}

// Returns true when two initializers carry byte-for-byte identical content
// (ignoring their names). Cheap discriminators -- element type then shape --
// are checked first so mismatched tensors are rejected before any payload is
// compared. Inline payloads (raw_data or typed data) and external data are all
// supported. Nothing is copied and nothing is serialized.
bool SameInitializerContent(const TensorProto &lhs, const TensorProto &rhs) {
  if (lhs.data_type() != rhs.data_type()) {
    return false;
  }
  if (lhs.dims().values() != rhs.dims().values()) {
    return false;
  }
  const int lhs_location = lhs.has_data_location() ? static_cast<int>(lhs.data_location()) : 0;
  const int rhs_location = rhs.has_data_location() ? static_cast<int>(rhs.data_location()) : 0;
  if (lhs_location != rhs_location) {
    return false;
  }
  if (lhs.raw_data() != rhs.raw_data()) {
    return false;
  }
  if (!SamePackedData(lhs.float_data(), rhs.float_data()) ||
      !SamePackedData(lhs.int32_data(), rhs.int32_data()) ||
      !SamePackedData(lhs.int64_data(), rhs.int64_data()) ||
      !SamePackedData(lhs.double_data(), rhs.double_data()) ||
      !SamePackedData(lhs.uint64_data(), rhs.uint64_data())) {
    return false;
  }
  if (lhs.string_data().values() != rhs.string_data().values()) {
    return false;
  }
  return SameExternalData(lhs, rhs);
}

// Materializes an initializer ``TensorProto`` (owning its payload) from a
// runtime tensor produced by constant folding. Non-STRING tensors copy their
// little-endian byte buffer into ``raw_data``; STRING tensors copy their
// elements into ``string_data``.
TensorProto ProtoFromRuntimeTensor(const core::runtime::Tensor &tensor, const std::string &name) {
  TensorProto proto;
  proto.set_name(name);
  proto.set_data_type(tensor.data_type);
  for (int64_t dim : tensor.shape) {
    proto.add_dims(dim);
  }
  if (static_cast<TensorProto::DataType>(tensor.data_type) == TensorProto::DataType::STRING) {
    for (const std::string &value : tensor.string_data) {
      proto.add_string_data(utils::String(value));
    }
  } else {
    proto.set_raw_data(tensor.bytes(), tensor.size_bytes());
  }
  return proto;
}

// Returns true when ``node`` carries a control-flow subgraph, either inline
// (GRAPH / GRAPHS attribute) or through a builder ``*_ref`` reference. Such
// nodes are never folded.
bool NodeCarriesSubgraph(const NodeProto &node) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.type() == AttributeProto::AttributeType::GRAPH ||
        attribute.type() == AttributeProto::AttributeType::GRAPHS) {
      return true;
    }
    const std::string &attr_name = attribute.name().value();
    if (attr_name.size() >= 4 && attr_name.compare(attr_name.size() - 4, 4, "_ref") == 0) {
      return true;
    }
  }
  return false;
}

} // namespace

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
  // Seed the incremental annotations: an initializer is a "weight", is kept
  // from in-place reuse, and is a constant value.
  compute_.SeedValueTag(reserved, "weight");
  compute_.SeedReuseInput(reserved, /*is_graph_input=*/false, /*is_initializer=*/true,
                          /*allow_input_overwrite=*/false);
  compute_.SeedConstant(reserved);
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
  SeedInputAnnotations(reserved);
  return reserved;
}

const std::string &GraphBuilder::MakeInput(const std::string &name, const SymTensor &type) {
  ValueInfoProto vi;
  vi.set_name(name);
  SymTensorToValueInfo(type, vi);
  const std::string &reserved = ReserveName(name);
  inputs_.add() = std::move(vi);
  SeedShape(reserved, type);
  SeedInputAnnotations(reserved);
  return reserved;
}

const std::string &GraphBuilder::MakeInput(const std::string &name, TensorType dtype,
                                           const SymShape &shape) {
  return MakeInput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const ValueInfoProto &value_info) {
  outputs_.push_back(value_info);
  compute_.SeedReuseOutput(value_info.name().value());
}

void GraphBuilder::MakeOutput(const std::string &name, const SymTensor &type) {
  ValueInfoProto vi;
  vi.set_name(name);
  SymTensorToValueInfo(type, vi);
  outputs_.add() = std::move(vi);
  compute_.SeedReuseOutput(name);
}

void GraphBuilder::MakeOutput(const std::string &name, TensorType dtype, const SymShape &shape) {
  MakeOutput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const std::string &name) {
  ValueInfoProto vi;
  vi.set_name(name);
  outputs_.add() = std::move(vi);
  compute_.SeedReuseOutput(name);
}

// Seeds the incremental annotations for a declared graph input: it is a
// "weight" and is kept from in-place reuse.
void GraphBuilder::SeedInputAnnotations(const std::string &name) {
  compute_.SeedValueTag(name, "weight");
  compute_.SeedReuseInput(name, /*is_graph_input=*/true, /*is_initializer=*/false,
                          /*allow_input_overwrite=*/false);
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

std::vector<std::string>
GraphBuilder::MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs, const std::string &domain,
                       const std::string &name,
                       const utils::RepeatedProtoField<AttributeProto> &attributes) {
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
  // reuse up to date with the graph built so far. Both are maintained
  // incrementally: only the appended node and its immediate dependents are
  // touched, so importing N nodes stays linear instead of O(N^2). Release-after
  // lifetime and per-node memory are finalized over the complete graph (see
  // Finalize), since a value's last use is only settled once no further node
  // can extend it.
  const std::size_t node_index = nodes_.size() - 1;
  compute_.AppendNodeTags(nodes_, node_index);
  compute_.AppendNodeReuse(stored, node_index, compute_.Shapes());
  compute_.AppendNodeConstant(stored, node_index);

  return resolved_outputs;
}

std::vector<GraphBuilder *> GraphBuilder::ReferencedSubgraphs(const NodeProto &node) const {
  std::vector<GraphBuilder *> referenced;
  for (const auto &attribute : node.attribute()) {
    if (!HasGraphReferenceSuffix(attribute.name().value())) {
      continue;
    }
    if (attribute.type() == AttributeProto::AttributeType::STRING && attribute.has_s()) {
      GraphBuilder *subgraph = FindNamedBuilder(subgraphs_, attribute.s().value());
      if (subgraph != nullptr) {
        referenced.push_back(subgraph);
      }
    } else if (attribute.type() == AttributeProto::AttributeType::STRINGS) {
      const auto &refs = attribute.strings();
      for (int i = 0; i < refs.size(); ++i) {
        GraphBuilder *subgraph =
            FindNamedBuilder(subgraphs_, std::string(refs[static_cast<std::size_t>(i)]));
        if (subgraph != nullptr) {
          referenced.push_back(subgraph);
        }
      }
    }
  }
  return referenced;
}

void GraphBuilder::CollectImplicitInputs(std::unordered_set<std::string> &out) const {
  // Every value name defined in this scope: graph inputs, initializers and the
  // outputs of the nodes accumulated so far.
  std::unordered_set<std::string> defined;
  for (const ValueInfoProto &input : inputs_) {
    defined.insert(input.name().value());
  }
  for (const TensorProto &initializer : initializers_) {
    defined.insert(initializer.name().value());
  }
  for (const NodeProto &node : nodes_) {
    for (int i = 0; i < node.output().size(); ++i) {
      std::string name(node.output(static_cast<std::size_t>(i)));
      if (!name.empty()) {
        defined.insert(std::move(name));
      }
    }
  }
  const auto reference = [&](const std::string &name) {
    if (!name.empty() && defined.find(name) == defined.end()) {
      out.insert(name);
    }
  };
  for (const NodeProto &node : nodes_) {
    std::vector<std::string> refs;
    CollectNodeReferences(node, refs);
    for (const std::string &ref : refs) {
      reference(ref);
    }
  }
  for (const ValueInfoProto &output : outputs_) {
    reference(output.name().value());
  }
}

void GraphBuilder::CollectNodeReferences(const NodeProto &node,
                                         std::vector<std::string> &refs) const {
  for (int i = 0; i < node.input().size(); ++i) {
    std::string name(node.input(static_cast<std::size_t>(i)));
    if (!name.empty()) {
      refs.push_back(std::move(name));
    }
  }
  for (const GraphBuilder *subgraph : ReferencedSubgraphs(node)) {
    std::unordered_set<std::string> implicit_inputs;
    subgraph->CollectImplicitInputs(implicit_inputs);
    for (const std::string &name : implicit_inputs) {
      refs.push_back(name);
    }
  }
}

std::size_t GraphBuilder::RemoveUnusedNodes() {
  // Prune nested builders first: a leaner subgraph or local function may stop
  // referencing outer values, which can in turn make an owning node unused.
  std::size_t removed = 0;
  for (const auto &function : local_functions_) {
    removed += function->RemoveUnusedNodes();
  }
  for (const auto &subgraph : subgraphs_) {
    removed += subgraph->RemoveUnusedNodes();
  }

  const std::size_t num_nodes = nodes_.size();
  if (num_nodes == 0) {
    return removed;
  }

  // Map each produced value name to the index of the node that produces it.
  std::unordered_map<std::string, std::size_t> producer;
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const NodeProto &node = nodes_[i];
    for (int j = 0; j < node.output().size(); ++j) {
      std::string name(node.output(static_cast<std::size_t>(j)));
      if (!name.empty()) {
        producer.emplace(std::move(name), i);
      }
    }
  }

  // Recursively mark every node reachable backwards from the graph outputs.
  std::vector<bool> live(num_nodes, false);
  std::function<void(std::size_t)> mark_node = [&](std::size_t index) {
    if (live[index]) {
      return;
    }
    live[index] = true;
    std::vector<std::string> refs;
    CollectNodeReferences(nodes_[index], refs);
    for (const std::string &ref : refs) {
      auto it = producer.find(ref);
      if (it != producer.end()) {
        mark_node(it->second);
      }
    }
  };
  for (const ValueInfoProto &output : outputs_) {
    auto it = producer.find(output.name().value());
    if (it != producer.end()) {
      mark_node(it->second);
    }
  }

  // Rebuild the node list, keeping the live nodes in their original order.
  utils::RepeatedProtoField<NodeProto> kept;
  kept.reserve(num_nodes);
  std::size_t local_removed = 0;
  for (std::size_t i = 0; i < num_nodes; ++i) {
    if (live[i]) {
      kept.push_back(std::move(nodes_[i]));
    } else {
      ++local_removed;
    }
  }
  nodes_ = std::move(kept);
  return removed + local_removed;
}

namespace {

// Returns true when the node is a plain default-domain Identity that forwards
// its single input to its single output, both under a non-empty name.
bool IsForwardingIdentity(const NodeProto &node) {
  const std::string domain = node.domain().empty() ? std::string() : node.domain().value();
  if (!domain.empty() || node.op_type().value() != "Identity") {
    return false;
  }
  if (node.input().size() != 1 || node.output().size() != 1) {
    return false;
  }
  return !std::string(node.input(0)).empty() && !std::string(node.output(0)).empty();
}

// Returns true when ``node`` can be dropped in favour of a survivor whose
// outputs are ``survivor_outputs``. The survivor must expose at least as many
// outputs, and every position where ``node`` names a non-empty output must be
// backed by a non-empty survivor output so the consumer can be rewired.
bool CanReuseOutputs(const NodeProto &node, const std::vector<std::string> &survivor_outputs) {
  if (static_cast<std::size_t>(node.output().size()) > survivor_outputs.size()) {
    return false;
  }
  for (int i = 0; i < node.output().size(); ++i) {
    if (!std::string(node.output(static_cast<std::size_t>(i))).empty() &&
        survivor_outputs[static_cast<std::size_t>(i)].empty()) {
      return false;
    }
  }
  return true;
}

} // namespace

std::size_t GraphBuilder::RemoveIdentityNodes() {
  // Prune nested builders first so a collapsed identity inside a subgraph or
  // local function is handled in its own scope.
  std::size_t removed = 0;
  for (const auto &function : local_functions_) {
    removed += function->RemoveIdentityNodes();
  }
  for (const auto &subgraph : subgraphs_) {
    removed += subgraph->RemoveIdentityNodes();
  }

  const std::size_t num_nodes = nodes_.size();
  if (num_nodes == 0) {
    return removed;
  }

  // Declared graph outputs must keep their own name, so an Identity that
  // produces one is never dropped.
  std::unordered_set<std::string> output_names;
  for (const ValueInfoProto &output : outputs_) {
    output_names.insert(output.name().value());
  }

  // Drop every forwarding Identity, recording output -> input so consumers can
  // be rewired to the identity's source.
  std::unordered_map<std::string, std::string> rename;
  utils::RepeatedProtoField<NodeProto> kept;
  kept.reserve(num_nodes);
  std::size_t local_removed = 0;
  for (NodeProto &node : nodes_) {
    if (IsForwardingIdentity(node) &&
        output_names.find(std::string(node.output(0))) == output_names.end()) {
      rename.emplace(std::string(node.output(0)), std::string(node.input(0)));
      ++local_removed;
      continue;
    }
    kept.push_back(std::move(node));
  }
  nodes_ = std::move(kept);

  // Collapse chains of identities: an identity input can itself be another
  // removed identity's output, so follow each rename target to its final value.
  for (auto &entry : rename) {
    std::string target = entry.second;
    std::unordered_set<std::string> seen{entry.first};
    auto it = rename.find(target);
    while (it != rename.end() && seen.insert(target).second) {
      target = it->second;
      it = rename.find(target);
    }
    entry.second = target;
  }

  // Rewrite every consumer of a dropped identity output, descending into
  // subgraphs whose bodies capture values from this enclosing scope.
  RewriteInitializerReferences(rename);

  return removed + local_removed;
}

std::size_t GraphBuilder::RemoveDuplicateNodes() {
  // Prune nested builders first so each scope collapses its own duplicates.
  std::size_t removed = 0;
  for (const auto &function : local_functions_) {
    removed += function->RemoveDuplicateNodes();
  }
  for (const auto &subgraph : subgraphs_) {
    removed += subgraph->RemoveDuplicateNodes();
  }

  const std::size_t num_nodes = nodes_.size();
  if (num_nodes == 0) {
    return removed;
  }

  // A node whose output is a declared graph output is never dropped, because
  // the graph must keep producing a value under that name (it can still act as
  // the survivor for a later duplicate).
  std::unordered_set<std::string> output_names;
  for (const ValueInfoProto &output : outputs_) {
    output_names.insert(output.name().value());
  }

  // Maps a node signature to the outputs of the first node that carried it (the
  // survivor). Nodes are visited in insertion order, which is topological, so a
  // duplicate is always seen after its survivor.
  std::unordered_map<std::string, std::vector<std::string>> seen;
  // Maps a dropped node output to the surviving output that replaces it. A
  // survivor is never itself dropped, so this mapping has no chains.
  std::unordered_map<std::string, std::string> rename;
  const auto resolve = [&](const std::string &name) -> std::string {
    auto it = rename.find(name);
    return it != rename.end() ? it->second : name;
  };

  utils::RepeatedProtoField<NodeProto> kept;
  kept.reserve(num_nodes);
  std::size_t local_removed = 0;
  for (NodeProto &node : nodes_) {
    // Resolve inputs against earlier-dropped duplicates so a chain of identical
    // sub-computations collapses in a single forward pass.
    std::vector<std::string> resolved_inputs;
    resolved_inputs.reserve(static_cast<std::size_t>(node.input().size()));
    for (int i = 0; i < node.input().size(); ++i) {
      resolved_inputs.push_back(resolve(std::string(node.input(static_cast<std::size_t>(i)))));
    }

    bool produces_output = false;
    for (int i = 0; i < node.output().size(); ++i) {
      if (output_names.find(std::string(node.output(static_cast<std::size_t>(i)))) !=
          output_names.end()) {
        produces_output = true;
        break;
      }
    }

    // Nodes referencing subgraphs carry per-node-unique ``*_ref`` attribute
    // names, so their signatures never collide and control flow is never merged.
    const std::string signature = node.Signature(resolved_inputs);
    auto it = seen.find(signature);
    if (it != seen.end() && !produces_output && CanReuseOutputs(node, it->second)) {
      // Drop the duplicate and rewire each of its outputs onto the survivor.
      for (int i = 0; i < node.output().size(); ++i) {
        std::string out(node.output(static_cast<std::size_t>(i)));
        if (!out.empty()) {
          rename.emplace(std::move(out), it->second[static_cast<std::size_t>(i)]);
        }
      }
      ++local_removed;
      continue;
    }

    // Keep the node; register it as the survivor for its signature only when it
    // is the first one, so later duplicates collapse onto the earliest copy.
    if (it == seen.end()) {
      std::vector<std::string> outputs;
      outputs.reserve(static_cast<std::size_t>(node.output().size()));
      for (int i = 0; i < node.output().size(); ++i) {
        outputs.push_back(std::string(node.output(static_cast<std::size_t>(i))));
      }
      seen.emplace(signature, std::move(outputs));
    }
    kept.push_back(std::move(node));
  }
  nodes_ = std::move(kept);

  // Rewrite every consumer of a dropped node's output, descending into
  // subgraphs whose bodies capture values from this enclosing scope.
  RewriteInitializerReferences(rename);

  return removed + local_removed;
}

GraphBuilder *GraphBuilder::FindCalledFunction(const std::vector<GraphBuilder *> &functions,
                                               const NodeProto &node) {
  const std::string node_domain = node.domain().empty() ? std::string() : node.domain().value();
  const std::string normalised = NormaliseDomain(node_domain);
  for (GraphBuilder *function : functions) {
    if (function->name() == node.op_type().value() &&
        NormaliseDomain(function->function_domain_) == normalised) {
      return function;
    }
  }
  return nullptr;
}

std::size_t GraphBuilder::CountFunctionCalls(const std::string &name,
                                             const std::string &domain) const {
  const std::string normalised = NormaliseDomain(domain);
  std::size_t count = 0;
  for (const NodeProto &node : nodes_) {
    const std::string node_domain = node.domain().empty() ? std::string() : node.domain().value();
    if (node.op_type().value() == name && NormaliseDomain(node_domain) == normalised) {
      ++count;
    }
  }
  for (const auto &subgraph : subgraphs_) {
    count += subgraph->CountFunctionCalls(name, domain);
  }
  return count;
}

void GraphBuilder::AppendInlinedBody(GraphBuilder &function, const NodeProto &call,
                                     utils::RepeatedProtoField<NodeProto> &out) {
  // Build the value rename map: formal inputs/outputs are rewired to the call
  // inputs/outputs, everything else the body defines gets a fresh, unused name.
  std::unordered_map<std::string, std::string> rename;
  const std::size_t num_inputs =
      std::min<std::size_t>(function.inputs_.size(), call.input().size());
  for (std::size_t i = 0; i < num_inputs; ++i) {
    rename[function.inputs_[i].name().value()] = std::string(call.input(i));
  }
  const std::size_t num_outputs =
      std::min<std::size_t>(function.outputs_.size(), call.output().size());
  for (std::size_t j = 0; j < num_outputs; ++j) {
    rename[function.outputs_[j].name().value()] = std::string(call.output(j));
  }

  // Clone any function initializers under a fresh name (functions rarely carry
  // initializers, but the builder allows them before finalization).
  for (const TensorProto &initializer : function.initializers_) {
    const std::string &old_name = initializer.name().value();
    if (rename.find(old_name) != rename.end()) {
      continue;
    }
    std::string new_name = function.name() + "_" + old_name;
    int suffix = 0;
    while (HasName(new_name)) {
      new_name = function.name() + "_" + old_name + "_" + std::to_string(suffix++);
    }
    TensorProto clone = initializer;
    clone.set_name(new_name);
    MakeInitializer(clone);
    rename.emplace(old_name, new_name);
  }

  // Fresh names for every remaining body value (node outputs not yet mapped).
  for (const NodeProto &node : function.nodes_) {
    for (int j = 0; j < node.output().size(); ++j) {
      std::string produced(node.output(static_cast<std::size_t>(j)));
      if (produced.empty() || rename.find(produced) != rename.end()) {
        continue;
      }
      const std::string fresh = UniqueName(function.name() + "_" + produced);
      rename.emplace(std::move(produced), fresh);
    }
  }

  const auto remap = [&](const std::string &value) -> std::string {
    if (value.empty()) {
      return value;
    }
    auto it = rename.find(value);
    return it != rename.end() ? it->second : value;
  };

  for (const NodeProto &body : function.nodes_) {
    NodeProto node;
    node.set_op_type(body.op_type().value());
    if (!body.domain().empty()) {
      node.set_domain(body.domain().value());
    }
    if (!body.name().empty()) {
      node.set_name(body.name().value());
    }
    for (int i = 0; i < body.input().size(); ++i) {
      node.add_input(remap(std::string(body.input(static_cast<std::size_t>(i)))));
    }
    for (int i = 0; i < body.output().size(); ++i) {
      node.add_output(remap(std::string(body.output(static_cast<std::size_t>(i)))));
    }
    for (const AttributeProto &attribute : body.attribute()) {
      if (!attribute.ref_attr_name().empty()) {
        // The body attribute references a function attribute; resolve it against
        // the value carried by the call node, or drop it (operator default).
        const std::string reference = attribute.ref_attr_name().value();
        const AttributeProto *actual = nullptr;
        for (const AttributeProto &call_attribute : call.attribute()) {
          if (call_attribute.name().value() == reference) {
            actual = &call_attribute;
            break;
          }
        }
        if (actual != nullptr) {
          AttributeProto resolved = *actual;
          resolved.set_name(attribute.name().value());
          node.add_attribute(std::move(resolved));
        }
        continue;
      }
      if (HasGraphReferenceSuffix(attribute.name().value())) {
        throw BuilderError("GraphBuilder: cannot inline local function '" + function.name() +
                           "'; inlining a function whose body contains control-flow subgraphs is "
                           "not supported.");
      }
      node.add_attribute(attribute);
    }
    out.push_back(std::move(node));
  }
}

std::size_t GraphBuilder::InlineFunctionCalls(const std::vector<GraphBuilder *> &functions) {
  std::size_t inlined = 0;
  // Subgraph bodies live in their own scope but may call the enclosing local
  // functions, so inline them too using the same function table.
  for (const auto &subgraph : subgraphs_) {
    inlined += subgraph->InlineFunctionCalls(functions);
  }
  if (functions.empty() || nodes_.size() == 0) {
    return inlined;
  }

  // Rebuild the node list, expanding every call. Repeat to a fixed point: a
  // pasted body may itself call another local function.
  bool changed = true;
  while (changed) {
    changed = false;
    utils::RepeatedProtoField<NodeProto> kept;
    kept.reserve(nodes_.size());
    for (NodeProto &node : nodes_) {
      GraphBuilder *function = FindCalledFunction(functions, node);
      if (function != nullptr) {
        AppendInlinedBody(*function, node, kept);
        ++inlined;
        changed = true;
      } else {
        kept.push_back(std::move(node));
      }
    }
    nodes_ = std::move(kept);
  }
  return inlined;
}

std::size_t GraphBuilder::InlineLocalFunctions(
    const std::vector<std::pair<std::string, std::string>> &include,
    const std::vector<std::pair<std::string, std::string>> &exclude) {
  if (!include.empty() && !exclude.empty()) {
    throw BuilderError("GraphBuilder: InlineLocalFunctions accepts an include list or an exclude "
                       "list, not both.");
  }
  // A ``(domain, name)`` pair matches a function when both components match,
  // where an empty domain matches every domain and an empty name every name.
  const auto matches = [](const std::vector<std::pair<std::string, std::string>> &entries,
                          const std::string &domain, const std::string &name) {
    for (const auto &entry : entries) {
      if ((entry.first.empty() || entry.first == domain) &&
          (entry.second.empty() || entry.second == name)) {
        return true;
      }
    }
    return false;
  };
  const auto selected = [&](const std::string &domain, const std::string &name) {
    if (!include.empty()) {
      return matches(include, domain, name);
    }
    return !matches(exclude, domain, name);
  };

  std::vector<GraphBuilder *> functions;
  functions.reserve(local_functions_.size());
  for (const auto &function : local_functions_) {
    if (selected(function->function_domain_, function->name())) {
      functions.push_back(function.get());
    }
  }

  // Record which functions are called anywhere (the calling graph, its
  // subgraphs and the other function bodies) before expanding: only these are
  // eligible for removal once fully inlined, so a function that is never called
  // (e.g. exported for external use) is left in place.
  std::unordered_set<std::string> called_before;
  for (GraphBuilder *function : functions) {
    std::size_t callers = CountFunctionCalls(function->name(), function->function_domain_);
    for (GraphBuilder *other : functions) {
      if (other != function) {
        callers += other->CountFunctionCalls(function->name(), function->function_domain_);
      }
    }
    if (callers != 0) {
      called_before.insert(function->name());
    }
  }

  const std::size_t inlined = InlineFunctionCalls(functions);

  // Drop the definitions of functions that were called but no longer have any
  // caller. Removing one can drop the last reference to another (a function
  // called only from a now-removed function body), so iterate until stable.
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < local_functions_.size(); ++i) {
      GraphBuilder *function = local_functions_[i].get();
      if (called_before.find(function->name()) == called_before.end()) {
        continue;
      }
      std::size_t callers = CountFunctionCalls(function->name(), function->function_domain_);
      for (const auto &other : local_functions_) {
        if (other.get() != function) {
          callers += other->CountFunctionCalls(function->name(), function->function_domain_);
        }
      }
      if (callers == 0) {
        local_functions_.erase(local_functions_.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        break;
      }
    }
  }
  return inlined;
}

std::size_t GraphBuilder::ConstantFold(const ConstantFoldingOptions &options) {
  if (!options.enabled) {
    return 0;
  }

  // Fold nested builders first so a subgraph or local function shrinks before
  // this scope is folded, mirroring the other recursive passes.
  std::size_t removed = 0;
  for (const auto &function : local_functions_) {
    removed += function->ConstantFold(options);
  }
  for (const auto &subgraph : subgraphs_) {
    removed += subgraph->ConstantFold(options);
  }

  const std::size_t num_nodes = nodes_.size();
  if (num_nodes == 0) {
    return removed;
  }

  // Recompute the constant classification and value tags over the current node
  // list rather than relying on the incremental state maintained by MakeNode,
  // which the other mutating passes (RemoveUnusedNodes, ...) do not keep up to
  // date. Both analyses are aligned with ``nodes_`` because BuildGraph emits
  // the nodes in insertion order.
  const GraphProto graph = BuildGraph();
  const auto [value_tags, node_tags] = compute_.ComputeValueAndNodeTags(graph);
  const auto [constant_values, node_constant] = core::compute::InferConstants(graph);

  // ``(domain, op_type)`` blacklist: an empty component matches everything.
  const auto is_excluded = [&](const std::string &normalised_domain, const std::string &op_type) {
    for (const auto &entry : options.excluded_ops) {
      if ((entry.first.empty() || NormaliseDomain(entry.first) == normalised_domain) &&
          (entry.second.empty() || entry.second == op_type)) {
        return true;
      }
    }
    return false;
  };

  // Constant tensors already materialized: the graph initializers plus every
  // output folded so far in this pass. The initializer pointers stay valid
  // because ``initializers_`` is only appended to at the very end; the folded
  // outputs live in ``folded``, whose element addresses are stable across
  // inserts because ``std::unordered_map`` never invalidates pointers or
  // references to existing elements (only iterators) on rehash.
  std::unordered_map<std::string, const TensorProto *> const_tensors;
  for (const TensorProto &initializer : initializers_) {
    const_tensors.emplace(initializer.name().value(), &initializer);
  }
  std::unordered_map<std::string, TensorProto> folded;
  folded.reserve(num_nodes);
  std::vector<std::string> folded_order;

  const std::string device_suffix = core::symbolic::DeviceKeySuffix(device_);
  const auto &kernel_table = core::runtime::KernelDispatchTable();
  const auto &global_custom_kernels = core::runtime::GlobalCustomKernels();

  utils::RepeatedProtoField<NodeProto> kept;
  kept.reserve(num_nodes);
  std::size_t local_removed = 0;
  for (std::size_t idx = 0; idx < num_nodes; ++idx) {
    NodeProto &node = nodes_[idx];
    const std::string op_type = node.op_type().value();
    const std::string domain = node.domain().empty() ? std::string() : node.domain().value();
    const std::string normalised_domain = NormaliseDomain(domain);

    // A node is a fold candidate when the analysis flagged it constant and it is
    // neither blacklisted nor a control-flow node. A node whose output is a
    // declared graph output is still folded: the result is materialized as an
    // initializer carrying that name, which remains a valid graph output.
    bool candidate = node_constant[idx] == core::compute::ConstantInfo::kConstant &&
                     !is_excluded(normalised_domain, op_type) && !NodeCarriesSubgraph(node);
    // Every non-empty input must already be available as a materialized
    // constant; otherwise the data needed to evaluate the node is missing (for
    // example a predecessor that was left unfolded).
    if (candidate) {
      for (int i = 0; i < node.input().size(); ++i) {
        const std::string input(node.input(static_cast<std::size_t>(i)));
        if (!input.empty() && const_tensors.find(input) == const_tensors.end()) {
          candidate = false;
          break;
        }
      }
    }
    if (!candidate) {
      kept.push_back(std::move(node));
      continue;
    }

    // Classify the node by the value tag of its outputs. A shape-carrying result
    // must be foldable; a weight (or untagged) result is folded best-effort.
    bool is_shape_result = false;
    for (int i = 0; i < node.output().size(); ++i) {
      const std::string output(node.output(static_cast<std::size_t>(i)));
      const auto tag_it = value_tags.find(output);
      if (tag_it != value_tags.end() && tag_it->second == "shape") {
        is_shape_result = true;
        break;
      }
    }

    const std::string dispatch_key = normalised_domain + ":" + op_type;
    const bool have_kernel =
        kernel_table.find(dispatch_key + device_suffix) != kernel_table.end() ||
        global_custom_kernels.find(dispatch_key) != global_custom_kernels.end();
    if (!have_kernel) {
      if (is_shape_result) {
        throw BuilderError("GraphBuilder: constant folding requires a kernel for the shape-tagged "
                           "node '" +
                           op_type + "' in domain '" + normalised_domain +
                           "', but no runtime kernel is registered.");
      }
      if (options.raise_on_missing_weight_kernel) {
        throw BuilderError("GraphBuilder: constant folding requires a kernel for the node '" +
                           op_type + "' in domain '" + normalised_domain +
                           "', but no runtime kernel is registered.");
      }
      kept.push_back(std::move(node));
      continue;
    }

    // Evaluate the node in isolation: seed its inputs from the materialized
    // constants and run the registered kernel.
    const int opset = OpsetVersion(domain);
    core::runtime::RuntimeContext rt(core::runtime::KernelContext(core::runtime::DefaultOpset(
                                         opset > 0 ? static_cast<int64_t>(opset) : 0)),
                                     core::runtime::RuntimeContextOptions{.device = device_});
    for (int i = 0; i < node.input().size(); ++i) {
      const std::string input(node.input(static_cast<std::size_t>(i)));
      if (input.empty()) {
        continue;
      }
      rt.tensors()[input] = core::runtime::TensorFromProto(*const_tensors.at(input));
    }
    core::runtime::RunNode(node, rt);

    // Reject the fold when any output exceeds the size threshold; the node is
    // then kept untouched.
    bool within_threshold = true;
    if (options.max_element_count >= 0) {
      for (int i = 0; i < node.output().size(); ++i) {
        const std::string output(node.output(static_cast<std::size_t>(i)));
        if (output.empty()) {
          continue;
        }
        if (rt.tensors().at(output).element_count() > options.max_element_count) {
          within_threshold = false;
          break;
        }
      }
    }
    if (!within_threshold) {
      kept.push_back(std::move(node));
      continue;
    }

    // Materialize every output as a constant available to later folds.
    for (int i = 0; i < node.output().size(); ++i) {
      const std::string output(node.output(static_cast<std::size_t>(i)));
      if (output.empty()) {
        continue;
      }
      TensorProto proto = ProtoFromRuntimeTensor(rt.tensors().at(output), output);
      const auto inserted = folded.emplace(output, std::move(proto));
      const_tensors[output] = &inserted.first->second;
      folded_order.push_back(output);
    }
    ++local_removed;
  }

  nodes_ = std::move(kept);
  // Append the folded results as initializers, in fold order.
  for (const std::string &name : folded_order) {
    initializers_.push_back(std::move(folded.at(name)));
  }
  return removed + local_removed;
}

std::size_t GraphBuilder::RemoveDuplicateInitializers() {
  return DeduplicateInitializers(InitializerContentIndex{});
}

std::size_t GraphBuilder::DeduplicateInitializers(const InitializerContentIndex &inherited) {
  // Declared graph outputs must keep their own name; they are never dropped as
  // duplicates (but can still act as the survivor for a later duplicate).
  std::unordered_set<std::string> output_names;
  for (const ValueInfoProto &output : outputs_) {
    output_names.insert(output.name().value());
  }

  // Start from the initializers visible in the enclosing scope and add this
  // scope's survivors as we go. The first initializer with a given content
  // wins; every later duplicate is dropped and its references are rewritten to
  // the surviving name.
  InitializerContentIndex index = inherited;
  std::unordered_map<std::string, std::string> rename;
  utils::RepeatedProtoField<TensorProto> kept;
  kept.reserve(initializers_.size());
  std::size_t removed = 0;
  for (TensorProto &initializer : initializers_) {
    const int64_t hash = initializer.ContentHash(/*include_content=*/false);
    if (output_names.find(initializer.name().value()) == output_names.end()) {
      const TensorProto *survivor = nullptr;
      auto candidates = index.find(hash);
      if (candidates != index.end()) {
        for (const TensorProto *candidate : candidates->second) {
          if (SameInitializerContent(*candidate, initializer)) {
            survivor = candidate;
            break;
          }
        }
      }
      if (survivor != nullptr) {
        rename.emplace(initializer.name().value(), survivor->name().value());
        ++removed;
        continue;
      }
    }
    kept.push_back(std::move(initializer));
    // Register the survivor right away so later initializers in this same scope
    // (and, once the loop is done, subgraphs) can collapse onto it. Elements are
    // heap-owned by the field, so the pointer stays valid as ``kept`` grows.
    index[hash].push_back(&kept[kept.size() - 1]);
  }
  initializers_ = std::move(kept);

  RewriteInitializerReferences(rename);

  // Subgraph bodies see the enclosing scope, so pass the augmented index down.
  // Local functions have an isolated scope and start from a fresh index.
  for (const auto &subgraph : subgraphs_) {
    removed += subgraph->DeduplicateInitializers(index);
  }
  for (const auto &function : local_functions_) {
    removed += function->DeduplicateInitializers(InitializerContentIndex{});
  }
  return removed;
}

void GraphBuilder::RewriteInitializerReferences(
    const std::unordered_map<std::string, std::string> &rename) {
  if (rename.empty()) {
    return;
  }
  for (NodeProto &node : nodes_) {
    for (int i = 0; i < node.input().size(); ++i) {
      auto it = rename.find(node.input(static_cast<std::size_t>(i)));
      if (it != rename.end()) {
        node.mutable_input(static_cast<std::size_t>(i))->assign(it->second);
      }
    }
  }
  // Subgraph bodies capture values from the enclosing scope, so a duplicate
  // initializer they reference must be rewritten here too. Local functions have
  // an isolated scope and cannot see this builder's initializers.
  for (const auto &subgraph : subgraphs_) {
    subgraph->RewriteInitializerReferences(rename);
  }
}

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
  // Records constant-value / constant-node information.
  core::compute::WriteConstantInfoToMetadata(graph);
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

} // namespace ONNX_LIGHT_NAMESPACE::core::builder

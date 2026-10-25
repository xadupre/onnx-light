// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

#include "onnx_core/compute/constant_info.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_proto/onnx_alias.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_tree_ensemble.h"
#include "onnx_proto/onnx_verify.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

BuilderError::~BuilderError() = default;

using ::onnx_light::core::shapes::CompatibleEncodedDefault;
using ::onnx_light::core::shapes::kUnknownOpsetVersion;
using ::onnx_light::core::shapes::SameDeclaredType;
using ::onnx_light::core::shapes::ShapesContext;
using ::onnx_light::core::symbolic::SymTensorFromTensorProto;
using ::onnx_light::core::symbolic::SymTensorFromValueInfo;
using ::onnx_light::core::symbolic::SymTensorToValueInfo;
using ::onnx_light::core::symbolic::TensorTypeToDataType;

namespace {

template <typename Proto> std::string DeclaredValueTag(const Proto &value) {
  for (const auto &entry : value.metadata_props()) {
    if (entry.key() == core::compute::kValueTagMetadataKey) {
      return entry.value();
    }
  }
  return {};
}

bool HasUnboundAttributes(const NodeProto &node) {
  for (const auto &attribute : node.attribute()) {
    if (!attribute.ref_attr_name().empty() ||
        (attribute.type() == AttributeProto::AttributeType::GRAPH && !attribute.has_g()) ||
        (attribute.type() == AttributeProto::AttributeType::GRAPHS && attribute.graphs().empty())) {
      return true;
    }
    if (attribute.has_g()) {
      for (const auto &child : attribute.g().node()) {
        if (HasUnboundAttributes(child)) {
          return true;
        }
      }
    }
    for (const auto &graph : attribute.graphs()) {
      for (const auto &child : graph.node()) {
        if (HasUnboundAttributes(child)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool ShapeInputsAvailable(const ShapesContext &shapes, const NodeProto &node) {
  for (const std::string &input : core::graph::CollectNodeInputs(node)) {
    if (input.empty() || shapes.Has(input) || shapes.HasSequence(input)) {
      continue;
    }
    if (!shapes.HasType(input) ||
        (shapes.GetType(input).has_tensor_type() &&
         !core::shapes::AcceptsTypeOnlyTensorShapeInputs(node.domain(), node.op_type()))) {
      return false;
    }
  }
  return true;
}

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

bool SameEncodedContent(const EncodedValueProto &lhs, const EncodedValueProto &rhs) {
  // The complete layout (including constants, codecs and nominal type IDs)
  // participates in identity, not merely its byte extent or payload.
  EncodedValueProto left = utils::CopyWithSharedTensorStorage(lhs);
  EncodedValueProto right = utils::CopyWithSharedTensorStorage(rhs);
  left.clear_name();
  right.clear_name();
  return left.Equals(right);
}

bool HasStructuredType(const TypeProto &type) {
  if (type.has_struct_type()) {
    return true;
  }
  if (type.has_sequence_type()) {
    return HasStructuredType(type.sequence_type().elem_type());
  }
  if (type.has_optional_type()) {
    return HasStructuredType(type.optional_type().elem_type());
  }
  return type.has_map_type() && HasStructuredType(type.map_type().value_type());
}

using DeclaredTypes = std::unordered_map<std::string, const TypeProto *>;

struct AttributeBinding {
  const AttributeProto *attribute;
  bool caller_scoped;
};

using AttributeBindings = std::unordered_map<std::string, AttributeBinding>;

DeclaredTypes StructuredDeclarations(const utils::RepeatedProtoField<ValueInfoProto> &values) {
  DeclaredTypes declarations;
  for (const auto &value : values) {
    if (value.has_type() && !value.type().has_tensor_type()) {
      declarations.emplace(value.name().value(), &value.type());
    }
  }
  return declarations;
}

void SeedDeclaredOutputs(ShapesContext &shapes, const NodeProto &node,
                         const DeclaredTypes &declarations) {
  for (const auto &name : node.output()) {
    const auto declared = declarations.find(name);
    if (declared == declarations.end()) {
      continue;
    }
    if (shapes.HasType(name)) {
      std::string difference;
      if (!SameDeclaredType(shapes.GetType(name), *declared->second, &difference)) {
        throw BuilderError("GraphBuilder: incompatible declared structured type for '" +
                           std::string(name) + "': " + difference);
      }
    } else if (!shapes.Has(name) && !shapes.HasSequence(name)) {
      shapes.SetType(name, *declared->second);
    }
  }
}

SymTensor RefineTensorDtype(const SymTensor &inferred, TensorType dtype) {
  SymTensor refined(inferred.Data(), dtype, inferred.Shape());
  refined.SetDevice(inferred.GetDevice());
  if (inferred.HasMin()) {
    refined.SetMin(inferred.Min());
  }
  if (inferred.HasMax()) {
    refined.SetMax(inferred.Max());
  }
  if (inferred.HasValueAsShape()) {
    refined.SetValueAsShape(inferred.ValueAsShape());
  }
  return refined;
}

void SeedDeclaredValueInfo(ShapesContext &shapes, const ValueInfoProto &value) {
  if (!value.has_type()) {
    return;
  }
  const std::string name = value.name().value();
  if (value.type().has_tensor_type() && shapes.Has(name)) {
    const SymTensor &inferred = shapes.Get(name);
    const auto dtype = value.type().tensor_type().elem_type();
    if (inferred.Dtype() == TensorType::kUndefined && dtype != TensorProto::UNDEFINED) {
      const auto refined_dtype = core::symbolic::DataTypeToTensorType(dtype);
      if (refined_dtype == TensorType::kUndefined) {
        throw BuilderError("GraphBuilder: output '" + name +
                           "' has an unsupported tensor element type.");
      }
      shapes.Set(name, RefineTensorDtype(inferred, refined_dtype));
    }
    return;
  }
  SymTensor descriptor;
  if (value.type().has_tensor_type() && value.type().tensor_type().has_shape() &&
      !shapes.Has(name) && SymTensorFromValueInfo(value, descriptor)) {
    shapes.Set(name, std::move(descriptor));
  } else if (!value.type().has_tensor_type() && !shapes.HasType(name) && !shapes.Has(name)) {
    shapes.SetType(name, value.type());
  }
}

std::unordered_set<std::string> GraphDefinedNames(const GraphProto &graph) {
  std::unordered_set<std::string> defined;
  for (const auto &input : graph.input()) {
    defined.insert(input.name().value());
  }
  for (const auto &initializer : graph.initializer()) {
    defined.insert(initializer.name().value());
  }
  for (const auto &initializer : graph.sparse_initializer()) {
    defined.insert(initializer.values().name().value());
  }
  for (const auto &initializer : graph.extended_initializer()) {
    defined.insert(initializer.name().value());
  }
  for (const auto &node : graph.node()) {
    for (const auto &output : node.output()) {
      if (!output.empty()) {
        defined.insert(output);
      }
    }
  }
  return defined;
}

void RewriteGraphLocalCaptures(GraphProto &graph,
                               const std::unordered_map<std::string, std::string> &rename) {
  const auto defined = GraphDefinedNames(graph);
  const auto rewrite = [&](std::string &name) {
    if (defined.count(name) == 0) {
      const auto found = rename.find(name);
      if (found != rename.end()) {
        name = found->second;
      }
    }
  };
  for (auto &node : graph.ref_node()) {
    for (auto &input : node.ref_input()) {
      rewrite(input);
    }
  }
  for (auto &output : graph.ref_output()) {
    std::string name = output.name().value();
    rewrite(name);
    output.set_name(name);
  }
  for (auto &value : graph.ref_value_info()) {
    std::string name = value.name().value();
    rewrite(name);
    value.set_name(name);
  }
  for (auto &annotation : graph.ref_quantization_annotation()) {
    std::string name = annotation.tensor_name().value();
    rewrite(name);
    annotation.set_tensor_name(name);
    for (auto &parameter : annotation.ref_quant_parameter_tensor_names()) {
      name = parameter.value().value();
      rewrite(name);
      parameter.set_value(name);
    }
  }
}

using ValueRename = std::unordered_map<std::string, std::string>;

void ResolveFunctionAttributes(GraphProto &graph, const AttributeBindings &bindings,
                               const ValueRename *rename, bool function_owned);

void ResolveFunctionAttributes(NodeProto &node, const AttributeBindings &bindings,
                               const ValueRename *rename, bool enclosing_function_owned) {
  utils::RepeatedProtoField<AttributeProto> resolved;
  resolved.reserve(node.attribute().size());
  for (const auto &attribute : node.attribute()) {
    AttributeProto clone;
    bool function_owned = enclosing_function_owned;
    if (!attribute.ref_attr_name().empty()) {
      const auto found = bindings.find(attribute.ref_attr_name().value());
      if (found == bindings.end()) {
        throw BuilderError("GraphBuilder: cannot inline a function with an unresolved attribute '" +
                           attribute.ref_attr_name().value() + "'.");
      }
      clone = utils::CopyWithSharedTensorStorage(*found->second.attribute);
      function_owned = !found->second.caller_scoped;
      clone.set_name(attribute.name().value());
      clone.clear_ref_attr_name();
    } else {
      clone = utils::CopyWithSharedTensorStorage(attribute);
    }
    if (clone.has_g()) {
      ResolveFunctionAttributes(*clone.mutable_g(), bindings, rename, function_owned);
    }
    for (auto &graph : clone.ref_graphs()) {
      ResolveFunctionAttributes(graph, bindings, rename, function_owned);
    }
    resolved.push_back(std::move(clone));
  }
  node.ref_attribute() = std::move(resolved);
}

void ResolveFunctionAttributes(GraphProto &graph, const AttributeBindings &bindings,
                               const ValueRename *rename, bool function_owned) {
  ValueRename nested_rename;
  const ValueRename *nested = rename;
  if (rename != nullptr) {
    nested_rename = *rename;
    for (const auto &name : GraphDefinedNames(graph)) {
      nested_rename.erase(name);
    }
    nested = &nested_rename;
    if (function_owned) {
      RewriteGraphLocalCaptures(graph, *rename);
    }
  }
  for (auto &node : graph.ref_node()) {
    ResolveFunctionAttributes(node, bindings, nested, function_owned);
  }
}

void RequireStandardGraph(const GraphProto &graph);

void RequireStandardAttribute(const AttributeProto &attribute) {
  if (attribute.has_tp() && HasStructuredType(attribute.tp())) {
    throw BuilderError("GraphBuilder: standard ONNX cannot represent structured type attributes.");
  }
  for (const auto &type : attribute.type_protos()) {
    if (HasStructuredType(type)) {
      throw BuilderError(
          "GraphBuilder: standard ONNX cannot represent structured type attributes.");
    }
  }
  if (attribute.has_g()) {
    RequireStandardGraph(attribute.g());
  }
  for (const auto &graph : attribute.graphs()) {
    RequireStandardGraph(graph);
  }
}

template <typename Proto> void RequireStandardBody(const Proto &body) {
  for (const auto &value : body.value_info()) {
    if (value.has_type() && HasStructuredType(value.type())) {
      throw BuilderError("GraphBuilder: standard ONNX cannot represent structured value types.");
    }
  }
  for (const auto &node : body.node()) {
    for (const auto &attribute : node.attribute()) {
      RequireStandardAttribute(attribute);
    }
  }
}

void RequireStandardGraph(const GraphProto &graph) {
  if (!graph.persistent_bindings().empty()) {
    throw BuilderError("GraphBuilder: standard ONNX cannot represent persistent bindings.");
  }
  if (!graph.extended_initializer().empty()) {
    throw BuilderError("GraphBuilder: standard ONNX cannot represent extended initializers.");
  }
  for (const auto &value : graph.input()) {
    if (value.has_type() && HasStructuredType(value.type())) {
      throw BuilderError("GraphBuilder: standard ONNX cannot represent structured inputs.");
    }
  }
  for (const auto &value : graph.output()) {
    if (value.has_type() && HasStructuredType(value.type())) {
      throw BuilderError("GraphBuilder: standard ONNX cannot represent structured outputs.");
    }
  }
  RequireStandardBody(graph);
}

} // namespace

// Materializes an initializer ``TensorProto`` (owning its payload) from a
// runtime tensor produced by constant folding. Non-STRING tensors copy their
// little-endian byte buffer into ``raw_data``; STRING tensors copy their
// elements into ``string_data``.
TensorProto GraphBuilder::ProtoFromRuntimeTensor(const core::runtime::Tensor &tensor,
                                                 const std::string &name) {
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

GraphBuilder::GraphBuilder(std::string name, SchemaLookupFn schema_lookup,
                           const std::unordered_map<std::string, int> &opsets)
    : name_(std::move(name)), schema_lookup_(std::move(schema_lookup)) {
  for (const auto &[domain, version] : opsets) {
    SetOpsetVersion(domain, version);
  }
}

GraphBuilder::GraphBuilder(const ModelProto &model, SchemaLookupFn schema_lookup)
    : name_(model.graph().name().empty() ? std::string("graph") : model.graph().name().value()),
      schema_lookup_(std::move(schema_lookup)) {
  // Copy only metadata: copying the optional graph would serialize its payload
  // before discarding it. Graphs, functions, types and opsets are imported below.
  model_template_.ir_version_ = model.ir_version_;
  model_template_.producer_name_ = model.producer_name_;
  model_template_.producer_version_ = model.producer_version_;
  model_template_.domain_ = model.domain_;
  model_template_.model_version_ = model.model_version_;
  model_template_.doc_string_ = model.doc_string_;
  model_template_.metadata_props_ = model.metadata_props_;
  model_template_.configuration_ = model.configuration_;
  compute_.Shapes().SetStructTypes(model.struct_types());
  for (const auto &opset : model.opset_import()) {
    SetOpsetVersion(opset.domain().empty() ? std::string() : opset.domain().value(),
                    static_cast<int>(opset.version()));
  }
  std::set<FunctionIdentifier> model_functions;
  for (const auto &function : model.functions()) {
    if (function.name().empty() ||
        !model_functions
             .insert(
                 MakeFunctionIdentifier(function.domain(), function.name(), function.overload()))
             .second)
      throw BuilderError("GraphBuilder: empty or duplicate model-local function identity.");
  }
  std::set<FunctionIdentifier> imported_functions;
  const auto check_function_calls = [&](const auto &self, const NodeProto &node) -> void {
    const auto identity = MakeFunctionIdentifier(node.domain(), node.op_type(), node.overload());
    if (model_functions.count(identity) != 0 && imported_functions.count(identity) == 0) {
      throw BuilderError("GraphBuilder: model-local function '" + node.domain().value() +
                         "::" + node.op_type().value() + ":" + node.overload().value() +
                         "' is not yet imported; definitions must place callees before callers.");
    }
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g()) {
        for (const auto &child : attribute.g().node()) {
          self(self, child);
        }
      }
      for (const auto &graph : attribute.graphs()) {
        for (const auto &child : graph.node()) {
          self(self, child);
        }
      }
    }
  };
  for (const auto &function : model.functions()) {
    for (const auto &node : function.node()) {
      check_function_calls(check_function_calls, node);
    }
    for (const auto &attribute : function.attribute_proto()) {
      if (attribute.has_g()) {
        for (const auto &node : attribute.g().node()) {
          check_function_calls(check_function_calls, node);
        }
      }
      for (const auto &graph : attribute.graphs()) {
        for (const auto &node : graph.node()) {
          check_function_calls(check_function_calls, node);
        }
      }
    }
    const std::string function_name = function.name().value();
    const std::string function_domain = function.domain().value();
    // ONNX functions have an independent namespace and are identified by all
    // three fields, not by graph value names or the bare function name.
    auto child = std::make_unique<GraphBuilder>(function_name, schema_lookup_, opsets_);
    child->is_local_function_ = true;
    child->function_domain_ = function_domain;
    child->compute_.Shapes().SetStructTypes(Shapes().StructTypes());
    child->parent_ = this;
    GraphBuilder &local = *child;
    local_functions_.push_back(std::move(child));
    local.ImportFunction(function);
    imported_functions.insert(
        MakeFunctionIdentifier(function_domain, function_name, function.overload()));
  }
  ImportGraph(model.graph());
  rewritings_ = utils::CopyWithSharedTensorStorage(model.rewritings());
}

GraphBuilder::~GraphBuilder() = default;
GraphBuilder::GraphBuilder(GraphBuilder &&other) noexcept : GraphBuilder(std::string()) {
  *this = std::move(other);
}

GraphBuilder &GraphBuilder::operator=(GraphBuilder &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  name_ = std::move(other.name_);
  rewriting_undo_ = std::move(other.rewriting_undo_);
  is_local_function_ = other.is_local_function_;
  function_domain_ = std::move(other.function_domain_);
  function_attributes_ = std::move(other.function_attributes_);
  function_attribute_protos_ = std::move(other.function_attribute_protos_);
  metadata_ = std::move(other.metadata_);
  doc_string_ = std::move(other.doc_string_);
  schema_lookup_ = std::move(other.schema_lookup_);
  schema_table_ = std::move(other.schema_table_);
  compute_ = std::move(other.compute_);
  inputs_ = std::move(other.inputs_);
  outputs_ = std::move(other.outputs_);
  value_infos_ = std::move(other.value_infos_);
  nodes_ = std::move(other.nodes_);
  initializers_ = std::move(other.initializers_);
  extended_initializers_ = std::move(other.extended_initializers_);
  model_template_ = std::move(other.model_template_);
  rewritings_ = std::move(other.rewritings_);
  graph_template_ = std::move(other.graph_template_);
  function_template_ = std::move(other.function_template_);
  local_functions_ = std::move(other.local_functions_);
  subgraphs_ = std::move(other.subgraphs_);
  names_ = std::move(other.names_);
  inherited_names_ = std::move(other.inherited_names_);
  opsets_ = std::move(other.opsets_);
  device_ = other.device_;
  auto_counter_ = other.auto_counter_;
  for (const auto &child : local_functions_) {
    child->parent_ = this;
  }
  for (const auto &child : subgraphs_) {
    child->parent_ = this;
  }
  return *this;
}

// ── Opset management ───────────────────────────────────────────────────

void GraphBuilder::SetOpsetVersion(const std::string &domain, int version) {
  if (version <= 0) {
    throw BuilderError("GraphBuilder: opset versions must be positive.");
  }
  const std::string key = NormaliseDomain(domain);
  GraphBuilder *owner = this;
  while (owner->parent_ != nullptr && !owner->is_local_function_)
    owner = owner->parent_;
  if (owner != this && owner->OpsetVersion(key) == kUnknownOpsetVersion) {
    owner->SetOpsetVersion(key, version);
    if (OpsetVersion(key) == version)
      return;
  }
  const bool changed = OpsetVersion(key) != version;
  if (changed)
    InvalidateRewritings();
  if (RefreshOpsetVersion(key, version) || (changed && is_local_function_)) {
    RefreshAncestors();
  }
}

bool GraphBuilder::RefreshOpsetVersion(const std::string &domain, int version,
                                       const std::unordered_set<std::string> &inherited_values) {
  const auto existing = opsets_.find(domain);
  const bool changed = existing == opsets_.end() || existing->second != version;
  opsets_[domain] = version;
  auto &shapes = compute_.Shapes();
  shapes.SetOpsetVersion(domain, version);
  std::unordered_set<std::string> affected_values;
  for (const auto &name : inherited_values) {
    if (inherited_names_.count(name) != 0) {
      shapes.Erase(name);
      SeedInheritedValue(name);
      affected_values.insert(name);
    }
  }
  // Refresh children at their referencing nodes, after any captured producers.
  std::unordered_map<GraphBuilder *, bool> refreshed;
  const bool affected = RefreshDependentNodes(
      [&](const NodeProto &node, const std::unordered_set<std::string> &values) {
        bool nested_changed = false;
        for (GraphBuilder *subgraph : ReferencedSubgraphs(node)) {
          auto it = refreshed.find(subgraph);
          if (it == refreshed.end()) {
            it = refreshed.emplace(subgraph, subgraph->RefreshOpsetVersion(domain, version, values))
                     .first;
          }
          nested_changed |= it->second;
        }
        return nested_changed || (changed && NormaliseDomain(node.domain().value()) == domain);
      },
      affected_values, /*propagate_to_parent=*/false);
  for (const auto &subgraph : subgraphs_) {
    if (refreshed.count(subgraph.get()) == 0) {
      subgraph->RefreshOpsetVersion(domain, version, affected_values);
    }
  }
  return affected;
}

int GraphBuilder::OpsetVersion(const std::string &domain) const {
  auto it = opsets_.find(NormaliseDomain(domain));
  return it == opsets_.end() ? kUnknownOpsetVersion : it->second;
}

// ── Name management ────────────────────────────────────────────────────

bool GraphBuilder::HasName(const std::string &name) const noexcept {
  return names_.find(name) != names_.end() || inherited_names_.find(name) != inherited_names_.end();
}

const std::string &GraphBuilder::ReserveName(const std::string &name) {
  if (name.empty()) {
    throw BuilderError("GraphBuilder: cannot reserve an empty name.");
  }
  if (inherited_names_.find(name) != inherited_names_.end()) {
    throw BuilderError("GraphBuilder: the name '" + name +
                       "' is already visible in an ancestor scope; SSA shadowing is not allowed.");
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

void GraphBuilder::SeedInheritedValue(const std::string &name) {
  auto &shapes = compute_.Shapes();
  if (name.empty() || shapes.Has(name) || shapes.HasSequence(name) || shapes.HasType(name) ||
      parent_ == nullptr || inherited_names_.count(name) == 0) {
    return;
  }
  parent_->SeedInheritedValue(name);
  const auto &outer = parent_->Shapes();
  if (outer.Has(name) || outer.HasSequence(name) || outer.HasType(name) ||
      outer.HasEncodedValue(name)) {
    shapes.CopyValueFrom(name, outer, name);
  }
}

void GraphBuilder::CopyInferredValuesFrom(const ShapesContext &source) {
  std::unordered_set<std::string> names;
  for (const auto &input : inputs_) {
    names.insert(input.name().value());
  }
  for (const auto &initializer : initializers_) {
    names.insert(initializer.name().value());
  }
  for (const auto &initializer : extended_initializers_) {
    names.insert(initializer.name().value());
  }
  for (const auto &node : nodes_) {
    names.insert(node.output().begin(), node.output().end());
  }
  for (const auto &output : outputs_) {
    names.insert(output.name().value());
  }
  for (const auto &name : names) {
    if (!name.empty() && (source.Has(name) || source.HasSequence(name) || source.HasType(name) ||
                          source.HasEncodedValue(name))) {
      compute_.Shapes().CopyValueFrom(name, source, name);
    }
  }
}

void GraphBuilder::RefreshInferredInputsFrom(
    const ShapesContext &source, const std::unordered_set<std::string> &changed_captures) {
  std::unordered_set<std::string> candidates;
  for (const auto &input : inputs_) {
    candidates.insert(input.name().value());
  }
  for (const auto &name : changed_captures) {
    if (inherited_names_.count(name) != 0) {
      candidates.insert(name);
    }
  }
  auto &shapes = compute_.Shapes();
  std::unordered_set<std::string> changed_values;
  for (const auto &name : candidates) {
    if (!source.Has(name) && !source.HasType(name) && !source.HasSequence(name) &&
        !source.HasEncodedValue(name)) {
      continue;
    }
    const bool unchanged =
        shapes.Has(name) == source.Has(name) &&
        (!source.Has(name) || shapes.Get(name) == source.Get(name)) &&
        shapes.HasType(name) == source.HasType(name) &&
        (!source.HasType(name) || shapes.GetType(name).Equals(source.GetType(name))) &&
        shapes.HasSequence(name) == source.HasSequence(name) &&
        (!source.HasSequence(name) || shapes.GetSequence(name) == source.GetSequence(name)) &&
        shapes.HasEncodedValue(name) == source.HasEncodedValue(name) &&
        (!source.HasEncodedValue(name) ||
         shapes.GetEncodedValue(name).Equals(source.GetEncodedValue(name)));
    if (!unchanged) {
      shapes.Erase(name);
      shapes.CopyValueFrom(name, source, name);
      changed_values.insert(name);
    }
  }
  if (!changed_values.empty()) {
    RefreshDependentNodes(
        [](const NodeProto &, const std::unordered_set<std::string> &) { return false; },
        changed_values, /*propagate_to_parent=*/false);
  }
}

void GraphBuilder::set_device(Device device) {
  if (device_ == device) {
    return;
  }
  device_ = device;
  compute_.ComputePeakMemory(BuildGraph(), device_);
}

void GraphBuilder::CopyDeviceTargetsFrom(const GraphBuilder &source) {
  std::map<FunctionIdentifier, const GraphBuilder *> functions;
  const auto collect_functions = [&](const auto &self, const GraphBuilder &scope) -> void {
    for (const auto &function : scope.local_functions_) {
      const auto identity = MakeFunctionIdentifier(function->function_domain_, function->name_,
                                                   function->function_template_.overload());
      if (!functions.emplace(identity, function.get()).second)
        throw BuilderError("CopyDeviceTargetsFrom: duplicate source function identity.");
      self(self, *function);
    }
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph);
  };
  collect_functions(collect_functions, source);
  std::vector<std::pair<GraphBuilder *, Device>> updates;
  const auto collect = [&](const auto &self, GraphBuilder &target,
                           const GraphBuilder &original) -> void {
    // Exported functions materialize their initializers before the original nodes.
    const size_t offset =
        target.nodes_.size() != original.nodes_.size() && original.is_local_function_
            ? original.initializers_.size()
            : 0;
    if (target.nodes_.size() != original.nodes_.size() + offset)
      throw BuilderError("CopyDeviceTargetsFrom: incompatible node counts.");
    for (size_t index = 0; index < original.nodes_.size(); ++index) {
      const auto originals = original.ReferencedSubgraphs(original.nodes_[index]);
      const auto children = target.ReferencedSubgraphs(target.nodes_[index + offset]);
      if (originals.size() != children.size())
        throw BuilderError("CopyDeviceTargetsFrom: incompatible subgraph references.");
      for (size_t child = 0; child < originals.size(); ++child)
        self(self, *children[child], *originals[child]);
    }
    for (const auto &function : target.local_functions_) {
      const auto identity = MakeFunctionIdentifier(function->function_domain_, function->name_,
                                                   function->function_template_.overload());
      const auto found = functions.find(identity);
      if (found == functions.end())
        throw BuilderError("CopyDeviceTargetsFrom: incompatible exported functions.");
      self(self, *function, *found->second);
    }
    updates.emplace_back(&target, original.device_);
  };
  collect(collect, *this, source);
  for (const auto &[scope, device] : updates)
    scope->set_device(device);
}

void GraphBuilder::SetStructTypes(const utils::RepeatedProtoField<StructTypeProto> &types) {
  InvalidateRewritings();
  compute_.Shapes().SetStructTypes(types);
  for (const auto &child : subgraphs_) {
    child->SetStructTypes(types);
  }
  for (const auto &child : local_functions_) {
    child->SetStructTypes(types);
  }
}

void GraphBuilder::MakeStructType(const StructTypeProto &type) {
  if (parent_ != nullptr) {
    parent_->MakeStructType(type);
    return;
  }
  auto declarations = Shapes().StructTypes();
  declarations.push_back(type);
  SetStructTypes(declarations);
  RebuildStructuredState();
}

void GraphBuilder::InvalidateRewritings() {
  for (GraphBuilder *builder = this; builder != nullptr; builder = builder->parent_) {
    builder->rewritings_.clear();
    builder->rewriting_undo_.reset();
  }
}

void GraphBuilder::RefreshAncestors() {
  InvalidateRewritings();
  if (ancestor_refresh_suppressed_ || outputs_.empty() || parent_ == nullptr) {
    return;
  }
  if (is_local_function_) {
    parent_->RefreshLocalFunctionDefinition(*this);
  }
  parent_->RefreshNestedDependency(*this);
}

void GraphBuilder::RefreshLocalFunctionDefinition(GraphBuilder &function) {
  GraphBuilder *root = this;
  while (root->parent_ != nullptr) {
    root = root->parent_;
  }
  const auto definition =
      std::make_shared<FunctionProto>(function.BuildFunction(function.function_domain_));
  std::function<void(GraphBuilder &)> install = [&](GraphBuilder &builder) {
    builder.compute_.Shapes().SetLocalFunction(definition);
    for (const auto &subgraph : builder.subgraphs_) {
      install(*subgraph);
    }
    for (const auto &local_function : builder.local_functions_) {
      install(*local_function);
    }
  };
  install(*root);
}

void GraphBuilder::RefreshNestedDependency(GraphBuilder &child, bool propagate_to_parent) {
  std::function<bool(const GraphBuilder &)> depends_on_function = [&](const GraphBuilder &builder) {
    for (const NodeProto &node : builder.nodes_) {
      if (MakeFunctionIdentifier(node.domain(), node.op_type(), node.overload()) ==
          MakeFunctionIdentifier(child.function_domain_, child.name_,
                                 child.function_template_.overload())) {
        return true;
      }
      for (const GraphBuilder *subgraph : builder.ReferencedSubgraphs(node)) {
        if (depends_on_function(*subgraph)) {
          return true;
        }
      }
    }
    return false;
  };
  if (child.is_local_function_) {
    for (const auto &subgraph : subgraphs_) {
      if (subgraph.get() != &child && depends_on_function(*subgraph)) {
        subgraph->RefreshNestedDependency(child, false);
      }
    }
    for (const auto &function : local_functions_) {
      if (function.get() != &child && depends_on_function(*function)) {
        function->RefreshNestedDependency(child);
      }
    }
  }
  std::unordered_set<std::string> affected_values;
  RefreshDependentNodes(
      [&](const NodeProto &node, const std::unordered_set<std::string> &) {
        if (child.is_local_function_ &&
            MakeFunctionIdentifier(node.domain(), node.op_type(), node.overload()) ==
                MakeFunctionIdentifier(child.function_domain_, child.name_,
                                       child.function_template_.overload())) {
          return true;
        }
        for (GraphBuilder *subgraph : ReferencedSubgraphs(node)) {
          if (subgraph == &child || (child.is_local_function_ && depends_on_function(*subgraph))) {
            return true;
          }
        }
        return false;
      },
      affected_values, propagate_to_parent);
}

bool GraphBuilder::RefreshDependentNodes(
    const std::function<bool(const NodeProto &, const std::unordered_set<std::string> &)>
        &is_affected,
    std::unordered_set<std::string> &affected_values, bool propagate_to_parent) {
  bool affected = false;
  std::map<std::size_t, std::vector<std::string>> refreshed_references;
  const auto declarations = StructuredDeclarations(value_infos_);
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    const NodeProto &stored = nodes_[index];
    const bool directly_affected = is_affected(stored, affected_values);
    if (!directly_affected && affected_values.empty()) {
      continue;
    }
    NodeProto materialized = utils::CopyWithSharedTensorStorage(stored);
    MaterializeGraphReferences(materialized, /*share_initializers=*/true);
    std::vector<std::string> references;
    CollectNodeReferences(materialized, references);
    const bool consumes_affected =
        std::any_of(references.begin(), references.end(),
                    [&](const std::string &name) { return affected_values.count(name) != 0; });
    if (!directly_affected && !consumes_affected) {
      continue;
    }
    affected = true;
    compute_.Shapes().EraseNodeConstraints(stored);
    for (const std::string &output : stored.output()) {
      if (!output.empty()) {
        affected_values.insert(output);
        compute_.Shapes().Erase(output);
      }
    }

    const bool ready = ShapeFunctionAvailable(materialized) &&
                       !HasUnboundAttributes(materialized) &&
                       ShapeInputsAvailable(compute_.Shapes(), materialized);
    if (ready) {
      compute_.Shapes().set_current_node_index(static_cast<int64_t>(index));
      compute_.Shapes().ComputeShapeNode(materialized);
      for (const auto &attribute : stored.attribute()) {
        if (HasGraphReferenceSuffix(attribute.name().value()) &&
            attribute.type() == AttributeProto::AttributeType::STRING && attribute.has_s()) {
          const std::string attribute_name =
              attribute.name().value().substr(0, attribute.name().value().size() - 4);
          if (compute_.Shapes().HasSubgraphContext(static_cast<int64_t>(index), attribute_name)) {
            NamedBuilderOrThrow(subgraphs_, attribute.s().value(), "subgraph")
                .RefreshInferredInputsFrom(compute_.Shapes().GetSubgraphContext(
                                               static_cast<int64_t>(index), attribute_name),
                                           affected_values);
          }
        }
      }
    }
    SeedDeclaredOutputs(compute_.Shapes(), stored, declarations);
    for (const auto &output : outputs_) {
      if (affected_values.count(output.name().value()) != 0) {
        SeedDeclaredValueInfo(compute_.Shapes(), output);
      }
    }
    refreshed_references.emplace(index, std::move(references));
    compute_.AppendNodeConstant(stored, index);
  }
  for (auto &output : outputs_) {
    const std::string name = output.name().value();
    if (affected_values.count(name) != 0 && output.has_type() && output.type().has_tensor_type() &&
        Shapes().Has(name)) {
      ValueInfoProto inferred;
      if (SymTensorToValueInfo(Shapes().Get(name), inferred)) {
        *output.mutable_type() = inferred.type();
      }
    }
  }
  compute_.RefreshNodes(nodes_, refreshed_references, compute_.Shapes(), affected_values, device_,
                        [this](const NodeProto &node) { return LayerBodyForMemory(node); });
  if (propagate_to_parent && !ancestor_refresh_suppressed_ && parent_ != nullptr &&
      !affected_values.empty()) {
    if (is_local_function_) {
      parent_->RefreshLocalFunctionDefinition(*this);
    }
    parent_->RefreshNestedDependency(*this);
  }
  return affected;
}

void GraphBuilder::RefreshAfterRewriting(const std::vector<std::size_t> &old_indices,
                                         std::unordered_set<std::string> affected_values) {
  compute::ComputeContext::ValueTagPreservationGuard preserve_tags(compute_);
  std::unordered_set<GraphBuilder *> referenced_subgraphs;
  for (const auto &node : nodes_) {
    for (GraphBuilder *child : ReferencedSubgraphs(node)) {
      referenced_subgraphs.insert(child);
    }
  }
  subgraphs_.erase(std::remove_if(subgraphs_.begin(), subgraphs_.end(),
                                  [&](const auto &child) {
                                    return referenced_subgraphs.count(child.get()) == 0;
                                  }),
                   subgraphs_.end());
  std::unordered_set<std::string> available = inherited_names_;
  for (const auto &input : inputs_) {
    available.insert(input.name().value());
  }
  for (const auto &initializer : initializers_) {
    available.insert(initializer.name().value());
  }
  for (const auto &initializer : extended_initializers_) {
    available.insert(initializer.name().value());
  }
  for (const auto &initializer : graph_template_.sparse_initializer()) {
    available.insert(initializer.values().name().value());
  }
  const auto old_count = compute_.Size();
  std::vector<std::vector<std::string>> references(nodes_.size());
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    CollectNodeReferences(nodes_[index], references[index]);
    if (old_indices[index] == old_count) {
      for (const auto &name : references[index])
        SeedInheritedValue(name);
    }
  }
  for (const auto &initializer : initializers_) {
    const std::string name = initializer.name().value();
    if (affected_values.count(name) != 0) {
      const bool is_input = std::any_of(inputs_.begin(), inputs_.end(), [&](const auto &input) {
        return input.name() == initializer.name();
      });
      compute_.SeedReuseInput(name, is_input, /*is_initializer=*/true,
                              /*allow_input_overwrite=*/false);
    }
  }
  std::vector<bool> added(nodes_.size());
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    added[index] = old_indices[index] == old_count;
  }
  auto order = InsertNodesInDependencyOrder(nodes_, available, references, added);
  auto inserted_indices = old_indices;
  if (!order.empty()) {
    utils::RepeatedProtoField<NodeProto> inserted;
    inserted.reserve(nodes_.size());
    std::vector<std::vector<std::string>> inserted_references;
    inserted_references.reserve(nodes_.size());
    for (std::size_t position = 0; position < order.size(); ++position) {
      const auto index = order[position];
      inserted.add();
      inserted.get(position) = nodes_.shared_at(index);
      inserted_indices[position] = old_indices[index];
      inserted_references.push_back(std::move(references[index]));
    }
    nodes_ = std::move(inserted);
    references = std::move(inserted_references);
  }
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    if (inserted_indices[index] == old_count ||
        std::any_of(references[index].begin(), references[index].end(),
                    [&](const auto &name) { return affected_values.count(name) != 0; })) {
      for (const auto &output : nodes_[index].output()) {
        if (!output.empty()) {
          affected_values.insert(output.value());
        }
      }
    }
  }
  compute_.UpdateNodes(nodes_, inserted_indices, references, affected_values, device_);
  auto &shapes = compute_.Shapes();
  for (const auto &name : affected_values) {
    if (parent_ != nullptr && inherited_names_.count(name) != 0) {
      shapes.Erase(name);
      SeedInheritedValue(name);
    }
  }
  for (const auto &initializer : initializers_) {
    const std::string name = initializer.name().value();
    if (affected_values.count(name) == 0) {
      continue;
    }
    const bool is_input = std::any_of(inputs_.begin(), inputs_.end(), [&](const auto &input) {
      return input.name() == initializer.name();
    });
    if (!is_input) {
      shapes.Erase(name);
      SymTensor tensor;
      if (SymTensorFromTensorProto(initializer, tensor)) {
        shapes.Set(name, std::move(tensor));
      }
      compute_.SeedValueTag(name, "weight");
      compute_.SeedConstant(name);
    }
    compute_.SeedValueTag(name, DeclaredValueTag(initializer), nodes_);
  }
  for (const auto &info : value_infos_) {
    if (affected_values.count(info.name().value()) != 0) {
      compute_.SeedValueTag(info.name().value(), DeclaredValueTag(info), nodes_);
    }
  }
  for (const auto &input : inputs_) {
    if (affected_values.count(input.name().value()) != 0) {
      compute_.SeedValueTag(input.name().value(), DeclaredValueTag(input), nodes_);
    }
  }
  for (const auto &output : outputs_) {
    if (affected_values.count(output.name().value()) != 0) {
      compute_.SeedValueTag(output.name().value(), DeclaredValueTag(output), nodes_);
    }
  }
  compute_.RefreshValueTags(nodes_, affected_values);
  RefreshDependentNodes(
      [&](const NodeProto &node, const std::unordered_set<std::string> &) {
        return std::any_of(node.output().begin(), node.output().end(),
                           [&](const auto &name) { return affected_values.count(name) != 0; });
      },
      affected_values, /*propagate_to_parent=*/false);
  for (const auto &child : subgraphs_) {
    std::unordered_set<std::string> captures;
    child->CollectImplicitInputs(captures);
    const auto &child_shapes = child->Shapes();
    std::unordered_set<std::string> changed_captures;
    for (const auto &name : affected_values) {
      if (captures.count(name) != 0 ||
          (child->inherited_names_.count(name) != 0 &&
           (child_shapes.Has(name) || child_shapes.HasType(name) ||
            child_shapes.HasSequence(name) || child_shapes.HasEncodedValue(name)))) {
        changed_captures.insert(name);
      }
    }
    if (!changed_captures.empty()) {
      AncestorRefreshGuard guard(*child);
      std::vector<std::size_t> child_indices(child->nodes_.size());
      std::iota(child_indices.begin(), child_indices.end(), 0);
      child->RefreshAfterRewriting(child_indices, std::move(changed_captures));
    }
  }
  RefreshAfterMutation();
}

void GraphBuilder::RefreshAfterMutation() {
  compute_.RefreshIncrementalMemory(
      nodes_, compute_.Shapes(), device_,
      [this](const NodeProto &node) { return LayerBodyForMemory(node); });
  RefreshAncestors();
}

compute::ComputeContext::LayerBodyView
GraphBuilder::LayerBodyForMemory(const NodeProto &node) const {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() != "body_ref" || !attribute.has_s())
      continue;
    const GraphBuilder &body = Subgraph(attribute.s());
    compute::ComputeContext::LayerBodyView view;
    view.nodes = &body.nodes_;
    view.inputs = &body.inputs_;
    view.outputs = &body.outputs_;
    for (const auto &initializer : body.initializers_)
      view.initializer_names.push_back(initializer.name());
    for (const auto &initializer : body.graph_template_.sparse_initializer())
      view.initializer_names.push_back(initializer.values().name());
    for (const auto &initializer : body.extended_initializers_)
      view.initializer_names.push_back(initializer.name());
    view.nested_body = [&body](const NodeProto &inner) { return body.LayerBodyForMemory(inner); };
    view.node_inputs = [&body](std::size_t index) -> const std::vector<std::string> & {
      return body.compute_.NodeReferences(index);
    };
    return view;
  }
  throw BuilderError("GraphBuilder: missing Layer body reference for memory estimation.");
}

void GraphBuilder::RebuildMetadata() {
  InvalidateRewritings();
  compute_.ResetMetadata();
  for (const auto &value : value_infos_) {
    compute_.SeedValueTag(value.name().value(), DeclaredValueTag(value));
  }
  std::unordered_set<std::string> input_names;
  input_names.reserve(inputs_.size());
  for (const auto &input : inputs_) {
    const std::string name = input.name().value();
    input_names.insert(name);
    SeedInputAnnotations(name);
    compute_.SeedValueTag(name, DeclaredValueTag(input));
  }
  for (const auto &initializer : initializers_) {
    const std::string name = initializer.name().value();
    const bool is_input = input_names.count(name) != 0;
    compute_.SeedValueTag(name, DeclaredValueTag(initializer));
    compute_.SeedValueTag(name, "weight");
    compute_.SeedReuseInput(name, is_input, /*is_initializer=*/true,
                            /*allow_input_overwrite=*/false);
    if (!is_input) {
      compute_.SeedConstant(name);
    }
  }
  for (const auto &initializer : extended_initializers_) {
    const std::string name = initializer.name().value();
    const bool is_input = input_names.count(name) != 0;
    compute_.SeedValueTag(name, "weight");
    compute_.SeedReuseInput(name, is_input, /*is_initializer=*/true,
                            /*allow_input_overwrite=*/false);
    if (!is_input) {
      compute_.SeedConstant(name);
    }
  }
  for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
    const NodeProto &node = nodes_[node_index];
    compute_.AppendNodeTags(nodes_, node_index);
    std::vector<std::string> references;
    CollectNodeReferences(node, references);
    compute_.AppendNodeReuse(nodes_, node_index, compute_.Shapes(), references);
    compute_.AppendNodeConstant(node, node_index);
    compute_.AppendNodePeakMemory(node, node_index, device_);
  }
  for (const auto &output : outputs_) {
    const std::string name = output.name().value();
    compute_.SeedValueTag(name, DeclaredValueTag(output), nodes_);
    compute_.SeedReuseOutput(name, nodes_, compute_.Shapes());
  }

  GraphProto graph = BuildGraphImpl(/*validate_persistent_bindings=*/false,
                                    /*share_initializers=*/true);
  const auto tags = compute_.ComputeValueAndNodeTags(graph);
  compute_.ComputeConstants(graph);
  compute_.ComputeInPlaceReuseGraph(graph, compute_.Shapes(), /*allow_input_overwrite=*/false,
                                    tags.first);
  compute_.ComputePeakMemory(graph, device_);
}

void GraphBuilder::RebuildStructuredState(bool refresh_ancestors) {
  InvalidateRewritings();
  auto &shapes = compute_.Shapes();
  shapes.ClearInferredState();
  for (const auto &[domain, version] : opsets_) {
    shapes.SetOpsetVersion(domain, version);
  }
  if (parent_ != nullptr) {
    for (const auto &name : inherited_names_) {
      const auto &outer = parent_->Shapes();
      if (outer.Has(name) || outer.HasType(name) || outer.HasSequence(name)) {
        shapes.CopyValueFrom(name, outer, name);
      }
    }
  }
  std::unordered_set<std::string> input_names;
  for (const auto &input : inputs_) {
    input_names.insert(input.name().value());
    SeedDeclaredValueInfo(shapes, input);
    for (const auto &output : outputs_) {
      if (output.name() == input.name()) {
        SeedDeclaredValueInfo(shapes, output);
      }
    }
  }
  for (const auto &initializer : initializers_) {
    if (input_names.count(initializer.name().value()) == 0) {
      SymTensor tensor;
      if (SymTensorFromTensorProto(initializer, tensor)) {
        shapes.Set(initializer.name().value(), std::move(tensor));
      }
    }
  }
  for (const auto &initializer : extended_initializers_) {
    if (input_names.count(initializer.name().value()) == 0) {
      if (initializer.has_encoded_value()) {
        shapes.SetEncodedValue(initializer.name().value(), initializer.encoded_value());
      } else {
        TypeProto type;
        type.mutable_struct_type();
        shapes.SetType(initializer.name().value(), type);
      }
    }
  }
  const auto declarations = StructuredDeclarations(value_infos_);
  RefreshLocalFunctions();
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    NodeProto node = utils::CopyWithSharedTensorStorage(nodes_[i]);
    MaterializeGraphReferences(node, /*share_initializers=*/true);
    const bool ready = ShapeFunctionAvailable(node) && !HasUnboundAttributes(node) &&
                       ShapeInputsAvailable(shapes, node);
    if (ready) {
      shapes.set_current_node_index(static_cast<int64_t>(i));
      shapes.ComputeShapeNode(node);
    }
    SeedDeclaredOutputs(shapes, node, declarations);
    for (const auto &output : outputs_) {
      if (std::find(node.output().begin(), node.output().end(), output.name().value()) !=
          node.output().end()) {
        SeedDeclaredValueInfo(shapes, output);
      }
    }
  }
  for (const auto &output : outputs_) {
    SeedDeclaredValueInfo(shapes, output);
  }
  for (const auto &child : subgraphs_) {
    child->RebuildStructuredState(/*refresh_ancestors=*/false);
  }
  RebuildMetadata();
  if (refresh_ancestors) {
    RefreshAncestors();
  }
}

const std::string &GraphBuilder::MakeEncodedInitializer(const EncodedValueProto &value) {
  const std::string name = value.name().value();
  const auto input = std::find_if(inputs_.begin(), inputs_.end(), [&](const ValueInfoProto &v) {
    return v.name().value() == name;
  });
  const bool is_input = input != inputs_.end();
  const bool has_initializer =
      std::any_of(initializers_.begin(), initializers_.end(),
                  [&](const TensorProto &v) { return v.name().value() == name; }) ||
      std::any_of(extended_initializers_.begin(), extended_initializers_.end(),
                  [&](const auto &v) { return v.name().value() == name; });
  if (name.empty() || (HasName(name) && (!is_input || has_initializer))) {
    throw BuilderError("GraphBuilder: encoded initializer name is empty or already defined: '" +
                       name + "'.");
  }
  // Validation happens in an isolated context so failure cannot reserve a name
  // or partially replace an input's descriptor.
  ShapesContext validated;
  validated.SetStructTypes(Shapes().StructTypes());
  validated.SetEncodedValue(name, value);
  if (is_input && input->has_type()) {
    const TypeProto &actual = value.has_logical_type() && input->type().has_tensor_type()
                                  ? value.logical_type()
                                  : validated.GetType(name);
    if (!CompatibleEncodedDefault(input->type(), actual)) {
      throw BuilderError("GraphBuilder: encoded initializer '" + name +
                         "' has a different type from its declared input.");
    }
  }
  const std::string &reserved = is_input ? *names_.find(name) : ReserveName(name);
  // Native assignment retains borrowed payload owners without a wire round-trip.
  *extended_initializers_.Add()->mutable_encoded_value() =
      utils::CopyWithSharedTensorStorage(value);
  if (!is_input) {
    compute_.Shapes().SetEncodedValue(name, value);
  }
  compute_.SeedValueTag(reserved, "weight");
  compute_.SeedReuseInput(reserved, is_input, /*is_initializer=*/true,
                          /*allow_input_overwrite=*/false);
  if (!is_input) {
    compute_.SeedConstant(reserved);
  }
  RefreshAfterMutation();
  return reserved;
}

const std::string &GraphBuilder::MakePagedCacheInitializer(const PagedCacheProto &value) {
  const std::string name = value.name().value();
  const auto input = std::find_if(inputs_.begin(), inputs_.end(),
                                  [&](const auto &v) { return v.name().value() == name; });
  const bool is_input = input != inputs_.end();
  const auto named = [&](const auto &v) { return v.name().value() == name; };
  const bool has_initializer =
      std::any_of(initializers_.begin(), initializers_.end(), named) ||
      std::any_of(extended_initializers_.begin(), extended_initializers_.end(), named);
  if (name.empty() || (HasName(name) && (!is_input || has_initializer)))
    throw BuilderError("GraphBuilder: paged cache initializer name is empty or already defined.");
  ModelProto declarations;
  declarations.ref_struct_types() = Shapes().StructTypes();
  StructTypeCatalogue catalogue;
  catalogue.Build(declarations);
  catalogue.ValidatePagedCache(value, true,
                               is_input && input->has_type() ? &input->type() : nullptr);
  const std::string &reserved = is_input ? *names_.find(name) : ReserveName(name);
  *extended_initializers_.Add()->mutable_paged_cache() = utils::CopyWithSharedTensorStorage(value);
  if (!is_input) {
    TypeProto type;
    type.mutable_struct_type();
    compute_.Shapes().SetType(name, type);
  }
  compute_.SeedValueTag(reserved, "weight");
  compute_.SeedReuseInput(reserved, is_input, true, false);
  if (!is_input)
    compute_.SeedConstant(reserved);
  RefreshAfterMutation();
  return reserved;
}

const std::string &GraphBuilder::MakeInitializer(const TensorProto &tensor) {
  return MakeInitializerImpl(utils::CopyWithSharedTensorStorage(tensor));
}

const std::string &GraphBuilder::MakeInitializerMove(TensorProto &&tensor) {
  const std::string &name = MakeInitializerImpl(std::move(tensor));
  tensor.Clear();
  return name;
}

template <typename Tensor> const std::string &GraphBuilder::MakeInitializerImpl(Tensor &&tensor) {
  const std::string tensor_name = tensor.name().value();
  const auto input = std::find_if(inputs_.begin(), inputs_.end(), [&](const ValueInfoProto &value) {
    return value.name().value() == tensor_name;
  });
  const bool is_input = input != inputs_.end();
  const bool has_initializer =
      std::any_of(initializers_.begin(), initializers_.end(),
                  [&](const TensorProto &initializer) {
                    return initializer.name().value() == tensor_name;
                  }) ||
      std::any_of(
          extended_initializers_.begin(), extended_initializers_.end(),
          [&](const auto &initializer) { return initializer.name().value() == tensor_name; });
  if (is_input && !has_initializer && input->has_type()) {
    if (!input->type().has_tensor_type()) {
      throw BuilderError("GraphBuilder: initializer '" + tensor_name +
                         "' cannot supply a default for a non-tensor input.");
    }
    const auto &declared = input->type().tensor_type();
    if (declared.elem_type() != TensorProto::DataType::UNDEFINED &&
        declared.elem_type() != tensor.data_type()) {
      throw BuilderError("GraphBuilder: initializer '" + tensor_name +
                         "' has a different element type from its declared input.");
    }
    if (declared.has_shape()) {
      if (declared.shape().dim().size() != tensor.dims().size()) {
        throw BuilderError("GraphBuilder: initializer '" + tensor_name +
                           "' has a different rank from its declared input.");
      }
      for (std::size_t i = 0; i < tensor.dims().size(); ++i) {
        const auto &dimension = declared.shape().dim(i);
        if (dimension.has_dim_value() && dimension.dim_value() != tensor.dims(i)) {
          throw BuilderError("GraphBuilder: initializer '" + tensor_name + "' dimension " +
                             std::to_string(i) + " differs from its declared input.");
        }
      }
    }
  }
  SymTensor descriptor;
  const bool has_descriptor = !is_input && SymTensorFromTensorProto(tensor, descriptor);
  // ONNX permits an initializer to supply the default for a public input.
  const std::string &reserved =
      is_input && !has_initializer ? *names_.find(tensor_name) : ReserveName(tensor_name);
  initializers_.push_back(std::forward<Tensor>(tensor));
  compute_.SeedValueTag(reserved, DeclaredValueTag(initializers_.back()), nodes_);
  if (is_input) {
    // Defaults remain overridable; neither their values nor their dimensions
    // specialize the public input declaration.
    compute_.SeedReuseInput(reserved, /*is_graph_input=*/true, /*is_initializer=*/true,
                            /*allow_input_overwrite=*/false);
    RefreshAfterMutation();
    return reserved;
  }
  if (has_descriptor) {
    SeedShape(reserved, std::move(descriptor));
  }
  // Seed the incremental annotations: an initializer is a "weight", is kept
  // from in-place reuse, and is a constant value.
  compute_.SeedValueTag(reserved, "weight");
  compute_.SeedReuseInput(reserved, /*is_graph_input=*/false, /*is_initializer=*/true,
                          /*allow_input_overwrite=*/false);
  compute_.SeedConstant(reserved);
  RefreshAfterMutation();
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
  if (value_info.has_type() && HasStructuredType(value_info.type())) {
    ShapesContext validated;
    validated.SetStructTypes(Shapes().StructTypes());
    validated.SetType(value_info.name().value(), value_info.type());
  }
  // Reserve the name first so a duplicate is rejected before ``inputs_`` is
  // mutated, leaving the builder unchanged on error.
  const std::string &reserved = ReserveName(value_info.name().value());
  inputs_.push_back(value_info);
  if (value_info.has_type() && !value_info.type().has_tensor_type()) {
    compute_.Shapes().SetType(reserved, value_info.type());
  }
  SymTensor descriptor;
  if (value_info.has_type() && value_info.type().has_tensor_type() &&
      value_info.type().tensor_type().has_shape() &&
      SymTensorFromValueInfo(value_info, descriptor)) {
    SeedShape(reserved, std::move(descriptor));
  }
  SeedInputAnnotations(reserved);
  compute_.SeedValueTag(reserved, DeclaredValueTag(value_info), nodes_);
  RefreshAfterMutation();
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
  RefreshAfterMutation();
  return reserved;
}

const std::string &GraphBuilder::MakeInput(const std::string &name, TensorType dtype,
                                           const SymShape &shape) {
  return MakeInput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const ValueInfoProto &value_info) {
  const std::string name = value_info.name().value();
  ValidateSubgraphOutput(name);
  ValueInfoProto output = value_info;
  if (output.has_type() && output.type().has_tensor_type() &&
      output.type().tensor_type().has_shape()) {
    for (auto &dim : output.mutable_type()->mutable_tensor_type()->mutable_shape()->ref_dim()) {
      if (!dim.has_dim_value() && !dim.has_dim_param()) {
        dim.set_dim_param("");
      }
    }
  }
  bool tensor_refined = false;
  SymTensor refined;
  if (value_info.has_type() && value_info.type().has_tensor_type() && Shapes().Has(name)) {
    const auto &declared = value_info.type().tensor_type();
    const SymTensor &inferred = Shapes().Get(name);
    auto dtype = TensorTypeToDataType(inferred.Dtype());
    bool compatible = declared.elem_type() == TensorProto::UNDEFINED ||
                      dtype == TensorProto::UNDEFINED || declared.elem_type() == dtype;
    if (declared.has_shape()) {
      const auto &dims = declared.shape().dim();
      compatible = compatible && dims.size() == inferred.Shape().Rank();
      for (std::size_t i = 0; compatible && i < dims.size(); ++i) {
        const auto &actual = inferred.Shape()[i];
        compatible =
            !dims[i].has_dim_value() || !actual.IsInt() || dims[i].dim_value() == actual.AsInt();
      }
    }
    if (!compatible) {
      throw BuilderError("GraphBuilder: output '" + name +
                         "' has an incompatible tensor type or shape.");
    }
    if (dtype == TensorProto::UNDEFINED && declared.elem_type() != TensorProto::UNDEFINED) {
      dtype = declared.elem_type();
      const auto refined_dtype = core::symbolic::DataTypeToTensorType(dtype);
      if (refined_dtype == TensorType::kUndefined) {
        throw BuilderError("GraphBuilder: output '" + name +
                           "' has an unsupported tensor element type.");
      }
      refined = RefineTensorDtype(inferred, refined_dtype);
      tensor_refined = true;
    }
    auto *tensor = output.mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(dtype);
    tensor->clear_shape();
    auto *shape = tensor->mutable_shape();
    for (const auto &dimension : inferred.Shape().Dims()) {
      auto *dim = shape->add_dim();
      if (dimension.IsInt()) {
        dim->set_dim_value(dimension.AsInt());
      } else {
        dim->set_dim_param(dimension.AsExpr());
      }
    }
  } else if (value_info.has_type() && value_info.type().has_tensor_type() &&
             Shapes().HasType(name) && Shapes().GetType(name).has_tensor_type()) {
    const auto inferred_dtype = Shapes().GetType(name).tensor_type().elem_type();
    const auto declared_dtype = value_info.type().tensor_type().elem_type();
    if (inferred_dtype != TensorProto::UNDEFINED && declared_dtype != TensorProto::UNDEFINED &&
        inferred_dtype != declared_dtype) {
      throw BuilderError("GraphBuilder: output '" + name +
                         "' has an incompatible tensor element type.");
    }
    if (inferred_dtype != TensorProto::UNDEFINED) {
      output.mutable_type()->mutable_tensor_type()->set_elem_type(inferred_dtype);
    }
  }
  bool declaration_added = false;
  if (value_info.has_type() &&
      (HasStructuredType(value_info.type()) || Shapes().HasEncodedValue(name))) {
    ShapesContext validated;
    validated.SetStructTypes(Shapes().StructTypes());
    validated.SetType(name, value_info.type());
    if (Shapes().HasType(name) &&
        !CompatibleEncodedDefault(value_info.type(), Shapes().GetType(name))) {
      throw BuilderError("GraphBuilder: output '" + name +
                         "' has an incompatible structured type.");
    }
  }
  if (value_info.has_type() && !value_info.type().has_tensor_type() &&
      !Shapes().HasEncodedValue(value_info.name().value())) {
    compute_.Shapes().SetType(value_info.name().value(), value_info.type());
    declaration_added = true;
  }
  if (tensor_refined) {
    SeedShape(name, std::move(refined));
  }
  SymTensor descriptor;
  if (value_info.has_type() && value_info.type().has_tensor_type() &&
      value_info.type().tensor_type().has_shape() && !Shapes().Has(name) &&
      SymTensorFromValueInfo(output, descriptor)) {
    SeedShape(name, std::move(descriptor));
    tensor_refined = true;
  }
  outputs_.push_back(std::move(output));
  compute_.SeedValueTag(name, DeclaredValueTag(value_info), nodes_);
  compute_.SeedReuseOutput(value_info.name().value(), nodes_, compute_.Shapes());
  if (declaration_added) {
    RebuildStructuredState();
  } else {
    if (tensor_refined) {
      std::unordered_set<std::string> affected_values{name};
      RefreshDependentNodes(
          [](const NodeProto &, const std::unordered_set<std::string> &) { return false; },
          affected_values, /*propagate_to_parent=*/false);
    }
    RefreshAfterMutation();
  }
}

void GraphBuilder::MakeOutput(const std::string &name, const SymTensor &type) {
  ValueInfoProto vi;
  vi.set_name(name);
  SymTensorToValueInfo(type, vi);
  MakeOutput(vi);
}

void GraphBuilder::MakeOutput(const std::string &name, TensorType dtype, const SymShape &shape) {
  MakeOutput(name, SymTensor(nullptr, dtype, shape));
}

void GraphBuilder::MakeOutput(const std::string &name) {
  ValidateSubgraphOutput(name);
  ValueInfoProto vi;
  vi.set_name(name);
  outputs_.add() = std::move(vi);
  compute_.SeedReuseOutput(name, nodes_, compute_.Shapes());
  RefreshAfterMutation();
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

bool GraphBuilder::NodeCarriesSubgraph(const NodeProto &node) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.type() == AttributeProto::AttributeType::GRAPH ||
        attribute.type() == AttributeProto::AttributeType::GRAPHS ||
        HasGraphReferenceSuffix(attribute.name().value())) {
      return true;
    }
  }
  return false;
}

utils::RepeatedProtoField<AttributeProto>
GraphBuilder::ImportAttributes(const NodeProto &node,
                               const std::unordered_set<std::string> &excluded_inherited_names) {
  const auto has_graph_content = [](const GraphProto &graph) {
    return !graph.name().empty() || graph.input().size() > 0 || graph.output().size() > 0 ||
           graph.node().size() > 0 || graph.initializer().size() > 0 ||
           graph.value_info().size() > 0 || !graph.extended_initializer().empty();
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
      for (const std::string &name : excluded_inherited_names) {
        subgraph.inherited_names_.erase(name);
        subgraph.compute_.Shapes().Erase(name);
      }
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
      for (std::size_t i = 0; i < graphs.size(); ++i) {
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
        for (const std::string &name : excluded_inherited_names) {
          subgraph.inherited_names_.erase(name);
          subgraph.compute_.Shapes().Erase(name);
        }
        subgraph.ImportGraph(graph);
        refs.add_strings(subgraph_name);
      }
      imported.push_back(std::move(refs));
      continue;
    }
    imported.push_back(utils::CopyWithSharedTensorStorage(attribute));
  }
  return imported;
}

void GraphBuilder::ImportGraph(const GraphProto &graph) {
  if (parent_ != nullptr && !graph.persistent_bindings().empty()) {
    throw BuilderError("GraphBuilder: persistent bindings are supported only on the root graph.");
  }
  // Preserve only fields emitted from the template. Copying the complete graph
  // serializes nested graph attributes and their sparse payloads before the
  // imported node list is immediately discarded.
  graph_template_ = GraphProto{};
  graph_template_.name_ = graph.name_;
  graph_template_.sparse_initializer_ =
      utils::CopyWithSharedTensorStorage(graph.sparse_initializer_);
  graph_template_.doc_string_ = graph.doc_string_;
  graph_template_.quantization_annotation_ = graph.quantization_annotation_;
  graph_template_.metadata_props_ = graph.metadata_props_;
  graph_template_.persistent_bindings_ = graph.persistent_bindings_;
  value_infos_ = graph.value_info();
  for (const auto &value : value_infos_) {
    compute_.SeedValueTag(value.name().value(), DeclaredValueTag(value), nodes_);
  }
  for (const auto &input : graph.input()) {
    MakeInput(input);
  }
  for (const auto &initializer : graph.initializer()) {
    MakeInitializer(initializer);
  }
  for (const auto &initializer : graph.extended_initializer()) {
    if (initializer.has_encoded_value())
      MakeEncodedInitializer(initializer.encoded_value());
    else if (initializer.has_paged_cache())
      MakePagedCacheInitializer(initializer.paged_cache());
    else
      throw BuilderError("GraphBuilder: extended initializer has no payload.");
  }
  std::unordered_set<std::string> sparse_names;
  for (const auto &initializer : graph.sparse_initializer()) {
    const std::string name = initializer.values().name().value();
    const bool is_input = std::any_of(inputs_.begin(), inputs_.end(),
                                      [&](const auto &input) { return input.name() == name; });
    const auto named = [&](const auto &value) { return value.name().value() == name; };
    const bool has_other_initializer =
        std::any_of(initializers_.begin(), initializers_.end(), named) ||
        std::any_of(extended_initializers_.begin(), extended_initializers_.end(), named);
    if (name.empty() || !sparse_names.insert(name).second || has_other_initializer ||
        (names_.find(name) != names_.end() && !is_input)) {
      throw BuilderError("GraphBuilder: sparse initializer name is empty or already defined: '" +
                         name + "'.");
    }
    if (!is_input) {
      inherited_names_.erase(name);
      compute_.Shapes().Erase(name);
      ReserveName(name);
    }
  }
  const auto declarations = StructuredDeclarations(value_infos_);
  for (const auto &node : graph.node()) {
    std::vector<std::string> inputs;
    inputs.reserve(node.input().size());
    for (std::size_t i = 0; i < node.input().size(); ++i) {
      inputs.push_back(node.input(static_cast<std::size_t>(i)));
    }
    std::vector<std::string> outputs;
    outputs.reserve(node.output().size());
    for (std::size_t i = 0; i < node.output().size(); ++i) {
      outputs.push_back(node.output(static_cast<std::size_t>(i)));
    }
    MakeNode(node.op_type().value(), inputs, outputs,
             node.domain().empty() ? std::string() : node.domain().value(),
             node.name().empty() ? std::string() : node.name().value(), ImportAttributes(node),
             node.overload().value());
    NodeProto &imported_node = nodes_.back();
    imported_node.name_ = node.name_;
    imported_node.op_type_ = node.op_type_;
    imported_node.domain_ = node.domain_;
    imported_node.overload_ = node.overload_;
    imported_node.doc_string_ = node.doc_string_;
    imported_node.metadata_props_ = node.metadata_props_;
    imported_node.device_configurations_ = node.device_configurations_;
    SeedDeclaredOutputs(compute_.Shapes(), node, declarations);
  }
  for (const auto &output : graph.output()) {
    MakeOutput(output);
  }
}

void GraphBuilder::ImportFunction(const FunctionProto &function) {
  // Copy scalar metadata directly. Copying the whole function serializes nested
  // graph attributes before immediately discarding its repeated fields.
  function_template_ = FunctionProto{};
  function_template_.name_ = function.name_;
  function_template_.domain_ = function.domain_;
  function_template_.doc_string_ = function.doc_string_;
  function_template_.overload_ = function.overload_;
  function_attributes_.assign(function.attribute().begin(), function.attribute().end());
  function_attribute_protos_ = utils::CopyWithSharedTensorStorage(function.attribute_proto());
  metadata_ = function.metadata_props();
  if (function.has_doc_string()) {
    doc_string_ = function.doc_string().value();
  }
  for (const auto &opset : function.opset_import()) {
    SetOpsetVersion(opset.domain().empty() ? std::string() : opset.domain().value(),
                    static_cast<int>(opset.version()));
  }
  for (std::size_t i = 0; i < function.input().size(); ++i) {
    ValueInfoProto value_info;
    value_info.set_name(function.input(static_cast<std::size_t>(i)));
    for (const auto &declared : function.value_info()) {
      if (declared.name() == value_info.name()) {
        value_info = declared;
        break;
      }
    }
    MakeInput(value_info);
  }
  for (const auto &node : function.node()) {
    std::vector<std::string> inputs;
    inputs.reserve(node.input().size());
    for (std::size_t i = 0; i < node.input().size(); ++i) {
      inputs.push_back(node.input(static_cast<std::size_t>(i)));
    }
    std::vector<std::string> outputs;
    outputs.reserve(node.output().size());
    for (std::size_t i = 0; i < node.output().size(); ++i) {
      outputs.push_back(node.output(static_cast<std::size_t>(i)));
    }
    MakeNode(node.op_type().value(), inputs, outputs,
             node.domain().empty() ? std::string() : node.domain().value(),
             node.name().empty() ? std::string() : node.name().value(), ImportAttributes(node),
             node.overload().value());
    NodeProto &imported_node = nodes_.back();
    imported_node.name_ = node.name_;
    imported_node.op_type_ = node.op_type_;
    imported_node.domain_ = node.domain_;
    imported_node.overload_ = node.overload_;
    imported_node.doc_string_ = node.doc_string_;
    imported_node.metadata_props_ = node.metadata_props_;
    imported_node.device_configurations_ = node.device_configurations_;
  }
  for (std::size_t i = 0; i < function.output().size(); ++i) {
    ValueInfoProto value_info;
    value_info.set_name(function.output(static_cast<std::size_t>(i)));
    for (const auto &declared : function.value_info()) {
      if (declared.name() == value_info.name()) {
        value_info = declared;
        break;
      }
    }
    MakeOutput(value_info);
  }
  for (const auto &declared : function.value_info()) {
    const bool is_input = std::find(function.input().begin(), function.input().end(),
                                    declared.name()) != function.input().end();
    const bool is_output = std::find(function.output().begin(), function.output().end(),
                                     declared.name()) != function.output().end();
    if (!is_input && !is_output) {
      value_infos_.push_back(declared);
      compute_.SeedValueTag(declared.name().value(), DeclaredValueTag(declared), nodes_);
    }
  }
  RebuildStructuredState();
}

void GraphBuilder::MaterializeGraphReferences(NodeProto &node, bool share_initializers) const {
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
        *graph_attribute.mutable_g() = Subgraph(ref_name).BuildGraphImpl(true, share_initializers);
        attribute = std::move(graph_attribute);
      }
    } else if (attribute.type() == AttributeProto::AttributeType::STRINGS &&
               attribute.strings().size() > 0) {
      AttributeProto graphs_attribute;
      graphs_attribute.set_name(base_name);
      graphs_attribute.set_type(AttributeProto::AttributeType::GRAPHS);
      const auto &refs = attribute.strings();
      for (std::size_t i = 0; i < refs.size(); ++i) {
        const std::string ref_name = refs[static_cast<std::size_t>(i)];
        if (HasSubgraph(ref_name)) {
          *graphs_attribute.add_graphs() =
              Subgraph(ref_name).BuildGraphImpl(true, share_initializers);
        }
      }
      attribute = std::move(graphs_attribute);
    }
  }
}

// ── Nodes ──────────────────────────────────────────────────────────────

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
  return (domain == "ai.rt" && node.op_type() == "Layer") || table.find(key) != table.end() ||
         compute_.Shapes().HasLocalFunction(
             ShapesContext::LocalFunctionKey(domain, node.op_type(), node.overload()));
}

void GraphBuilder::RefreshLocalFunctions() {
  GraphBuilder *root = this;
  while (root->parent_ != nullptr) {
    root = root->parent_;
  }
  std::vector<GraphBuilder *> builders;
  std::function<void(GraphBuilder &)> collect = [&](GraphBuilder &builder) {
    builders.push_back(&builder);
    builder.compute_.Shapes().ClearLocalFunctions();
    for (const auto &child : builder.local_functions_) {
      collect(*child);
    }
    for (const auto &child : builder.subgraphs_) {
      collect(*child);
    }
  };
  collect(*root);
  std::unordered_set<std::string> function_keys;
  for (GraphBuilder *builder : builders) {
    for (const auto &function : builder->local_functions_) {
      const std::string key = ShapesContext::LocalFunctionKey(
          function->function_domain_, function->name_, function->function_template_.overload());
      if (!function_keys.insert(key).second) {
        throw BuilderError("GraphBuilder: duplicate local function '" + key + "'.");
      }
      const auto definition =
          std::make_shared<FunctionProto>(function->BuildFunction(function->function_domain_));
      for (GraphBuilder *target : builders) {
        target->compute_.Shapes().SetLocalFunction(definition);
      }
    }
  }
}

std::vector<std::string> GraphBuilder::MakeNode(
    const std::string &op_type, const std::vector<std::string> &inputs,
    const std::vector<std::string> &outputs, const std::string &domain, const std::string &name,
    const utils::RepeatedProtoField<AttributeProto> &attributes, const std::string &overload) {
  const int opset = OpsetVersion(domain);
  if (opset <= 0) {
    throw BuilderError("GraphBuilder: node '" + op_type + "' has no explicit opset for domain '" +
                       NormaliseDomain(domain) +
                       "'; call SetOpsetVersion() first or supply "
                       "opsets to the constructor.");
  }
  {
    GraphBuilder *root = this;
    while (root->parent_ != nullptr) {
      root = root->parent_;
    }
    const std::string function_key = ShapesContext::LocalFunctionKey(domain, op_type, overload);
    bool calls_function = false;
    bool has_local_functions = false;
    std::function<void(const GraphBuilder &)> inspect = [&](const GraphBuilder &builder) {
      for (const auto &function : builder.local_functions_) {
        has_local_functions = true;
        if (ShapesContext::LocalFunctionKey(function->function_domain_, function->name_,
                                            function->function_template_.overload()) ==
            function_key) {
          calls_function = true;
        }
        inspect(*function);
      }
      for (const auto &subgraph : builder.subgraphs_) {
        inspect(*subgraph);
      }
    };
    inspect(*root);
    const bool may_call_function_in_graph =
        has_local_functions &&
        std::any_of(attributes.begin(), attributes.end(), [](const AttributeProto &attribute) {
          return attribute.type() == AttributeProto::AttributeType::GRAPH ||
                 attribute.type() == AttributeProto::AttributeType::GRAPHS ||
                 HasGraphReferenceSuffix(attribute.name().value());
        });
    if (calls_function || may_call_function_in_graph ||
        compute_.Shapes().HasLocalFunction(function_key)) {
      RefreshLocalFunctions();
    }
  }
  // Every non-empty input must reference a value that already exists. Empty
  // strings denote skipped optional inputs and are allowed.
  for (const std::string &input : inputs) {
    if (!input.empty() && !HasName(input)) {
      throw BuilderError("GraphBuilder: node '" + op_type + "' reads the unknown value '" + input +
                         "'.");
    }
  }

  // Select the matching schema (if any) via the cached
  // per-operator, per-domain lookup table.
  const std::string normalised_domain = NormaliseDomain(domain);
  const FunctionProto *local_function = compute_.Shapes().GetLocalFunction(
      ShapesContext::LocalFunctionKey(domain, op_type, overload));
  const LightOpSchema *schema = nullptr;
  if (local_function == nullptr) {
    for (const LightOpSchema &candidate : DomainSchemas(op_type, normalised_domain)) {
      if (candidate.since_version() <= opset &&
          (schema == nullptr || candidate.since_version() > schema->since_version())) {
        schema = &candidate;
      }
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
    num_outputs = local_function == nullptr ? 1 : local_function->output().size();
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
  if (!overload.empty())
    node.set_overload(overload);
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
    attribute_source.ref_attribute().push_back(utils::CopyWithSharedTensorStorage(attribute));
  }
  const std::unordered_set<std::string> excluded_inherited_names(resolved_outputs.begin(),
                                                                 resolved_outputs.end());
  const utils::RepeatedProtoField<AttributeProto> normalized_attributes =
      ImportAttributes(attribute_source, excluded_inherited_names);
  for (const AttributeProto &attribute : normalized_attributes) {
    node.ref_attribute().push_back(utils::CopyWithSharedTensorStorage(attribute));
  }
  NodeProto materialized = utils::CopyWithSharedTensorStorage(node);
  MaterializeGraphReferences(materialized);
  for (const std::string &input : core::graph::CollectNodeInputs(materialized)) {
    SeedInheritedValue(input);
  }
  if (domain == "ai.onnx.ml" && op_type == "TreeEnsemble") {
    const auto error = ValidateTreeEnsembleAttributes(
        [&node](const char *attribute_name) { return FindAttribute(node, attribute_name); });
    if (!error.empty()) {
      throw BuilderError(error);
    }
  }

  // Nodes() is read-only, but legacy callers may still have mutated its
  // storage through a const_cast. Bring only those appended nodes into the
  // incremental metadata before processing this node.
  for (std::size_t missing_index = compute_.Size(); missing_index < nodes_.size();
       ++missing_index) {
    const NodeProto &missing = nodes_[missing_index];
    compute_.AppendNodeTags(nodes_, missing_index);
    std::vector<std::string> missing_references;
    CollectNodeReferences(missing, missing_references);
    compute_.AppendNodeReuse(nodes_, missing_index, compute_.Shapes(), missing_references);
    compute_.AppendNodeConstant(missing, missing_index);
    compute_.AppendNodePeakMemory(missing, missing_index, device_);
  }
  if (compute_.Size() != nodes_.size()) {
    throw BuilderError("GraphBuilder: incremental metadata is not aligned with the node list.");
  }
  compute_.RefreshIncrementalMemory(
      nodes_, compute_.Shapes(), device_,
      [this](const NodeProto &node) { return LayerBodyForMemory(node); });

  // Run incremental shape inference for the new node when a shape function is
  // registered for its operator; unregistered operators simply leave their
  // outputs without an inferred descriptor.
  if (ShapeFunctionAvailable(node)) {
    if (ShapeInputsAvailable(compute_.Shapes(), materialized) &&
        !HasUnboundAttributes(materialized)) {
      compute_.Shapes().set_current_node_index(static_cast<int64_t>(nodes_.size()));
      compute_.Shapes().ComputeShapeNode(materialized);
    }
  }

  NodeProto &stored = nodes_.add();
  stored = std::move(node);
  const int64_t index = static_cast<int64_t>(nodes_.size() - 1);
  for (const auto &attribute : stored.attribute()) {
    if (HasGraphReferenceSuffix(attribute.name().value()) &&
        attribute.type() == AttributeProto::AttributeType::STRING && attribute.has_s()) {
      const std::string attribute_name =
          attribute.name().value().substr(0, attribute.name().value().size() - 4);
      if (compute_.Shapes().HasSubgraphContext(index, attribute_name)) {
        NamedBuilderOrThrow(subgraphs_, attribute.s().value(), "subgraph")
            .CopyInferredValuesFrom(compute_.Shapes().GetSubgraphContext(index, attribute_name));
      }
    }
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
  std::vector<std::string> references;
  CollectNodeReferences(stored, references);
  compute_.AppendNodeReuse(nodes_, node_index, compute_.Shapes(), references);
  compute_.AppendNodeConstant(stored, node_index);
  compute_.AppendNodePeakMemory(stored, node_index, device_);
  RefreshAfterMutation();

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
      for (std::size_t i = 0; i < refs.size(); ++i) {
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
  for (const auto &initializer : extended_initializers_) {
    defined.insert(initializer.name().value());
  }
  for (const auto &initializer : graph_template_.sparse_initializer()) {
    defined.insert(initializer.values().name().value());
  }
  for (const NodeProto &node : nodes_) {
    for (std::size_t i = 0; i < node.output().size(); ++i) {
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
  NodeProto materialized = utils::CopyWithSharedTensorStorage(node);
  MaterializeGraphReferences(materialized);
  refs = core::graph::CollectNodeInputs(materialized);
}

std::size_t GraphBuilder::RemoveUnusedNodes() {
  const auto removed = RemoveUnusedNodesImpl(true);
  RebuildStructuredState();
  return removed;
}

std::size_t GraphBuilder::RemoveUnusedNodesImpl(bool recursive) {
  // Prune nested builders first: a leaner subgraph or local function may stop
  // referencing outer values, which can in turn make an owning node unused.
  std::size_t removed = 0;
  if (recursive) {
    for (const auto &function : local_functions_) {
      removed += function->RemoveUnusedNodes();
    }
    for (const auto &subgraph : subgraphs_) {
      removed += subgraph->RemoveUnusedNodes();
    }
  }

  const std::size_t num_nodes = nodes_.size();
  // Map each produced value name to the index of the node that produces it.
  std::unordered_map<std::string, std::size_t> producer;
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const NodeProto &node = nodes_[i];
    for (std::size_t j = 0; j < node.output().size(); ++j) {
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
  std::unordered_set<std::string> used;
  for (const ValueInfoProto &input : inputs_) {
    used.insert(input.name().value());
  }
  for (const ValueInfoProto &output : outputs_) {
    used.insert(output.name().value());
  }
  for (const NodeProto &node : nodes_) {
    std::vector<std::string> refs;
    CollectNodeReferences(node, refs);
    used.insert(refs.begin(), refs.end());
  }
  for (auto it = initializers_.begin(); it != initializers_.end();) {
    if (used.find(it->name().value()) == used.end()) {
      it = initializers_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = extended_initializers_.begin(); it != extended_initializers_.end();) {
    if (used.count(it->name().value()) == 0) {
      it = extended_initializers_.erase(it);
    } else {
      ++it;
    }
  }
  PruneValueInfos();
  return removed + local_removed;
}

void GraphBuilder::PruneValueInfos() {
  std::unordered_set<std::string> existing;
  for (const ValueInfoProto &input : inputs_) {
    existing.insert(input.name().value());
  }
  for (const ValueInfoProto &output : outputs_) {
    existing.insert(output.name().value());
  }
  for (const TensorProto &initializer : initializers_) {
    existing.insert(initializer.name().value());
  }
  for (const auto &initializer : extended_initializers_) {
    existing.insert(initializer.name().value());
  }
  for (const NodeProto &node : nodes_) {
    for (std::size_t i = 0; i < node.input().size(); ++i) {
      existing.insert(node.input(static_cast<std::size_t>(i)));
    }
    for (std::size_t i = 0; i < node.output().size(); ++i) {
      existing.insert(node.output(static_cast<std::size_t>(i)));
    }
  }
  for (auto it = value_infos_.begin(); it != value_infos_.end();) {
    if (existing.find(it->name().value()) == existing.end()) {
      it = value_infos_.erase(it);
    } else {
      ++it;
    }
  }
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
  for (std::size_t i = 0; i < node.output().size(); ++i) {
    if (!std::string(node.output(static_cast<std::size_t>(i))).empty() &&
        survivor_outputs[static_cast<std::size_t>(i)].empty()) {
      return false;
    }
  }
  return true;
}

} // namespace

std::size_t GraphBuilder::RemoveIdentityNodes() {
  const auto removed = RemoveIdentityNodesImpl(true, nullptr);
  RebuildStructuredState();
  return removed;
}

std::size_t GraphBuilder::RemoveIdentityNodesImpl(
    bool recursive, std::unordered_map<std::string, std::string> *applied_renames) {
  // Prune nested builders first so a collapsed identity inside a subgraph or
  // local function is handled in its own scope.
  std::size_t removed = 0;
  if (recursive) {
    for (const auto &function : local_functions_) {
      removed += function->RemoveIdentityNodes();
    }
    for (const auto &subgraph : subgraphs_) {
      removed += subgraph->RemoveIdentityNodes();
    }
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
  if (applied_renames != nullptr) {
    *applied_renames = rename;
  }

  return removed + local_removed;
}

std::size_t GraphBuilder::RemoveDuplicateNodes() {
  const auto removed = RemoveDuplicateNodesImpl(true, nullptr);
  RebuildStructuredState();
  return removed;
}

namespace {

bool SameNodeComputation(const NodeProto &left, const NodeProto &right) {
  if (left.op_type() != right.op_type() ||
      NormaliseDomain(left.domain()) != NormaliseDomain(right.domain()) ||
      left.overload().sv() != right.overload().sv() ||
      left.attribute().size() != right.attribute().size())
    return false;
  std::vector<bool> matched(right.attribute().size(), false);
  for (const auto &attribute : left.attribute()) {
    size_t index = 0;
    for (; index < right.attribute().size(); ++index)
      if (!matched[index] && attribute.Equals(right.attribute(index))) {
        matched[index] = true;
        break;
      }
    if (index == right.attribute().size())
      return false;
  }
  return true;
}

} // namespace

std::size_t GraphBuilder::RemoveDuplicateNodesImpl(
    bool recursive, std::unordered_map<std::string, std::string> *applied_renames) {
  // Prune nested builders first so each scope collapses its own duplicates.
  std::size_t removed = 0;
  if (recursive) {
    for (const auto &function : local_functions_) {
      removed += function->RemoveDuplicateNodes();
    }
    for (const auto &subgraph : subgraphs_) {
      removed += subgraph->RemoveDuplicateNodes();
    }
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

  // Hash collisions retain separate native-equality groups, each with its first survivor.
  // Insertion order is topological, so a duplicate is always seen after its survivor.
  struct Candidate {
    std::size_t position;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
  };
  std::unordered_map<std::string, std::vector<Candidate>> seen;
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
    for (std::size_t i = 0; i < node.input().size(); ++i) {
      resolved_inputs.push_back(resolve(std::string(node.input(static_cast<std::size_t>(i)))));
    }

    bool produces_output = false;
    for (std::size_t i = 0; i < node.output().size(); ++i) {
      if (output_names.find(std::string(node.output(static_cast<std::size_t>(i)))) !=
          output_names.end()) {
        produces_output = true;
        break;
      }
    }

    // Nodes referencing subgraphs carry per-node-unique ``*_ref`` attribute
    // names, so native equality keeps distinct control-flow nodes separate.
    const std::string signature = node.Signature(resolved_inputs);
    auto &candidates = seen[signature];
    const auto survivor =
        std::find_if(candidates.begin(), candidates.end(), [&](const auto &candidate) {
          return candidate.inputs == resolved_inputs &&
                 SameNodeComputation(node, kept[candidate.position]);
        });
    if (survivor != candidates.end() && !produces_output &&
        CanReuseOutputs(node, survivor->outputs)) {
      // Drop the duplicate and rewire each of its outputs onto the survivor.
      for (std::size_t i = 0; i < node.output().size(); ++i) {
        std::string out(node.output(static_cast<std::size_t>(i)));
        if (!out.empty()) {
          rename.emplace(std::move(out), survivor->outputs[static_cast<std::size_t>(i)]);
        }
      }
      ++local_removed;
      continue;
    }

    // Keep the node; register it as the survivor for its signature only when it
    // is the first one, so later duplicates collapse onto the earliest copy.
    if (survivor == candidates.end()) {
      std::vector<std::string> outputs;
      outputs.reserve(static_cast<std::size_t>(node.output().size()));
      for (std::size_t i = 0; i < node.output().size(); ++i) {
        outputs.push_back(std::string(node.output(static_cast<std::size_t>(i))));
      }
      candidates.push_back({kept.size(), std::move(resolved_inputs), std::move(outputs)});
    }
    kept.push_back(std::move(node));
  }
  nodes_ = std::move(kept);

  // Rewrite every consumer of a dropped node's output, descending into
  // subgraphs whose bodies capture values from this enclosing scope.
  RewriteInitializerReferences(rename);
  if (applied_renames != nullptr) {
    *applied_renames = rename;
  }
  return removed + local_removed;
}

namespace {

// Returns true when ``node`` is a default-domain Shape or Size node reading a
// single named input, i.e. a node that only inspects its input's metadata.
bool IsShapeOrSizeNode(const NodeProto &node) {
  const std::string domain = node.domain().empty() ? std::string() : node.domain().value();
  if (!domain.empty()) {
    return false;
  }
  const std::string &op_type = node.op_type().value();
  if (op_type != "Shape" && op_type != "Size") {
    return false;
  }
  return node.input().size() == 1 && !std::string(node.input(0)).empty();
}

} // namespace

std::size_t GraphBuilder::MoveShapeAndSizeNodes() { return MoveShapeAndSizeNodesImpl(true); }

std::size_t GraphBuilder::MoveShapeAndSizeNodesImpl(bool recursive) {
  // Reorder nested builders first so each scope hoists its own Shape/Size nodes.
  std::size_t moved = 0;
  if (recursive) {
    for (const auto &function : local_functions_) {
      moved += function->MoveShapeAndSizeNodes();
    }
    for (const auto &subgraph : subgraphs_) {
      moved += subgraph->MoveShapeAndSizeNodes();
    }
  }

  const std::size_t num_nodes = nodes_.size();
  if (num_nodes == 0) {
    return moved;
  }

  // Maps a value name to the index of the node producing it.
  std::unordered_map<std::string, std::size_t> producer_of;
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const NodeProto &node = nodes_[i];
    for (std::size_t j = 0; j < node.output().size(); ++j) {
      std::string out(node.output(static_cast<std::size_t>(j)));
      if (!out.empty()) {
        producer_of.emplace(std::move(out), i);
      }
    }
  }

  // For each Shape/Size node whose input has a producing node, record it as a
  // child to emit right after that producer, preserving their relative order.
  std::vector<std::vector<std::size_t>> children(num_nodes);
  std::vector<bool> is_relocatable(num_nodes, false);
  std::size_t relocatable = 0;
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const NodeProto &node = nodes_[i];
    if (!IsShapeOrSizeNode(node)) {
      continue;
    }
    auto it = producer_of.find(std::string(node.input(0)));
    // A producer always precedes its consumer in the insertion (topological)
    // order, so it < i; guard against unexpected forward references anyway.
    if (it == producer_of.end() || it->second >= i) {
      continue;
    }
    children[it->second].push_back(i);
    is_relocatable[i] = true;
    ++relocatable;
  }

  if (relocatable == 0) {
    return moved;
  }

  // Emit node indices in order; after emitting a node, emit its attached
  // Shape/Size children (which may themselves have attached children, e.g. a
  // Size reading a Shape output), so chains are relocated in a single pass.
  std::vector<std::size_t> order;
  order.reserve(num_nodes);
  std::vector<std::size_t> stack;
  for (std::size_t i = 0; i < num_nodes; ++i) {
    if (is_relocatable[i]) {
      continue;
    }
    stack.push_back(i);
    while (!stack.empty()) {
      const std::size_t index = stack.back();
      stack.pop_back();
      order.push_back(index);
      const std::vector<std::size_t> &kids = children[index];
      // Push children in reverse so they are emitted in their recorded order.
      for (std::size_t k = kids.size(); k > 0; --k) {
        stack.push_back(kids[k - 1]);
      }
    }
  }

  // Count only relocatable nodes whose position actually changed, so a second
  // pass over an already-tightened graph reports nothing moved.
  std::size_t local_moved = 0;
  for (std::size_t new_index = 0; new_index < order.size(); ++new_index) {
    if (is_relocatable[order[new_index]] && order[new_index] != new_index) {
      ++local_moved;
    }
  }
  if (local_moved == 0) {
    return moved;
  }

  utils::RepeatedProtoField<NodeProto> ordered;
  ordered.reserve(num_nodes);
  for (std::size_t index : order) {
    ordered.push_back(std::move(nodes_[index]));
  }
  nodes_ = std::move(ordered);
  std::vector<std::vector<std::string>> references(nodes_.size());
  for (std::size_t index = 0; index < nodes_.size(); ++index)
    CollectNodeReferences(nodes_[index], references[index]);
  compute_.UpdateNodes(nodes_, order, references, {}, device_);
  RefreshAfterMutation();
  InvalidateRewritings();

  return moved + local_moved;
}

GraphBuilder *GraphBuilder::FindCalledFunction(const std::vector<GraphBuilder *> &functions,
                                               const NodeProto &node) {
  const std::string node_domain = node.domain().empty() ? std::string() : node.domain().value();
  const std::string normalised = NormaliseDomain(node_domain);
  const std::string overload = node.overload().value();
  for (GraphBuilder *function : functions) {
    if (function->name() == node.op_type().value() &&
        NormaliseDomain(function->function_domain_) == normalised &&
        function->function_template_.overload().value() == overload) {
      return function;
    }
  }
  return nullptr;
}

GraphBuilder::FunctionIdentifier GraphBuilder::MakeFunctionIdentifier(const std::string &domain,
                                                                      const std::string &name,
                                                                      const std::string &overload) {
  return {NormaliseDomain(domain), name, overload};
}

utils::RepeatedProtoField<FunctionProto>
GraphBuilder::DependencyOrderFunctions(const utils::RepeatedProtoField<FunctionProto> &functions) {
  std::map<FunctionIdentifier, std::size_t> indices;
  for (std::size_t index = 0; index < functions.size(); ++index) {
    const auto &function = functions[index];
    if (!indices
             .emplace(
                 MakeFunctionIdentifier(function.domain(), function.name(), function.overload()),
                 index)
             .second) {
      throw BuilderError("GraphBuilder: duplicate function identity '" + function.name().value() +
                         "'.");
    }
  }
  const auto dependencies = [&](const FunctionProto &function) {
    std::set<FunctionIdentifier> result;
    const auto collect = [&](const auto &self, const NodeProto &node) -> void {
      const auto identity = MakeFunctionIdentifier(node.domain(), node.op_type(), node.overload());
      if (indices.count(identity) != 0)
        result.insert(identity);
      for (const auto &attribute : node.attribute()) {
        if (attribute.has_g())
          for (const auto &child : attribute.g().node())
            self(self, child);
        for (const auto &graph : attribute.graphs())
          for (const auto &child : graph.node())
            self(self, child);
      }
    };
    for (const auto &node : function.node())
      collect(collect, node);
    for (const auto &attribute : function.attribute_proto()) {
      if (attribute.has_g())
        for (const auto &node : attribute.g().node())
          collect(collect, node);
      for (const auto &graph : attribute.graphs())
        for (const auto &node : graph.node())
          collect(collect, node);
    }
    return result;
  };

  std::vector<std::set<FunctionIdentifier>> required(functions.size());
  for (std::size_t index = 0; index < functions.size(); ++index)
    required[index] = dependencies(functions[index]);
  std::vector<std::vector<std::size_t>> dependents(functions.size());
  std::vector<std::size_t> dependency_count(functions.size());
  for (std::size_t index = 0; index < functions.size(); ++index) {
    dependency_count[index] = required[index].size();
    for (const auto &callee : required[index])
      dependents[indices.at(callee)].push_back(index);
  }
  std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>> ready;
  for (std::size_t index = 0; index < functions.size(); ++index)
    if (dependency_count[index] == 0)
      ready.push(index);
  utils::RepeatedProtoField<FunctionProto> ordered;
  while (!ready.empty()) {
    const std::size_t index = ready.top();
    ready.pop();
    ordered.push_back(utils::CopyWithSharedTensorStorage(functions[index]));
    for (std::size_t caller : dependents[index])
      if (--dependency_count[caller] == 0)
        ready.push(caller);
  }
  if (ordered.size() != functions.size())
    throw BuilderError("GraphBuilder: function dependencies contain a cycle.");
  return ordered;
}

std::size_t GraphBuilder::CountFunctionCalls(const std::string &name, const std::string &domain,
                                             const std::string &overload) const {
  const std::string normalised = NormaliseDomain(domain);
  const std::function<std::size_t(const GraphProto &)> count_graph =
      [&](const GraphProto &graph) -> std::size_t {
    std::size_t count = 0;
    for (const auto &node : graph.node()) {
      const std::string node_domain = node.domain().empty() ? std::string() : node.domain().value();
      if (node.op_type().value() == name && NormaliseDomain(node_domain) == normalised &&
          node.overload().value() == overload) {
        ++count;
      }
      for (const auto &attribute : node.attribute()) {
        if (attribute.has_g()) {
          count += count_graph(attribute.g());
        }
        for (const auto &nested : attribute.graphs()) {
          count += count_graph(nested);
        }
      }
    }
    return count;
  };
  std::size_t count = 0;
  for (const NodeProto &node : nodes_) {
    const std::string node_domain = node.domain().empty() ? std::string() : node.domain().value();
    if (node.op_type().value() == name && NormaliseDomain(node_domain) == normalised &&
        node.overload().value() == overload) {
      ++count;
    }
  }
  for (const auto &subgraph : subgraphs_) {
    count += subgraph->CountFunctionCalls(name, domain, overload);
  }
  for (const auto &attribute : function_attribute_protos_) {
    if (attribute.has_g()) {
      count += count_graph(attribute.g());
    }
    for (const auto &graph : attribute.graphs()) {
      count += count_graph(graph);
    }
  }
  return count;
}

void GraphBuilder::AppendInlinedBody(GraphBuilder &function, const NodeProto &call,
                                     utils::RepeatedProtoField<NodeProto> &out) {
  AttributeBindings bindings;
  for (const auto &attribute : function.function_attribute_protos_) {
    bindings[attribute.name().value()] = {&attribute, false};
  }
  NodeProto materialized_call = utils::CopyWithSharedTensorStorage(call);
  MaterializeGraphReferences(materialized_call);
  for (const auto &attribute : materialized_call.attribute()) {
    bindings[attribute.name().value()] = {&attribute, true};
  }
  // Resolve the entire body before adding initializers, reserving names or importing subgraphs.
  struct MaterializedNode {
    NodeProto node;
  };
  std::vector<MaterializedNode> materialized_body;
  materialized_body.reserve(function.nodes_.size());
  for (const NodeProto &body : function.nodes_) {
    MaterializedNode materialized;
    materialized.node = utils::CopyWithSharedTensorStorage(body);
    function.MaterializeGraphReferences(materialized.node);
    NodeProto validated = utils::CopyWithSharedTensorStorage(materialized.node);
    ResolveFunctionAttributes(validated, bindings, nullptr, true);
    materialized_body.push_back(std::move(materialized));
  }

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
    TensorProto clone = utils::CopyWithSharedTensorStorage(initializer);
    clone.set_name(new_name);
    MakeInitializer(clone);
    rename.emplace(old_name, new_name);
  }
  for (const auto &initializer : function.extended_initializers_) {
    const std::string &old_name = initializer.name().value();
    if (rename.count(old_name) != 0) {
      throw BuilderError("GraphBuilder: cannot inline an extended function initializer that is "
                         "also a formal input or output.");
    }
    std::string new_name = function.name() + "_" + old_name;
    int suffix = 0;
    while (HasName(new_name)) {
      new_name = function.name() + "_" + old_name + "_" + std::to_string(suffix++);
    }
    if (initializer.has_encoded_value()) {
      EncodedValueProto clone = utils::CopyWithSharedTensorStorage(initializer.encoded_value());
      clone.set_name(new_name);
      MakeEncodedInitializer(clone);
    } else {
      PagedCacheProto clone = initializer.paged_cache();
      clone.set_name(new_name);
      MakePagedCacheInitializer(clone);
    }
    rename.emplace(old_name, new_name);
  }

  // Fresh names for every remaining body value (node outputs not yet mapped).
  for (const NodeProto &node : function.nodes_) {
    for (std::size_t j = 0; j < node.output().size(); ++j) {
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

  for (const MaterializedNode &materialized : materialized_body) {
    NodeProto resolved = utils::CopyWithSharedTensorStorage(materialized.node);
    ResolveFunctionAttributes(resolved, bindings, &rename, true);
    NodeProto node = utils::CopyWithSharedTensorStorage(resolved);
    node.ref_input().clear();
    node.ref_output().clear();
    node.ref_attribute().clear();
    for (std::size_t i = 0; i < resolved.input().size(); ++i) {
      node.add_input(remap(std::string(resolved.input(static_cast<std::size_t>(i)))));
    }
    for (std::size_t i = 0; i < resolved.output().size(); ++i) {
      node.add_output(remap(std::string(resolved.output(static_cast<std::size_t>(i)))));
    }
    for (const auto &attribute : resolved.attribute()) {
      node.ref_attribute().push_back(utils::CopyWithSharedTensorStorage(attribute));
    }
    const std::unordered_set<std::string> excluded_inherited_names(node.output().begin(),
                                                                   node.output().end());
    node.ref_attribute() = ImportAttributes(node, excluded_inherited_names);
    out.push_back(std::move(node));
  }
}

std::size_t GraphBuilder::InlineFunctionCalls(const std::vector<GraphBuilder *> &functions,
                                              std::set<FunctionIdentifier> &expanded_functions) {
  std::size_t inlined = 0;
  if (functions.empty()) {
    return inlined;
  }

  std::unordered_set<const GraphBuilder *> replaced_subgraphs;
  if (!nodes_.empty()) {
    // Rebuild the node list, expanding every call. Repeat to a fixed point: a
    // pasted body may itself call another local function.
    while (true) {
      std::vector<std::pair<std::size_t, GraphBuilder *>> calls;
      for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (GraphBuilder *function = FindCalledFunction(functions, nodes_[i])) {
          calls.emplace_back(i, function);
        }
      }
      if (calls.empty()) {
        break;
      }

      utils::RepeatedProtoField<NodeProto> kept;
      kept.reserve(nodes_.size());
      std::size_t next_call = 0;
      for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (next_call < calls.size() && calls[next_call].first == i) {
          GraphBuilder *function = calls[next_call++].second;
          const auto original_subgraphs = ReferencedSubgraphs(nodes_[i]);
          AppendInlinedBody(*function, nodes_[i], kept);
          replaced_subgraphs.insert(original_subgraphs.begin(), original_subgraphs.end());
          expanded_functions.insert(
              MakeFunctionIdentifier(function->function_domain_, function->name(),
                                     function->function_template_.overload().value()));
          ++inlined;
        } else {
          kept.push_back(std::move(nodes_[i]));
        }
      }
      nodes_ = std::move(kept);
    }
  }
  if (!replaced_subgraphs.empty()) {
    std::unordered_set<const GraphBuilder *> referenced;
    for (const auto &node : nodes_) {
      for (const GraphBuilder *subgraph : ReferencedSubgraphs(node)) {
        referenced.insert(subgraph);
      }
    }
    subgraphs_.erase(std::remove_if(subgraphs_.begin(), subgraphs_.end(),
                                    [&](const std::unique_ptr<GraphBuilder> &subgraph) {
                                      if (replaced_subgraphs.count(subgraph.get()) == 0 ||
                                          referenced.count(subgraph.get()) != 0) {
                                        return false;
                                      }
                                      names_.erase(subgraph->name());
                                      return true;
                                    }),
                     subgraphs_.end());
  }
  // Descend once after expanding this graph so the traversal sees both
  // retained pre-existing subgraphs and subgraphs cloned by the expansion above.
  for (const auto &subgraph : subgraphs_) {
    inlined += subgraph->InlineFunctionCalls(functions, expanded_functions);
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

  // Cleanup is limited to definitions that were actually expanded. This keeps
  // uncalled exported functions while covering calls introduced from graph-valued
  // function defaults during expansion.
  std::set<FunctionIdentifier> expanded_functions;
  const std::size_t inlined = InlineFunctionCalls(functions, expanded_functions);

  // Drop the definitions of functions that were called but no longer have any
  // caller. Removing one can drop the last reference to another (a function
  // called only from a now-removed function body), so iterate until stable.
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < local_functions_.size(); ++i) {
      GraphBuilder *function = local_functions_[i].get();
      const std::string overload = function->function_template_.overload().value();
      const FunctionIdentifier identifier =
          MakeFunctionIdentifier(function->function_domain_, function->name(), overload);
      if (expanded_functions.find(identifier) == expanded_functions.end()) {
        continue;
      }
      std::size_t callers =
          CountFunctionCalls(function->name(), function->function_domain_, overload);
      for (const auto &other : local_functions_) {
        if (other.get() != function) {
          callers +=
              other->CountFunctionCalls(function->name(), function->function_domain_, overload);
        }
      }
      if (callers == 0) {
        local_functions_.erase(local_functions_.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        break;
      }
    }
  }
  if (inlined != 0) {
    RebuildStructuredState();
  }
  return inlined;
}

std::size_t GraphBuilder::ConstantFold(const ConstantFoldingOptions &options) {
  const auto removed = ConstantFoldImpl(options, nullptr);
  RebuildStructuredState();
  return removed;
}

std::size_t
GraphBuilder::ConstantFoldNodes(const ConstantFoldingOptions &options,
                                const std::unordered_set<std::string> &included_outputs) {
  std::unordered_map<const NodeProto *, std::size_t> indices;
  std::vector<std::shared_ptr<NodeProto>> original_nodes;
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    indices.emplace(&nodes_[index], index);
    original_nodes.push_back(nodes_.shared_at(index));
  }
  const auto initializer_count = initializers_.size();
  const auto removed = ConstantFoldImpl(options, &included_outputs);
  if (removed != 0) {
    std::vector<std::size_t> old_indices;
    std::unordered_set<const NodeProto *> retained;
    for (const auto &node : nodes_) {
      old_indices.push_back(indices.at(&node));
      retained.insert(&node);
    }
    for (const auto &node : original_nodes)
      if (retained.count(node.get()) == 0)
        compute_.Shapes().EraseNodeConstraints(*node);
    std::unordered_set<std::string> affected_values;
    for (std::size_t index = initializer_count; index < initializers_.size(); ++index)
      affected_values.insert(initializers_[index].name().value());
    RefreshAfterRewriting(old_indices, std::move(affected_values));
  }
  return removed;
}

std::size_t
GraphBuilder::ConstantFoldImpl(const ConstantFoldingOptions &options,
                               const std::unordered_set<std::string> *included_outputs) {
  if (!options.enabled) {
    return 0;
  }

  // Fold nested builders first so a subgraph or local function shrinks before
  // this scope is folded, mirroring the other recursive passes.
  std::size_t removed = 0;
  if (included_outputs == nullptr) {
    for (const auto &function : local_functions_) {
      removed += function->ConstantFold(options);
    }
    for (const auto &subgraph : subgraphs_) {
      removed += subgraph->ConstantFold(options);
    }
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
  // Normalise every excluded domain once up front so the per-node check is a
  // handful of logarithmic set lookups instead of a linear scan: a node is
  // excluded when the exact pair, either wildcard, or the empty-empty pair is
  // present.
  std::set<std::pair<std::string, std::string>> excluded_ops;
  for (const auto &entry : options.excluded_ops) {
    excluded_ops.emplace(entry.first.empty() ? std::string() : NormaliseDomain(entry.first),
                         entry.second);
  }
  const auto is_excluded = [&](const std::string &normalised_domain, const std::string &op_type) {
    return excluded_ops.count({normalised_domain, op_type}) != 0 ||
           excluded_ops.count({normalised_domain, std::string()}) != 0 ||
           excluded_ops.count({std::string(), op_type}) != 0 ||
           excluded_ops.count({std::string(), std::string()}) != 0;
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
    if (candidate && compute_.Shapes().HasLocalFunction(
                         ShapesContext::LocalFunctionKey(domain, op_type, node.overload()))) {
      // Native-only folding must not evaluate a model-local override as the
      // primitive kernel; recipe replay resolves model functions first.
      candidate = false;
    }
    if (candidate && included_outputs != nullptr) {
      for (std::size_t i = 0; i < node.output().size(); ++i) {
        const std::string output(node.output(static_cast<std::size_t>(i)));
        if (!output.empty() && included_outputs->find(output) == included_outputs->end()) {
          candidate = false;
          break;
        }
      }
    }
    // Every non-empty input must already be available as a materialized
    // constant; otherwise the data needed to evaluate the node is missing (for
    // example a predecessor that was left unfolded).
    if (candidate) {
      for (std::size_t i = 0; i < node.input().size(); ++i) {
        const std::string input(node.input(static_cast<std::size_t>(i)));
        if (!input.empty() && const_tensors.find(input) == const_tensors.end()) {
          candidate = false;
          break;
        }
      }
    }
    if (!candidate) {
      kept.add();
      kept.get(kept.size() - 1) = nodes_.shared_at(idx);
      continue;
    }

    // Classify the node by the value tag of its outputs. A shape-carrying result
    // must be foldable; a weight (or untagged) result is folded best-effort.
    bool is_shape_result = false;
    for (std::size_t i = 0; i < node.output().size(); ++i) {
      const std::string output(node.output(static_cast<std::size_t>(i)));
      const auto tag_it = value_tags.find(output);
      if (tag_it != value_tags.end() && tag_it->second == "shape") {
        is_shape_result = true;
        break;
      }
    }

    // Shape results can be folded at any time; weight (or untagged) results are
    // only folded when the caller opts in, so they can be deferred to a final
    // pass.
    if (!is_shape_result && !options.fold_weights) {
      kept.add();
      kept.get(kept.size() - 1) = nodes_.shared_at(idx);
      continue;
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
      kept.add();
      kept.get(kept.size() - 1) = nodes_.shared_at(idx);
      continue;
    }

    // Evaluate the node in isolation: seed its inputs from the materialized
    // constants and run the registered kernel.
    const int opset = OpsetVersion(domain);
    core::runtime::RuntimeContext rt(core::runtime::KernelContext(core::runtime::DefaultOpset(
                                         opset > 0 ? static_cast<int64_t>(opset) : 0)),
                                     core::runtime::RuntimeContextOptions{.device = device_});
    for (std::size_t i = 0; i < node.input().size(); ++i) {
      const std::string input(node.input(static_cast<std::size_t>(i)));
      if (input.empty()) {
        continue;
      }
      rt.Put(input, core::runtime::TensorFromProto(*const_tensors.at(input)));
    }
    core::runtime::RunNode(node, rt);

    // Reject the fold when any output exceeds the size threshold; the node is
    // then kept untouched.
    bool within_threshold = true;
    if (options.max_element_count >= 0) {
      for (std::size_t i = 0; i < node.output().size(); ++i) {
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
      kept.add();
      kept.get(kept.size() - 1) = nodes_.shared_at(idx);
      continue;
    }

    // Materialize every output as a constant available to later folds.
    for (std::size_t i = 0; i < node.output().size(); ++i) {
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
  const auto removed = DeduplicateInitializers(InitializerContentIndex{}, true, nullptr);
  RebuildStructuredState();
  return removed;
}

std::size_t GraphBuilder::DeduplicateInitializers(
    const InitializerContentIndex &inherited, bool recursive,
    std::unordered_map<std::string, std::string> *applied_renames,
    const std::vector<const EncodedValueProto *> &encoded_inherited) {
  // Declared graph outputs must keep their own name; they are never dropped as
  // duplicates (but can still act as the survivor for a later duplicate).
  std::unordered_set<std::string> output_names;
  for (const ValueInfoProto &output : outputs_) {
    output_names.insert(output.name().value());
  }
  std::unordered_set<std::string> input_names;
  for (const ValueInfoProto &input : inputs_) {
    input_names.insert(input.name().value());
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
    if (input_names.count(initializer.name().value())) {
      kept.push_back(std::move(initializer));
      continue;
    }
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
  auto encoded_index = encoded_inherited;
  removed += DeduplicateEncodedInitializers(encoded_index, rename);

  RewriteInitializerReferences(rename);
  if (applied_renames != nullptr) {
    *applied_renames = rename;
  }

  // Subgraph bodies see the enclosing scope, so pass the augmented index down.
  // Local functions have an isolated scope and start from a fresh index.
  if (recursive) {
    for (const auto &subgraph : subgraphs_) {
      removed += subgraph->DeduplicateInitializers(index, true, nullptr, encoded_index);
    }
    for (const auto &function : local_functions_) {
      removed += function->DeduplicateInitializers(InitializerContentIndex{}, true, nullptr);
    }
  }
  return removed;
}

std::size_t
GraphBuilder::DeduplicateEncodedInitializers(std::vector<const EncodedValueProto *> &index,
                                             std::unordered_map<std::string, std::string> &rename) {
  std::unordered_set<std::string> input_names;
  std::unordered_set<std::string> output_names;
  for (const auto &input : inputs_) {
    input_names.insert(input.name().value());
  }
  for (const auto &output : outputs_) {
    output_names.insert(output.name().value());
  }
  std::size_t removed = 0;
  utils::RepeatedProtoField<ExtendedInitializerProto> kept;
  for (auto &initializer : extended_initializers_) {
    if (!initializer.has_encoded_value() || input_names.count(initializer.name().value())) {
      kept.push_back(std::move(initializer));
      continue;
    }
    const auto &value = initializer.encoded_value();
    const auto survivor = std::find_if(index.begin(), index.end(), [&](const auto *candidate) {
      return SameEncodedContent(*candidate, value);
    });
    if (!output_names.count(value.name().value()) && survivor != index.end()) {
      rename.emplace(value.name().value(), (*survivor)->name().value());
      ++removed;
    } else {
      kept.push_back(std::move(initializer));
      index.push_back(&kept.back().encoded_value());
    }
  }
  extended_initializers_ = std::move(kept);
  return removed;
}

void GraphBuilder::RewriteInitializerReferences(
    const std::unordered_map<std::string, std::string> &rename) {
  if (rename.empty()) {
    return;
  }
  for (NodeProto &node : nodes_) {
    for (std::size_t i = 0; i < node.input().size(); ++i) {
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
    std::unordered_set<std::string> captured;
    subgraph->CollectImplicitInputs(captured);
    std::unordered_map<std::string, std::string> captured_renames;
    for (const auto &entry : rename) {
      if (captured.find(entry.first) != captured.end()) {
        captured_renames.emplace(entry);
      }
    }
    subgraph->RewriteCapturedReferences(captured_renames);
  }
}

void GraphBuilder::RewriteCapturedReferences(
    const std::unordered_map<std::string, std::string> &rename) {
  if (rename.empty()) {
    return;
  }
  for (const auto &[source, target] : rename) {
    if (source != target && inherited_names_.erase(source) != 0) {
      inherited_names_.insert(target);
      compute_.Shapes().Erase(source);
      compute_.EraseValue(source);
    }
  }
  for (NodeProto &node : nodes_) {
    for (std::size_t i = 0; i < node.input().size(); ++i) {
      auto it = rename.find(node.input(static_cast<std::size_t>(i)));
      if (it != rename.end()) {
        node.mutable_input(static_cast<std::size_t>(i))->assign(it->second);
      }
    }
  }
  for (ValueInfoProto &output : outputs_) {
    auto it = rename.find(output.name().value());
    if (it != rename.end()) {
      output.set_name(it->second);
    }
  }
  for (const auto &subgraph : subgraphs_) {
    std::unordered_set<std::string> captured;
    subgraph->CollectImplicitInputs(captured);
    std::unordered_map<std::string, std::string> captured_renames;
    for (const auto &entry : rename) {
      if (captured.find(entry.first) != captured.end()) {
        captured_renames.emplace(entry);
      }
    }
    subgraph->RewriteCapturedReferences(captured_renames);
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
  auto child = std::make_unique<GraphBuilder>(name, schema_lookup_, opsets_);
  child->is_local_function_ = true;
  child->function_domain_ = domain;
  child->compute_.Shapes().SetStructTypes(Shapes().StructTypes());
  child->parent_ = this;
  GraphBuilder &ref = *child;
  local_functions_.push_back(std::move(child));
  RefreshLocalFunctionDefinition(ref);
  InvalidateRewritings();
  return ref;
}

GraphBuilder &GraphBuilder::MakeSubgraph(const std::string &name) {
  if (FindNamedBuilder(subgraphs_, name) != nullptr) {
    throw BuilderError("GraphBuilder: a subgraph named '" + name + "' already exists.");
  }
  ReserveName(name);
  auto child = std::make_unique<GraphBuilder>(name, schema_lookup_);
  child->parent_ = this;
  child->compute_.Shapes() = Shapes().MakeChildContext();
  for (const auto &entry : opsets_) {
    child->SetOpsetVersion(entry.first, entry.second);
  }
  child->inherited_names_ = inherited_names_;
  child->inherited_names_.insert(names_.begin(), names_.end());
  child->inherited_names_.erase(name);
  GraphBuilder &ref = *child;
  subgraphs_.push_back(std::move(child));
  return ref;
}

// ── Queries ────────────────────────────────────────────────────────────

bool GraphBuilder::HasShape(const std::string &name) const { return compute_.Shapes().Has(name); }

const SymTensor &GraphBuilder::GetShape(const std::string &name) const {
  return compute_.Shapes().Get(name);
}

void GraphBuilder::MakePersistentBinding(const PersistentBindingProto &binding) {
  if (parent_ != nullptr) {
    throw BuilderError("GraphBuilder: persistent bindings are supported only on the root graph.");
  }
  graph_template_.add_persistent_bindings(binding);
  InvalidateRewritings();
}

GraphProto GraphBuilder::BuildGraph() const {
  return BuildGraphImpl(/*validate_persistent_bindings=*/true);
}

void GraphBuilder::ValidateSubgraphOutput(const std::string &name) const {
  if (parent_ != nullptr && GraphDefinedNames(BuildGraphImpl(false)).count(name) == 0) {
    throw BuilderError("GraphBuilder: subgraph '" + name_ + "' output '" + name +
                       "' is not defined in this graph.");
  }
}

GraphProto GraphBuilder::BuildGraphImpl(bool validate_persistent_bindings,
                                        bool share_initializers) const {
  GraphProto graph = utils::CopyWithSharedTensorStorage(graph_template_);
  graph.set_name(name_);
  for (const ValueInfoProto &input : inputs_) {
    graph.add_input(input);
  }
  for (std::size_t i = 0; i < initializers_.size(); ++i) {
    if (share_initializers) {
      graph.ref_initializer().add();
      graph.ref_initializer().get(i) = initializers_.shared_at(i);
    } else {
      graph.ref_initializer().push_back(utils::CopyWithSharedTensorStorage(initializers_[i]));
    }
  }
  for (const auto &initializer : extended_initializers_) {
    graph.ref_extended_initializer().push_back(utils::CopyWithSharedTensorStorage(initializer));
  }
  for (const NodeProto &node : nodes_) {
    NodeProto materialized = utils::CopyWithSharedTensorStorage(node);
    MaterializeGraphReferences(materialized, share_initializers);
    graph.ref_node().push_back(std::move(materialized));
  }
  for (const ValueInfoProto &output : outputs_) {
    graph.add_output(output);
  }
  for (const ValueInfoProto &value_info : value_infos_) {
    graph.add_value_info(value_info);
  }
  if (parent_ != nullptr) {
    const auto locally_defined = GraphDefinedNames(graph);
    for (const ValueInfoProto &output : graph.output()) {
      if (locally_defined.count(output.name().value()) == 0) {
        throw BuilderError("GraphBuilder: subgraph '" + name_ + "' output '" +
                           output.name().value() + "' is not defined in this graph.");
      }
    }
  }
  if (validate_persistent_bindings && !graph.persistent_bindings().empty()) {
    // Supplies inferred IO types before validating the unfinalized graph view.
    for (auto &output : graph.ref_output()) {
      if (!output.has_type() && Shapes().HasType(output.name().value())) {
        *output.mutable_type() = Shapes().GetType(output.name().value());
      } else if (!output.has_type() && HasShape(output.name().value())) {
        SymTensorToValueInfo(GetShape(output.name().value()), output);
      }
    }
    StructTypeCatalogue catalogue;
    catalogue.Build(Shapes().StructTypes());
    VerifyPersistentBindings(&catalogue, graph, parent_ == nullptr);
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
    for (std::size_t i = 0; i < tensor.dims().size(); ++i) {
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
    for (std::size_t i = 0; i < node.input().size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << node.input(static_cast<std::size_t>(i));
    }
    os << ") -> ";
    for (std::size_t i = 0; i < node.output().size(); ++i) {
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

void GraphBuilder::SortNodesTopologically() {
  for (const auto &child : local_functions_) {
    child->SortNodesTopologically();
  }
  std::unordered_set<std::string> available = inherited_names_;
  for (const auto &input : inputs_) {
    available.insert(input.name().value());
  }
  for (const auto &initializer : initializers_) {
    available.insert(initializer.name().value());
  }
  for (const auto &initializer : extended_initializers_) {
    available.insert(initializer.name().value());
  }
  for (const auto &initializer : graph_template_.sparse_initializer()) {
    available.insert(initializer.values().name().value());
  }
  std::vector<std::vector<std::string>> references(nodes_.size());
  for (std::size_t i = 0; i < nodes_.size(); ++i)
    CollectNodeReferences(nodes_[i], references[i]);
  auto defined = available;
  bool reordered = false;
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (std::any_of(references[i].begin(), references[i].end(),
                    [&](const auto &name) { return defined.count(name) == 0; })) {
      reordered = true;
      break;
    }
    for (const auto &output : nodes_[i].output()) {
      if (!output.empty() && !defined.insert(output.value()).second) {
        throw BuilderError("GraphBuilder: multiple definitions of '" + output.value() + "'.");
      }
    }
  }
  if (reordered) {
    const auto order = TopologicalNodeOrder(nodes_, available, references);
    utils::RepeatedProtoField<NodeProto> sorted;
    sorted.reserve(nodes_.size());
    for (std::size_t i : order) {
      sorted.push_back(std::move(nodes_[i]));
    }
    nodes_ = std::move(sorted);
  }
  for (const NodeProto &node : nodes_) {
    for (GraphBuilder *child : ReferencedSubgraphs(node)) {
      child->inherited_names_ = available;
      child->SortNodesTopologically();
    }
    for (std::size_t i = 0; i < node.output().size(); ++i) {
      available.insert(node.output(i));
    }
  }
  if (reordered) {
    RebuildMetadata();
  }
}

std::vector<std::size_t>
GraphBuilder::InsertNodesInDependencyOrder(const utils::RepeatedProtoField<NodeProto> &nodes,
                                           const std::unordered_set<std::string> &available,
                                           const std::vector<std::vector<std::string>> &references,
                                           const std::vector<bool> &added) {
  const std::size_t count = nodes.size();
  auto defined = available;
  bool needs_insertion = false;
  for (std::size_t index = 0; index < count; ++index) {
    if (std::any_of(references[index].begin(), references[index].end(),
                    [&](const auto &name) { return defined.count(name) == 0; })) {
      needs_insertion = true;
      break;
    }
    for (const auto &output : nodes[index].output()) {
      if (!output.empty() && !defined.insert(output.value()).second) {
        throw BuilderError("GraphBuilder: multiple definitions of '" + output.value() + "'.");
      }
    }
  }
  if (!needs_insertion) {
    return {};
  }
  std::vector<std::size_t> order(count), positions(count);
  std::iota(order.begin(), order.end(), 0);
  std::iota(positions.begin(), positions.end(), 0);
  std::unordered_map<std::string, std::size_t> producers;
  for (std::size_t index = 0; index < count; ++index) {
    for (const auto &output : nodes[index].output()) {
      if (!output.empty() && (available.count(output.value()) != 0 ||
                              !producers.emplace(output.value(), index).second)) {
        throw BuilderError("GraphBuilder: multiple definitions of '" + output.value() + "'.");
      }
    }
  }
  std::vector<std::vector<std::size_t>> dependencies(count), consumers(count);
  for (std::size_t index = 0; index < count; ++index) {
    for (const auto &input : references[index]) {
      const auto producer = producers.find(input);
      if (producer != producers.end()) {
        dependencies[index].push_back(producer->second);
        consumers[producer->second].push_back(index);
      } else if (available.count(input) == 0) {
        throw BuilderError("GraphBuilder: node '" + nodes[index].op_type().value() +
                           "' reads the unknown value '" + input + "'.");
      }
    }
  }
  enum class VisitState { Unvisited, Active, Processed };
  std::vector<VisitState> moving(count, VisitState::Unvisited);
  // Inserts new producers before existing consumers whenever their inputs
  // permit it, rather than moving those consumers past unrelated nodes.
  for (std::size_t index = 0; index < count; ++index) {
    if (!added[index]) {
      continue;
    }
    std::size_t first = positions[index];
    for (const auto consumer : consumers[index]) {
      first = std::min(first, positions[consumer]);
    }
    if (first == positions[index] ||
        std::any_of(dependencies[index].begin(), dependencies[index].end(),
                    [&](const auto producer) { return positions[producer] >= first; })) {
      continue;
    }
    const auto last = positions[index];
    std::move_backward(order.begin() + first, order.begin() + last, order.begin() + last + 1);
    order[first] = index;
    for (std::size_t position = first; position <= last; ++position) {
      positions[order[position]] = position;
    }
  }
  for (std::size_t position = count; position-- > 0;) {
    const auto consumer = order[position];
    std::size_t last = position;
    for (const auto producer : dependencies[consumer]) {
      if (producer == consumer) {
        throw BuilderError("GraphBuilder: cyclic node dependencies.");
      }
      last = std::max(last, positions[producer]);
    }
    if (last == position) {
      continue;
    }
    // Delays only the blocked consumer and its affected consumers. Independent
    // nodes retain their relative order, even inside the insertion interval.
    // Active visits detect internal cycles; processed visits allow shared consumers.
    std::vector<std::pair<std::size_t, std::size_t>> pending{{consumer, 0}};
    std::size_t moving_count = 1;
    moving[consumer] = VisitState::Active;
    while (!pending.empty()) {
      auto &[current, next] = pending.back();
      if (next == consumers[current].size()) {
        moving[current] = VisitState::Processed;
        pending.pop_back();
        continue;
      }
      const auto dependent = consumers[current][next++];
      if (positions[dependent] < position || positions[dependent] > last) {
        continue;
      }
      if (moving[dependent] == VisitState::Active) {
        throw BuilderError("GraphBuilder: cyclic node dependencies.");
      }
      if (moving[dependent] == VisitState::Unvisited) {
        moving[dependent] = VisitState::Active;
        ++moving_count;
        pending.emplace_back(dependent, 0);
      }
    }
    if (moving[order[last]] != VisitState::Unvisited) {
      throw BuilderError("GraphBuilder: cyclic node dependencies.");
    }
    std::vector<std::size_t> delayed;
    delayed.reserve(moving_count);
    std::size_t destination = position;
    for (std::size_t source = position; source <= last; ++source) {
      const auto index = order[source];
      if (moving[index] != VisitState::Unvisited) {
        delayed.push_back(index);
      } else {
        order[destination++] = index;
      }
    }
    for (const auto index : delayed) {
      order[destination++] = index;
      moving[index] = VisitState::Unvisited;
    }
    for (std::size_t index = position; index <= last; ++index) {
      positions[order[index]] = index;
    }
  }
  return order;
}

std::vector<std::size_t>
GraphBuilder::TopologicalNodeOrder(const utils::RepeatedProtoField<NodeProto> &nodes,
                                   const std::unordered_set<std::string> &available,
                                   const std::vector<std::vector<std::string>> &references) {
  const std::size_t count = nodes.size();
  std::unordered_map<std::string, std::size_t> producers;
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = 0; j < nodes[i].output().size(); ++j) {
      const std::string output = nodes[i].output(j);
      if (!output.empty() &&
          (available.count(output) != 0 || !producers.emplace(output, i).second)) {
        throw BuilderError("GraphBuilder: multiple definitions of '" + output + "'.");
      }
    }
  }
  std::vector<std::vector<std::size_t>> consumers(count);
  std::vector<std::size_t> pending(count, 0);
  std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<std::size_t>> ready;
  for (std::size_t i = 0; i < count; ++i) {
    std::unordered_set<std::size_t> dependencies;
    for (const std::string &reference : references[i]) {
      const auto producer = producers.find(reference);
      if (producer != producers.end()) {
        dependencies.insert(producer->second);
      } else if (available.count(reference) == 0) {
        throw BuilderError("GraphBuilder: node '" + nodes[i].op_type().value() +
                           "' reads the unknown value '" + reference + "'.");
      }
    }
    pending[i] = dependencies.size();
    for (std::size_t dependency : dependencies) {
      consumers[dependency].push_back(i);
    }
    if (pending[i] == 0) {
      ready.push(i);
    }
  }
  std::vector<std::size_t> order;
  order.reserve(count);
  while (!ready.empty()) {
    const std::size_t i = ready.top();
    ready.pop();
    order.push_back(i);
    for (std::size_t consumer : consumers[i]) {
      if (--pending[consumer] == 0) {
        ready.push(consumer);
      }
    }
  }
  if (order.size() != count) {
    throw BuilderError("GraphBuilder: cyclic node dependencies prevent topological ordering.");
  }
  return order;
}

template <typename Proto> void GraphBuilder::Finalize(Proto &graph) {
  // Rewrites can carry old annotations onto new nodes. In particular, an
  // empty release list must clear, rather than retain, the old schedule.
  const std::unordered_set<std::string> computed_keys = {
      core::compute::kInPlaceReuseMetadataKey,   core::compute::kReleaseAfterMetadataKey,
      core::compute::kNotUsedAfterMetadataKey,   core::compute::kReleaseAfterShapeTagMetadataKey,
      core::compute::kNodePeakMemoryMetadataKey, core::compute::kConstantMetadataKey,
      core::compute::kNodeTagMetadataKey,        core::compute::kValueTagMetadataKey};
  for (std::size_t i = 0; i < graph.node().size(); ++i) {
    NodeProto &node = graph.ref_node()[i];
    auto &metadata = node.ref_metadata_props();
    for (auto it = metadata.begin(); it != metadata.end();) {
      if (computed_keys.count(it->key().value()) != 0) {
        it = metadata.erase(it);
      } else {
        ++it;
      }
    }
    for (const auto &ref : nodes_[i].attribute()) {
      if (!HasGraphReferenceSuffix(ref.name().value())) {
        continue;
      }
      const std::string name = ref.name().value().substr(0, ref.name().value().size() - 4);
      for (auto &attribute : node.ref_attribute()) {
        if (attribute.name().value() != name) {
          continue;
        }
        if (ref.type() == AttributeProto::AttributeType::STRING && attribute.has_g()) {
          NamedBuilderOrThrow(subgraphs_, ref.s().value(), "subgraph")
              .Finalize(*attribute.mutable_g());
        } else if (ref.type() == AttributeProto::AttributeType::STRINGS) {
          for (std::size_t j = 0; j < ref.strings().size(); ++j) {
            NamedBuilderOrThrow(subgraphs_, ref.strings()[j], "subgraph")
                .Finalize(attribute.ref_graphs()[j]);
          }
        }
      }
    }
  }
  EXT_ENFORCE_INVALID(compute_.Size() == graph.node().size(),
                      "GraphBuilder::Finalize: metadata is not aligned with the graph.");
  EXT_ENFORCE_INVALID(
      compute_.PeakMemory().size() == graph.node().size(),
      "GraphBuilder::Finalize: peak-memory metadata is not aligned with the graph.");
  // Writes inferred shapes (value_info), in-place / release-after / shape-tag
  // metadata and per-node peak memory.
  if constexpr (std::is_same_v<Proto, GraphProto>) {
    compute_.WriteToGraph(graph);
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      if (inputs_[i].has_type() && HasStructuredType(inputs_[i].type())) {
        graph.ref_input()[i].ref_type() = inputs_[i].type();
      }
    }
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      if (outputs_[i].has_type() && HasStructuredType(outputs_[i].type())) {
        graph.ref_output()[i].ref_type() = outputs_[i].type();
      }
    }
  } else {
    compute_.WriteToFunction(graph);
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      if (inputs_[i].has_type()) {
        graph.ref_value_info()[i].ref_type() = inputs_[i].type();
      } else {
        graph.ref_value_info()[i].clear_type();
      }
    }
    for (const auto &output : outputs_) {
      if (!output.has_type() || !HasStructuredType(output.type())) {
        continue;
      }
      for (auto &value : graph.ref_value_info()) {
        if (value.name() == output.name()) {
          value.ref_type() = output.type();
        }
      }
    }
  }
  std::unordered_set<std::string> live_values;
  if constexpr (std::is_same_v<Proto, FunctionProto>) {
    PrependInitializerConstants(graph);
    for (const auto &input : graph.input()) {
      live_values.insert(input);
    }
    for (const auto &output : graph.output()) {
      live_values.insert(output);
    }
  } else {
    for (const auto &initializer : graph.extended_initializer()) {
      live_values.insert(initializer.name().value());
    }
  }
  for (const auto &node : graph.node()) {
    for (std::size_t i = 0; i < node.output().size(); ++i) {
      live_values.insert(node.output(i));
    }
  }
  for (auto it = graph.ref_value_info().begin(); it != graph.ref_value_info().end();) {
    if (live_values.count(it->name().value()) == 0) {
      it = graph.ref_value_info().erase(it);
    } else {
      ++it;
    }
  }
  // Additionally records the per-node and per-value tags.
  core::compute::WriteValueAndNodeTagsToMetadata(graph);
  // Records constant-value / constant-node information.
  core::compute::WriteConstantInfoToMetadata(graph);
}

GraphProto GraphBuilder::ToGraph() {
  SortNodesTopologically();
  // Hoist Shape/Size nodes next to their producers before exporting so the
  // finalisation analyses (in-place reuse, peak memory) see the tighter order.
  MoveShapeAndSizeNodes();
  if (parent_ == nullptr) {
    RefreshLocalFunctions();
  }
  GraphProto graph = BuildGraph();
  Finalize(graph);
  if (!graph.persistent_bindings().empty()) {
    StructTypeCatalogue catalogue;
    catalogue.Build(Shapes().StructTypes());
    VerifyPersistentBindings(&catalogue, graph, parent_ == nullptr);
  }
  return graph;
}

ModelProto GraphBuilder::ToModel(int64_t ir_version) {
  GraphProto graph = ToGraph();
  ModelProto model = model_template_;
  model.set_ir_version(
      ir_version > 0
          ? ir_version
          : (model.has_ir_version() ? model.ir_version() : static_cast<int64_t>(IR_VERSION)));
  if (!model.has_producer_name()) {
    model.set_producer_name("onnx-light");
  }
  model.ref_struct_types() = Shapes().StructTypes();
  for (const auto &entry : opsets_) {
    model.add_opset(entry.first, entry.second);
  }
  *model.mutable_graph() = std::move(graph);
  std::set<FunctionIdentifier> exported;
  const auto export_functions = [&](const auto &self, GraphBuilder &scope) -> void {
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph);
    for (const auto &function : scope.local_functions_) {
      self(self, *function);
      FunctionProto definition = function->ExportFunction(function->function_domain_, true);
      const FunctionIdentifier identity =
          MakeFunctionIdentifier(definition.domain(), definition.name(), definition.overload());
      if (!exported.insert(identity).second) {
        const auto previous = std::find_if(
            model.functions().begin(), model.functions().end(), [&](const auto &candidate) {
              return MakeFunctionIdentifier(candidate.domain(), candidate.name(),
                                            candidate.overload()) == identity;
            });
        if (previous == model.functions().end() || !previous->Equals(definition)) {
          throw BuilderError("GraphBuilder: conflicting nested local function definition '" +
                             definition.name().value() + "'.");
        }
        continue;
      }
      model.ref_functions().push_back(std::move(definition));
    }
  };
  export_functions(export_functions, *this);
  model.ref_functions() = DependencyOrderFunctions(model.functions());
  model.ref_rewritings() = utils::CopyWithSharedTensorStorage(rewritings_);
  return model;
}

ModelProto GraphBuilder::ToStandardModel(int64_t ir_version) {
  if (!rewritings_.empty()) {
    throw BuilderError("GraphBuilder: standard ONNX cannot represent named rewriting variants.");
  }
  if (!Shapes().StructTypes().empty()) {
    throw BuilderError("GraphBuilder: standard ONNX cannot represent a structured type catalogue.");
  }
  RequireStandardGraph(BuildGraph());
  const auto validate_functions = [&](const auto &self, const GraphBuilder &scope) -> void {
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph);
    for (const auto &function : scope.local_functions_) {
      const auto body = function->BuildFunction(function->function_domain_);
      RequireStandardBody(body);
      for (const auto &attribute : body.attribute_proto())
        RequireStandardAttribute(attribute);
      self(self, *function);
    }
  };
  validate_functions(validate_functions, *this);
  return ToModel(ir_version);
}

FunctionProto GraphBuilder::ToFunction(const std::string &domain) {
  return ExportFunction(domain, false);
}

FunctionProto GraphBuilder::ExportFunction(const std::string &domain, bool model_scoped) {
  if (!model_scoped && !Shapes().StructTypes().empty()) {
    throw BuilderError("GraphBuilder: a FunctionProto cannot carry a structured type catalogue.");
  }
  if (!extended_initializers_.empty()) {
    throw BuilderError(
        "GraphBuilder: a FunctionProto cannot carry extended initializers; remove them or "
        "produce a model / graph instead.");
  }
  ValidateInitializerConstants();
  SortNodesTopologically();
  MoveShapeAndSizeNodes();
  if (parent_ == nullptr) {
    RefreshLocalFunctions();
  }
  FunctionProto function = BuildFunction(domain, /*include_initializer_constants=*/false);
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    ValueInfoProto &input = function.ref_value_info()[i];
    auto tag = std::find_if(
        input.ref_metadata_props().begin(), input.ref_metadata_props().end(),
        [](const auto &entry) { return entry.key() == core::compute::kValueTagMetadataKey; });
    if (tag == input.ref_metadata_props().end()) {
      auto *entry = input.add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    } else if (tag->value().empty()) {
      tag->set_value("weight");
    }
  }
  Finalize(function);
  return function;
}

void GraphBuilder::ValidateInitializerConstants() const {
  if (initializers_.empty())
    return;
  const int opset = OpsetVersion("");
  if (opset <= 0)
    throw BuilderError(
        "GraphBuilder: exporting initializer constants requires a default-domain opset.");
  const auto schemas = schema_lookup_ ? schema_lookup_("Constant") : std::vector<LightOpSchema>{};
  const LightOpSchema *schema = nullptr;
  for (const auto &candidate : schemas) {
    if (NormaliseDomain(candidate.domain()) == NormaliseDomain("") &&
        candidate.since_version() <= opset &&
        (schema == nullptr || candidate.since_version() > schema->since_version())) {
      schema = &candidate;
    }
  }
  for (const auto &initializer : initializers_) {
    const auto &name = initializer.name().value();
    if (std::any_of(inputs_.begin(), inputs_.end(),
                    [&](const auto &input) { return input.name() == name; }))
      throw BuilderError("GraphBuilder: a FunctionProto cannot preserve overridable initializer '" +
                         name + "'.");
    if (schema != nullptr) {
      const auto &constraints = schema->type_constraints();
      const auto constraint =
          std::find_if(constraints.begin(), constraints.end(), [&](const auto &entry) {
            return !schema->outputs().empty() &&
                   entry.type_param_str == schema->outputs().front().type;
          });
      const auto type = core::symbolic::DataTypeToTensorType(initializer.data_type());
      if (constraint == constraints.end() ||
          std::find(constraint->allowed_type_strs.begin(), constraint->allowed_type_strs.end(),
                    type) == constraint->allowed_type_strs.end()) {
        throw BuilderError("GraphBuilder: initializer '" + name + "' of type " +
                           TensorProto::DataType_Name(initializer.data_type()) +
                           " cannot be converted to Constant at ONNX opset " +
                           std::to_string(opset) + ".");
      }
    }
  }
}

void GraphBuilder::PrependInitializerConstants(FunctionProto &function) const {
  ValidateInitializerConstants();
  if (initializers_.empty())
    return;
  utils::RepeatedProtoField<NodeProto> nodes;
  nodes.reserve(initializers_.size() + function.node().size());
  for (const auto &initializer : initializers_) {
    auto &node = nodes.add();
    node.set_op_type("Constant");
    node.add_output(initializer.name().value());
    auto *attribute = node.add_attribute();
    attribute->set_name("value");
    attribute->set_type(AttributeProto::TENSOR);
    *attribute->mutable_t() = utils::CopyWithSharedTensorStorage(initializer);
  }
  for (auto &node : function.ref_node())
    nodes.push_back(std::move(node));
  function.ref_node() = std::move(nodes);
}

FunctionProto GraphBuilder::BuildFunction(const std::string &domain,
                                          bool include_initializer_constants) const {
  if (!graph_template_.persistent_bindings().empty()) {
    throw BuilderError("GraphBuilder: a FunctionProto cannot carry persistent bindings.");
  }
  FunctionProto function = utils::CopyWithSharedTensorStorage(function_template_);
  function.set_name(name_);
  function.set_domain(domain);
  for (const auto &attribute : function_attributes_) {
    function.add_attribute(attribute);
  }
  function.ref_attribute_proto() = utils::CopyWithSharedTensorStorage(function_attribute_protos_);
  function.ref_metadata_props() = metadata_;
  if (doc_string_) {
    function.set_doc_string(*doc_string_);
  }
  for (const ValueInfoProto &input : inputs_) {
    function.add_input(input.name().value());
    function.add_value_info(input);
  }
  for (const ValueInfoProto &output : outputs_) {
    function.add_output(output.name().value());
    const bool is_input =
        std::any_of(inputs_.begin(), inputs_.end(),
                    [&](const ValueInfoProto &input) { return input.name() == output.name(); });
    if (!is_input) {
      function.add_value_info(output);
    }
  }
  for (const ValueInfoProto &value_info : value_infos_) {
    function.add_value_info(value_info);
  }
  for (const auto &entry : opsets_) {
    function.add_opset(entry.first, entry.second);
  }
  for (const NodeProto &node : nodes_) {
    NodeProto materialized = utils::CopyWithSharedTensorStorage(node);
    MaterializeGraphReferences(materialized);
    function.ref_node().push_back(std::move(materialized));
  }
  if (include_initializer_constants)
    PrependInitializerConstants(function);
  return function;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder

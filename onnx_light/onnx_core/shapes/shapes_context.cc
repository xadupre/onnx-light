// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"

#include <string>
#include <utility>

#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

bool HasStructuredType(const TypeProto &type) {
  if (type.has_struct_type()) {
    return true;
  }
  if (type.has_sequence_type()) {
    return HasStructuredType(type.ref_sequence_type().ref_elem_type());
  }
  if (type.has_optional_type()) {
    return HasStructuredType(type.ref_optional_type().ref_elem_type());
  }
  if (type.has_map_type()) {
    return HasStructuredType(type.ref_map_type().ref_value_type());
  }
  return false;
}

namespace {

void CheckCatalogueCompatibility(const ShapesContext &left, const ShapesContext &right,
                                 const TypeProto &type, std::unordered_set<uint64_t> &visited) {
  if (type.has_sequence_type()) {
    CheckCatalogueCompatibility(left, right, type.ref_sequence_type().ref_elem_type(), visited);
  } else if (type.has_optional_type()) {
    CheckCatalogueCompatibility(left, right, type.ref_optional_type().ref_elem_type(), visited);
  } else if (type.has_map_type()) {
    CheckCatalogueCompatibility(left, right, type.ref_map_type().ref_value_type(), visited);
  } else if (type.has_struct_type()) {
    const StructTypeProto &structure = type.ref_struct_type();
    if (structure.has_type_ref() && !visited.insert(structure.ref_type_ref()).second) {
      return;
    }
    const auto &left_type = left.ResolveStructType(structure);
    const auto &right_type = right.ResolveStructType(structure);
    EXT_ENFORCE_INVALID(left_type.SerializeAsString() == right_type.SerializeAsString(),
                        "Structured values reference incompatible model declarations.");
    if (left_type.has_array()) {
      CheckCatalogueCompatibility(left, right, left_type.ref_array().ref_element_type(), visited);
    } else if (left_type.has_structure()) {
      for (const auto &field : left_type.ref_structure().ref_field()) {
        if (field.has_type()) {
          CheckCatalogueCompatibility(left, right, field.ref_type(), visited);
        }
      }
    }
  }
}

void CheckCatalogueCompatibility(const ShapesContext &left, const ShapesContext &right,
                                 const TypeProto &type) {
  std::unordered_set<uint64_t> visited;
  CheckCatalogueCompatibility(left, right, type, visited);
}

} // namespace

void ShapesContext::SetStructTypes(const utils::RepeatedProtoField<StructTypeProto> &types) {
  auto model = std::make_shared<ModelProto>();
  model->ref_struct_types() = types;
  StructTypeCatalogue catalogue;
  catalogue.Build(*model);
  for (const auto &entry : types_) {
    catalogue.ValidateType(entry.second);
  }
  for (const auto &entry : encoded_values_) {
    catalogue.ValidateEncodedValue(entry.second);
  }
  struct_types_model_ = std::move(model);
}

const utils::RepeatedProtoField<StructTypeProto> &ShapesContext::StructTypes() const {
  static const utils::RepeatedProtoField<StructTypeProto> empty;
  return struct_types_model_ ? struct_types_model_->ref_struct_types() : empty;
}

const StructTypeProto &ShapesContext::ResolveStructType(const StructTypeProto &type) const {
  StructTypeCatalogue catalogue;
  if (struct_types_model_) {
    catalogue.Build(*struct_types_model_);
  }
  return catalogue.Resolve(type);
}

void ShapesContext::SetType(const std::string &name, const TypeProto &type) {
  StructTypeCatalogue catalogue;
  if (struct_types_model_) {
    catalogue.Build(*struct_types_model_);
  }
  catalogue.ValidateType(type);
  TypeProto owned = type;
  const TypeProto *descriptor_type = &owned;
  while (descriptor_type->has_optional_type()) {
    descriptor_type = &descriptor_type->ref_optional_type().ref_elem_type();
  }
  ValueInfoProto vi;
  vi.set_name(name);
  vi.ref_type() = *descriptor_type;
  SymTensor tensor;
  const bool is_tensor = descriptor_type->has_tensor_type() &&
                         descriptor_type->ref_tensor_type().has_shape() &&
                         SymTensorFromValueInfo(vi, tensor);
  tensors_.erase(name);
  sequences_.erase(name);
  encoded_values_.erase(name);
  if (is_tensor) {
    Set(name, std::move(tensor));
  } else if (descriptor_type->has_sequence_type() &&
             descriptor_type->ref_sequence_type().ref_elem_type().has_tensor_type()) {
    const auto &element = descriptor_type->ref_sequence_type().ref_elem_type().ref_tensor_type();
    SetSequence(name,
                SymSequence(DataTypeToTensorType(element.elem_type()), SymDim(name + "_length")));
  } else if (descriptor_type->has_map_type() &&
             descriptor_type->ref_map_type().ref_value_type().has_tensor_type()) {
    const auto &element = descriptor_type->ref_map_type().ref_value_type().ref_tensor_type();
    Set(name, SymTensor(nullptr, DataTypeToTensorType(element.elem_type()), SymShape{}));
  }
  types_[name] = std::move(owned);
}

void ShapesContext::SetEncodedValue(const std::string &name, const EncodedValueProto &value) {
  StructTypeCatalogue catalogue;
  if (struct_types_model_) {
    catalogue.Build(*struct_types_model_);
  }
  EncodedValueProto owned = value;
  owned.set_name(name);
  catalogue.ValidateEncodedValue(owned);
  TypeProto type;
  if (owned.has_struct_type()) {
    type.ref_struct_type() = owned.ref_struct_type();
  } else {
    type = owned.ref_logical_type();
  }
  SymTensor tensor;
  bool has_logical_tensor = false;
  if (owned.has_logical_type() && owned.ref_logical_type().has_tensor_type() &&
      owned.ref_logical_type().ref_tensor_type().has_shape()) {
    ValueInfoProto vi;
    vi.ref_type() = owned.ref_logical_type();
    has_logical_tensor = SymTensorFromValueInfo(vi, tensor);
  }
  SetType(name, type);
  if (has_logical_tensor) {
    Set(name, std::move(tensor));
    types_[name] = std::move(type);
  }
  encoded_values_[name] = std::move(owned);
}

EncodedValueLayout ShapesContext::GetEncodedLayout(const std::string &name) const {
  StructTypeCatalogue catalogue;
  if (struct_types_model_) {
    catalogue.Build(*struct_types_model_);
  }
  return catalogue.ValidateEncodedValue(GetEncodedValue(name));
}

void ShapesContext::CopyValueFrom(const std::string &name, const ShapesContext &source,
                                  const std::string &source_name) {
  if (source.HasType(source_name) && HasStructuredType(source.GetType(source_name))) {
    CheckCatalogueCompatibility(*this, source, source.GetType(source_name));
  }
  if (source.HasEncodedValue(source_name)) {
    SetEncodedValue(name, source.GetEncodedValue(source_name));
  } else if (source.HasType(source_name)) {
    if (source.HasSequence(source_name)) {
      SymSequence sequence(source.GetSequence(source_name));
      SetType(name, source.GetType(source_name));
      sequences_[name] = std::move(sequence);
    } else if (source.Has(source_name)) {
      SymTensor tensor(source.Get(source_name));
      SetType(name, source.GetType(source_name));
      tensors_[name] = std::move(tensor);
    } else {
      SetType(name, source.GetType(source_name));
    }
  } else if (source.Has(source_name)) {
    Set(name, SymTensor(source.Get(source_name)));
  } else if (source.HasSequence(source_name)) {
    SetSequence(name, SymSequence(source.GetSequence(source_name)));
  } else {
    EXT_THROW_INVALID("CopyValueFrom: unknown value '", source_name, "'.");
  }
}

void ShapesContext::CheckStructuredCompatibility(const std::string &name,
                                                 const ShapesContext &other,
                                                 const std::string &other_name) const {
  const bool structured = HasType(name) && HasStructuredType(GetType(name));
  const bool other_structured =
      other.HasType(other_name) && HasStructuredType(other.GetType(other_name));
  EXT_ENFORCE_INVALID(structured == other_structured,
                      "Control-flow outputs have incompatible structured types.");
  if (structured) {
    CheckCatalogueCompatibility(*this, other, GetType(name));
    EXT_ENFORCE_INVALID(GetType(name).SerializeAsString() ==
                            other.GetType(other_name).SerializeAsString(),
                        "Control-flow outputs have incompatible structured types.");
  }
  EXT_ENFORCE_INVALID(HasEncodedValue(name) == other.HasEncodedValue(other_name),
                      "Control-flow outputs have incompatible encoded layouts.");
  if (HasEncodedValue(name)) {
    EncodedValueProto left = GetEncodedValue(name);
    EncodedValueProto right = other.GetEncodedValue(other_name);
    left.set_name("");
    right.set_name("");
    EXT_ENFORCE_INVALID(left.SerializeAsString() == right.SerializeAsString(),
                        "Control-flow merging of different encoded values is unsupported.");
  }
}

// ── Opset versions ──────────────────────────────────────────────────

void ShapesContext::SetOpsetVersion(const std::string &domain, int opset_version) {
  opsets_[NormaliseDomain(domain)] = opset_version;
}

bool ShapesContext::HasOpsetVersion(const std::string &domain) const {
  return opsets_.find(NormaliseDomain(domain)) != opsets_.end();
}

int ShapesContext::OpsetVersion(const std::string &domain) const {
  auto it = opsets_.find(NormaliseDomain(domain));
  return it == opsets_.end() ? kUnknownOpsetVersion : it->second;
}

// ── Model-local functions ───────────────────────────────────────────

void ShapesContext::SetLocalFunction(std::shared_ptr<const FunctionProto> func) {
  EXT_ENFORCE_INVALID(func != nullptr, "SetLocalFunction: func must not be nullptr.");
  const std::string key = std::string(func->domain()) + ":" + std::string(func->name());
  local_functions_[key] = func.get();
  owned_local_functions_[key] = std::move(func);
}

void ShapesContext::CopyLocalFunctions(const ShapesContext &context) {
  local_functions_ = context.local_functions_;
  owned_local_functions_ = context.owned_local_functions_;
}

void ShapesContext::ClearLocalFunctions() noexcept {
  local_functions_.clear();
  owned_local_functions_.clear();
}

// ── Custom shape-inference hooks ────────────────────────────────────

void ShapesContext::SetCustomShapeInferenceFunction(const std::string &domain,
                                                    const std::string &op_type,
                                                    CustomComputeShapeFn fn) {
  EXT_ENFORCE_INVALID(!op_type.empty(),
                      "SetCustomShapeInferenceFunction: op_type must not be empty.");
  EXT_ENFORCE_INVALID(static_cast<bool>(fn),
                      "SetCustomShapeInferenceFunction: fn must not be empty.");
  custom_shape_inference_[NormaliseDomain(domain) + ":" + op_type] = std::move(fn);
}

const ShapesContext::CustomComputeShapeFn *
ShapesContext::GetCustomShapeInferenceFunction(const std::string &domain,
                                               const std::string &op_type) const {
  auto it = custom_shape_inference_.find(NormaliseDomain(domain) + ":" + op_type);
  return it == custom_shape_inference_.end() ? nullptr : &it->second;
}

bool ShapesContext::RemoveCustomShapeInferenceFunction(const std::string &domain,
                                                       const std::string &op_type) {
  return custom_shape_inference_.erase(NormaliseDomain(domain) + ":" + op_type) > 0;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes

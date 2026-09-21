// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_value.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

RuntimeValue::RuntimeValue(EncodedValueProto value)
    : kind(Kind::kEncoded), encoded(std::make_shared<EncodedValueProto>(std::move(value))) {}

RuntimeValue RuntimeValue::FromEncodedView(const EncodedValueProto &value,
                                           std::shared_ptr<void> owner) {
  EXT_ENFORCE_INVALID(owner.use_count() != 0,
                      "RuntimeValue::FromEncodedView: an owner token is required.");
  RuntimeValue result;
  result.kind = Kind::kEncoded;
  result.encoded = std::shared_ptr<const EncodedValueProto>(std::move(owner), &value);
  return result;
}

const EncodedValueProto &RuntimeValue::Encoded() const {
  EXT_ENFORCE_INVALID(encoded != nullptr, "RuntimeValue: missing encoded message.");
  return *encoded;
}

RuntimeValue RuntimeValue::BorrowView() const { return CopyAtDepth(0, false); }

RuntimeValue RuntimeValue::DeepCopy() const { return CopyAtDepth(0, true); }

RuntimeValue RuntimeValue::Retain(const StructTypeCatalogue &catalogue) && {
  RetainAtDepth(0, catalogue);
  return std::move(*this);
}

void RuntimeValue::RetainAtDepth(size_t depth, const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
  if (kind == Kind::kTensor)
    tensor = std::move(tensor).RetainStorage();
  else if (kind == Kind::kEncoded) {
    if (Encoded().has_struct_type())
      ValidatePersistentStructType(catalogue, Encoded().struct_type());
    if (Encoded().has_logical_type())
      ValidatePersistentType(catalogue, Encoded().logical_type());
    const auto &raw = Encoded().raw_data();
    EXT_ENFORCE_INVALID(!raw.is_borrowed() || raw.owner().use_count() != 0 || raw.empty(),
                        "RuntimeValue::Retain: cannot retain an ownerless encoded payload.");
  } else
    for (auto &[name, value] : fields)
      value.RetainAtDepth(depth + 1, catalogue);
}

RuntimeValue RuntimeValue::CopyAtDepth(size_t depth, bool owned) const {
  EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
  if (kind == Kind::kTensor)
    return RuntimeValue(owned || tensor.data_type == DataType::STRING ? tensor.ToOwned()
                                                                      : tensor.BorrowView());
  if (kind == Kind::kEncoded) {
    if (!owned) {
      RuntimeValue result;
      result.kind = kind;
      result.encoded = encoded;
      return result;
    }
    EncodedValueProto copy;
    copy.ParseFromString(Encoded().SerializeAsString());
    return RuntimeValue(std::move(copy));
  }
  RuntimeValue result;
  for (const auto &[name, value] : fields)
    result.fields.emplace(name, value.CopyAtDepth(depth + 1, owned));
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

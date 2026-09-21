// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Represents tensor fields, named structures and immutable encoded proto values.
 *
 * RuntimeContext stores ordinary tensor edges in its existing tensor map, not
 * here. This recursive representation supplies the structured/encoded edges
 * required by TypeProto, independently of whether an edge is persistent.
 */
struct RuntimeValue {
  static constexpr size_t kMaxDepth = 64;
  enum class Kind { kTensor, kStruct, kEncoded };
  Kind kind = Kind::kStruct;
  Tensor tensor;
  std::unordered_map<std::string, RuntimeValue> fields;
  std::shared_ptr<const EncodedValueProto> encoded;

  RuntimeValue() = default;
  explicit RuntimeValue(Tensor value) : kind(Kind::kTensor), tensor(std::move(value)) {}
  explicit RuntimeValue(std::unordered_map<std::string, RuntimeValue> value)
      : kind(Kind::kStruct), fields(std::move(value)) {}
  explicit RuntimeValue(EncodedValueProto value)
      : kind(Kind::kEncoded), encoded(std::make_shared<EncodedValueProto>(std::move(value))) {}

  /** Returns an immutable message view with an explicit lifetime owner. */
  static RuntimeValue FromEncodedView(const EncodedValueProto &value, std::shared_ptr<void> owner) {
    EXT_ENFORCE_INVALID(owner.use_count() != 0,
                        "RuntimeValue::FromEncodedView: an owner token is required.");
    RuntimeValue result;
    result.kind = Kind::kEncoded;
    result.encoded = std::shared_ptr<const EncodedValueProto>(std::move(owner), &value);
    return result;
  }

  /** Returns the immutable encoded message. */
  const EncodedValueProto &Encoded() const {
    EXT_ENFORCE_INVALID(encoded != nullptr, "RuntimeValue: missing encoded message.");
    return *encoded;
  }

  /** Returns independent metadata borrowing tensor storage, without changing this value. */
  RuntimeValue BorrowView() const { return CopyAtDepth(0, false); }

  /** Returns independent owned payloads. */
  RuntimeValue DeepCopy() const { return CopyAtDepth(0, true); }

  /** Consumes a selected whole value and explicitly retains its backing storage. */
  RuntimeValue Retain() && {
    RetainAtDepth(0);
    return std::move(*this);
  }

private:
  void RetainAtDepth(size_t depth) {
    EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
    if (kind == Kind::kTensor)
      tensor = std::move(tensor).RetainStorage();
    else if (kind == Kind::kEncoded) {
      const auto &raw = Encoded().raw_data();
      EXT_ENFORCE_INVALID(!raw.is_borrowed() || raw.owner().use_count() != 0 || raw.empty(),
                          "RuntimeValue::Retain: cannot retain an ownerless encoded payload.");
    } else
      for (auto &[name, value] : fields)
        value.RetainAtDepth(depth + 1);
  }

  RuntimeValue CopyAtDepth(size_t depth, bool owned) const {
    EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
    if (kind == Kind::kTensor)
      return RuntimeValue(owned ? tensor.ToOwned() : tensor.BorrowView());
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
};

using RuntimeValueMap = std::unordered_map<std::string, RuntimeValue>;

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

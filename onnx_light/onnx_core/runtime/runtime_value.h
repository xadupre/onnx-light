// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Carries a tensor, named structured fields, or an explicitly encoded payload. */
struct RuntimeValue {
  static constexpr size_t kMaxDepth = 64;
  enum class Kind { kTensor, kStruct, kEncoded };
  Kind kind = Kind::kStruct;
  Tensor tensor;
  std::unordered_map<std::string, RuntimeValue> fields;
  mutable EncodedValueProto encoded;
  mutable std::shared_ptr<EncodedValueProto> encoded_owner;

  RuntimeValue() = default;
  explicit RuntimeValue(Tensor value) : kind(Kind::kTensor), tensor(std::move(value)) {}
  explicit RuntimeValue(std::unordered_map<std::string, RuntimeValue> value)
      : kind(Kind::kStruct), fields(std::move(value)) {}
  explicit RuntimeValue(EncodedValueProto value)
      : kind(Kind::kEncoded), encoded(std::move(value)) {}

  /**
   * Returns a read-only encoded view retaining its external message owner.
   *
   * Neither the message nor its payload is copied or modified. The owner must
   * keep the complete message and all of its borrowed storage alive.
   */
  static RuntimeValue FromEncodedView(const EncodedValueProto &value, std::shared_ptr<void> owner) {
    EXT_ENFORCE_INVALID(owner.use_count() != 0,
                        "RuntimeValue::FromEncodedView: an owner token is required.");
    RuntimeValue result;
    result.kind = Kind::kEncoded;
    result.encoded_owner = std::shared_ptr<EncodedValueProto>(
        std::move(owner), const_cast<EncodedValueProto *>(&value));
    return result;
  }

  /** Returns a recursively owned copy with no borrowed or allocator-backed storage. */
  RuntimeValue DeepCopy() const { return DeepCopyAtDepth(0); }

  /** Returns the encoded payload, including one promoted to shared ownership. */
  const EncodedValueProto &Encoded() const { return encoded_owner ? *encoded_owner : encoded; }

  /** Returns read-only aliases retaining payload owners, without payload copies. */
  RuntimeValue Share() const { return ShareAtDepth(0); }

private:
  RuntimeValue ShareAtDepth(size_t depth) const {
    EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
    if (kind == Kind::kTensor)
      return RuntimeValue(tensor.ShareStorage());
    RuntimeValue result;
    result.kind = kind;
    if (kind == Kind::kEncoded) {
      const auto &raw = Encoded().raw_data();
      EXT_ENFORCE_INVALID(!raw.is_borrowed() || raw.owner().use_count() != 0 || raw.empty(),
                          "RuntimeValue::Share: cannot retain an ownerless encoded payload.");
      if (!encoded_owner) {
        encoded_owner = std::make_shared<EncodedValueProto>(std::move(encoded));
      }
      result.encoded_owner = encoded_owner;
    } else {
      for (const auto &[name, value] : fields)
        result.fields.emplace(name, value.ShareAtDepth(depth + 1));
    }
    return result;
  }

  RuntimeValue DeepCopyAtDepth(size_t depth) const {
    EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
    if (kind == Kind::kTensor)
      return RuntimeValue(tensor.ToOwned());
    if (kind == Kind::kEncoded) {
      EncodedValueProto copy;
      copy.ParseFromString(Encoded().SerializeAsString());
      return RuntimeValue(std::move(copy));
    }
    RuntimeValue copy;
    for (const auto &[name, value] : fields)
      copy.fields.emplace(name, value.DeepCopyAtDepth(depth + 1));
    return copy;
  }
};

using RuntimeValueMap = std::unordered_map<std::string, RuntimeValue>;

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

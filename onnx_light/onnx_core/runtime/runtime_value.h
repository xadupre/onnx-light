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
  EncodedValueProto encoded;

  RuntimeValue() = default;
  explicit RuntimeValue(Tensor value) : kind(Kind::kTensor), tensor(std::move(value)) {}
  explicit RuntimeValue(std::unordered_map<std::string, RuntimeValue> value)
      : kind(Kind::kStruct), fields(std::move(value)) {}
  explicit RuntimeValue(EncodedValueProto value)
      : kind(Kind::kEncoded), encoded(std::move(value)) {}

  /** Returns a recursively owned copy with no borrowed or allocator-backed storage. */
  RuntimeValue DeepCopy() const { return DeepCopyAtDepth(0); }

private:
  RuntimeValue DeepCopyAtDepth(size_t depth) const {
    EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
    if (kind == Kind::kTensor)
      return RuntimeValue(tensor.ToOwned());
    if (kind == Kind::kEncoded) {
      EncodedValueProto copy;
      copy.ParseFromString(encoded.SerializeAsString());
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

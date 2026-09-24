// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_verify.h"
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

class QuantizationParameterCatalogue;
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
  std::shared_ptr<const QuantizationParameterCatalogue> quantization_parameters;

  RuntimeValue() = default;
  explicit RuntimeValue(Tensor value) : kind(Kind::kTensor), tensor(std::move(value)) {}
  explicit RuntimeValue(std::unordered_map<std::string, RuntimeValue> value)
      : kind(Kind::kStruct), fields(std::move(value)) {}
  explicit RuntimeValue(EncodedValueProto value);

  /** Returns an immutable message view with an explicit lifetime owner. */
  static RuntimeValue FromEncodedView(const EncodedValueProto &value, std::shared_ptr<void> owner);

  /** Returns the immutable encoded message. */
  const EncodedValueProto &Encoded() const;

  /** Borrows numeric storage and materializes ordinary string fields without changing this value.
   */
  RuntimeValue BorrowView() const;

  /** Returns independent owned payloads. */
  RuntimeValue DeepCopy() const;

  /** Consumes a selected whole value and explicitly retains its backing storage. */
  RuntimeValue Retain(const StructTypeCatalogue &catalogue = {}) &&;

private:
  void RetainAtDepth(size_t depth, const StructTypeCatalogue &catalogue);
  RuntimeValue CopyAtDepth(size_t depth, bool owned) const;
};

using RuntimeValueMap = std::unordered_map<std::string, RuntimeValue>;

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns::detail {

/// Returns whether ``type`` is one of the standard floating-point tensor types.
inline bool IsSupportedFloat(core::symbolic::TensorType type) {
  using core::symbolic::TensorType;
  return type == TensorType::kFloat16 || type == TensorType::kBfloat16 ||
         type == TensorType::kFloat || type == TensorType::kDouble;
}

/// Returns the storage width of a standard floating-point tensor type.
inline int FloatWidth(core::symbolic::TensorType type) {
  using core::symbolic::TensorType;
  switch (type) {
  case TensorType::kFloat16:
  case TensorType::kBfloat16:
    return 16;
  case TensorType::kFloat:
    return 32;
  case TensorType::kDouble:
    return 64;
  default:
    return 0;
  }
}

/// Returns whether ``node`` has the canonical unary ONNX Cast form.
inline bool IsDefaultCast(const NodeProto &node) {
  return node.op_type().value() == "Cast" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() == 1 &&
         node.output_size() == 1;
}

inline bool CastTarget(const NodeProto &node, core::symbolic::TensorType &type) {
  const AttributeProto *attribute = FindAttribute(node, "to");
  if (attribute == nullptr || attribute->type() != AttributeProto::AttributeType::INT) {
    return false;
  }
  type = core::symbolic::DataTypeToTensorType(static_cast<TensorProto::DataType>(attribute->i()));
  return type != core::symbolic::TensorType::kUndefined;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns::detail

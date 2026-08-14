// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns::detail {

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

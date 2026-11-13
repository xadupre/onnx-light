// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Cast in default domain from version 9 to 8

#pragma once

#include <memory>

#include "onnx_lib/version_converter/adapters/type_restriction.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Cast_9_8 final : public TypeRestriction {
public:
  explicit Cast_9_8()
      : TypeRestriction("Cast", OpSetID(9), OpSetID(8), {TensorProto_DataType_STRING}) {}

  void adapt_cast_9_8(const std::shared_ptr<Graph> & /*unused*/, Node *node) const {
    ONNX_ASSERTM(node->inputs().size() >= 1, "Cast node must have at least 1 input")
    const Value *input = node->inputs()[0];
    ONNX_ASSERTM(!input->uniqueName().empty(), "Cast requires an input.")
    ONNX_ASSERTM(input->type() == nullptr || !input->type()->has_type() ||
                     input->type()->has_tensor_type(),
                 "Cast requires a tensor input.")
    assertAllowed(input);
    ONNX_ASSERTM(node->hasAttribute(kto), "Cast requires a 'to' attribute.")
    ONNX_ASSERTM(node->i(kto) != TensorProto_DataType_UNDEFINED, "Cast target type must be known.")
    if (node->i(kto) == TensorProto_DataType_STRING)
      ONNX_ASSERTM(false, "Casting From/To STRING data type is not supported")
  }

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    adapt_cast_9_8(graph, node);
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

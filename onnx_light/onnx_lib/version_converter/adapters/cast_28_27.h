// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "onnx_lib/version_converter/adapters/type_restriction.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Cast_28_27 final : public TypeRestriction {
public:
  Cast_28_27()
      : TypeRestriction("Cast", OpSetID(28), OpSetID(27),
                        {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {}

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    ONNX_ASSERTM(node->hasAttribute(kto), "Cast requires a 'to' attribute.");
    const auto to = node->i(kto);
    ONNX_ASSERTM(to != TensorProto_DataType_FLOAT6E2M3 && to != TensorProto_DataType_FLOAT6E3M2,
                 "Cast downgrade to opset 27 does not support float6 destinations.");
    return TypeRestriction::adapt(std::move(graph), node);
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

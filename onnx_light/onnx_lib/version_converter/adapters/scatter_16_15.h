// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for ScatterElements and ScatterND in default domain from version 16 to 15.
// Opset 16 added bfloat16 and 'reduction' attribute; downgrading requires type
// restriction and removal of reduction (only "none" allowed).

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "onnx_lib/version_converter/adapters/transformers.h"
#include "onnx_lib/version_converter/adapters/type_restriction.h"
#include "onnx_lib/version_converter/internal_symbol.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Scatter_16_15 : public TypeRestriction {
public:
  explicit Scatter_16_15(const std::string &op_name, const OpSetID &initial, const OpSetID &target,
                         const std::vector<TensorProto_DataType> &unallowed_types)
      : TypeRestriction(op_name, initial, target, unallowed_types) {}

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    ONNX_ASSERTM(node->inputs().size() >= 3, "Scatter requires three inputs.")
    for (size_t i = 0; i < 3; ++i) {
      ONNX_ASSERTM(!node->inputs()[i]->uniqueName().empty(), "Scatter requires three inputs.")
    }
    adapt_type_restriction(graph, node);
    RemoveAttribute(Symbol("reduction"), std::string("none"))(graph, node);
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

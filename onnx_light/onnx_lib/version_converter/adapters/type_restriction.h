// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Add in default domain from version 6 to 5

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class TypeRestriction : public Adapter {
public:
  explicit TypeRestriction(const std::string &op_name, const OpSetID &initial,
                           const OpSetID &target,
                           const std::vector<TensorProto_DataType> &unallowed_types)
      : Adapter(op_name, initial, target), unallowed_types_(unallowed_types) {}

  void adapt_type_restriction(const std::shared_ptr<Graph> & /*unused*/, const Node *node) const {
    // Since consumed_inputs is optional, no need to add it (as in batchnorm)
    // Iterate over all inputs and outputs
    for (const Value *input : node->inputs()) {
      isUnallowed(input);
    }
    for (const Value *output : node->outputs()) {
      isUnallowed(output);
    }
  }

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    adapt_type_restriction(graph, node);
    return node;
  }

private:
  std::vector<TensorProto_DataType> unallowed_types_;

  void isUnallowed(const Value *val) const {
    if (val->uniqueName().empty()) {
      return;
    }

    int32_t elem_type = val->elemType();
    const TypeProto *type = val->type().get();
    bool has_type = elem_type != TensorProto_DataType_UNDEFINED;
    while (elem_type == TensorProto_DataType_UNDEFINED && type != nullptr) {
      switch (type->value_case()) {
      case TypeProto::kTensorType:
        if (type->tensor_type().has_elem_type()) {
          elem_type = type->tensor_type().elem_type();
          has_type = true;
        }
        type = nullptr;
        break;
      case TypeProto::kSparseTensorType:
        if (type->sparse_tensor_type().has_elem_type()) {
          elem_type = type->sparse_tensor_type().elem_type();
          has_type = true;
        }
        type = nullptr;
        break;
      case TypeProto::kSequenceType:
        type = &type->sequence_type().elem_type();
        break;
      case TypeProto::kOptionalType:
        type = &type->optional_type().elem_type();
        break;
      case TypeProto::kMapType:
        type = &type->map_type().value_type();
        break;
      default:
        has_type = type->value_case() != TypeProto::VALUE_NOT_SET;
        type = nullptr;
        break;
      }
    }
    ONNX_ASSERTM(has_type, "DataType of Input or Output of operator '", name(),
                 "' must be known to convert to Opset Version ",
                 static_cast<int64_t>(target_version().version()), ".")
    ONNX_ASSERTM(std::find(std::begin(unallowed_types_), std::end(unallowed_types_), elem_type) ==
                     std::end(unallowed_types_),
                 "DataType (", elem_type, ") of Input or Output of operator '", name(),
                 "' is unallowed for Opset Version ",
                 static_cast<int64_t>(target_version().version()), ".")
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

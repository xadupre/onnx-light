// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Add in default domain from version 6 to 5

#pragma once

#include <algorithm>
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
      assertAllowed(input);
    }
    for (const Value *output : node->outputs()) {
      assertAllowed(output);
    }
  }

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    adapt_type_restriction(graph, node);
    return node;
  }

protected:
  void assertAllowed(const Value *val) const {
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
          has_type = elem_type != TensorProto_DataType_UNDEFINED;
        }
        type = nullptr;
        break;
      case TypeProto::kSparseTensorType:
        if (type->sparse_tensor_type().has_elem_type()) {
          elem_type = type->sparse_tensor_type().elem_type();
          has_type = elem_type != TensorProto_DataType_UNDEFINED;
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
    const auto check_type = [&](int32_t candidate) {
      ONNX_ASSERTM(candidate != TensorProto_DataType_UNDEFINED &&
                       TensorProto::DataType_IsValid(candidate),
                   "DataType of Input or Output of operator '", name(), "' must be valid.")
      ONNX_ASSERTM(std::find(std::begin(unallowed_types_), std::end(unallowed_types_), candidate) ==
                       std::end(unallowed_types_),
                   "DataType (", candidate, ") of Input or Output of operator '", name(),
                   "' is unallowed for Opset Version ",
                   static_cast<int64_t>(target_version().version()), ".")
    };
    if (!has_type) {
      const auto *possible_types = val->possibleElemTypes();
      ONNX_ASSERTM(possible_types != nullptr && !possible_types->empty(),
                   "DataType of Input or Output of operator '", name(),
                   "' must be known to convert to Opset Version ",
                   static_cast<int64_t>(target_version().version()), ".")
      for (const int32_t possible_type : *possible_types) {
        check_type(possible_type);
      }
    } else if (elem_type != TensorProto_DataType_UNDEFINED) {
      check_type(elem_type);
    }
  }

private:
  std::vector<TensorProto_DataType> unallowed_types_;
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

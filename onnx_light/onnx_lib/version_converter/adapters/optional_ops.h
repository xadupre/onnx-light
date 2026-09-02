// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class OptionalOpsAdapter final : public Adapter {
public:
  OptionalOpsAdapter(const std::string &name, const OpSetID &initial_version,
                     const OpSetID &target_version,
                     const std::vector<TensorProto_DataType> &unallowed_types,
                     bool allow_optional_input = true, bool allow_nonoptional_input = true,
                     bool allow_no_input = false)
      : Adapter(name, initial_version, target_version), unallowed_types_(unallowed_types),
        allow_optional_input_(allow_optional_input),
        allow_nonoptional_input_(allow_nonoptional_input), allow_no_input_(allow_no_input) {}

  Node *adapt(std::shared_ptr<Graph> /*graph*/, Node *node) const override {
    const TypeProto *optional_or_element_type = nullptr;
    ONNX_ASSERTM(allow_no_input_ || !node->inputs().empty(), "No input to operator '", name(),
                 "' is unallowed for Opset Version ",
                 static_cast<int64_t>(target_version().version()));
    if (node->inputs().empty()) {
      const Symbol type("type");
      if (!node->hasAttribute(type)) {
        return node;
      }
      optional_or_element_type = &node->tp(type);
    } else {
      optional_or_element_type = node->input()->type().get();
    }

    int32_t tensor_element_type = -1;
    if (optional_or_element_type == nullptr) {
      ONNX_ASSERTM(allow_nonoptional_input_, "Non-Optional input to operator '", name(),
                   "' is unallowed for Opset Version ",
                   static_cast<int64_t>(target_version().version()));
      tensor_element_type = node->input()->elemType();
    } else {
      ONNX_ASSERTM(
          (allow_optional_input_ && optional_or_element_type->has_optional_type()) ||
              (allow_nonoptional_input_ && !optional_or_element_type->has_optional_type()),
          "Specified type of Input of operator '", name(), "' is unallowed for Opset Version ",
          static_cast<int64_t>(target_version().version()));
      const TypeProto &element_type = optional_or_element_type->has_optional_type()
                                          ? optional_or_element_type->optional_type().elem_type()
                                          : *optional_or_element_type;
      ONNX_ASSERT(element_type.has_tensor_type() || element_type.has_sequence_type() ||
                  element_type.has_sparse_tensor_type());
      const TypeProto &tensor_type =
          element_type.has_sequence_type() ? element_type.sequence_type().elem_type() : element_type;
      ONNX_ASSERT(tensor_type.has_tensor_type() || tensor_type.has_sparse_tensor_type());
      tensor_element_type = tensor_type.has_tensor_type()
                                ? tensor_type.tensor_type().elem_type()
                                : tensor_type.sparse_tensor_type().elem_type();
    }

    ONNX_ASSERTM(
        std::find(unallowed_types_.begin(), unallowed_types_.end(), tensor_element_type) ==
            unallowed_types_.end(),
        "DataType (", tensor_element_type, ") of Input of operator '", name(),
        "' is unallowed for Opset Version ", static_cast<int64_t>(target_version().version()));
    return node;
  }

private:
  std::vector<TensorProto_DataType> unallowed_types_;
  bool allow_optional_input_;
  bool allow_nonoptional_input_;
  bool allow_no_input_;
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

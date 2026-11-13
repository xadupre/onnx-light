// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnx_lib/version_converter/adapters/type_restriction.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class ReduceLogSum_27_28 final : public TypeRestriction {
public:
  explicit ReduceLogSum_27_28(const std::string &op_name,
                              const std::vector<TensorProto_DataType> &unallowed_types)
      : TypeRestriction(op_name, OpSetID(27), OpSetID(28), unallowed_types) {}

  Node *adapt(std::shared_ptr<Graph> /*graph*/, Node *node) const override {
    ONNX_ASSERTM(!node->inputs().empty() && !node->inputs()[0]->uniqueName().empty(),
                 "ReduceLogSum requires a data input.")
    assertAllowed(node->inputs()[0]);
    for (const Value *output : node->outputs()) {
      assertAllowed(output);
    }
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

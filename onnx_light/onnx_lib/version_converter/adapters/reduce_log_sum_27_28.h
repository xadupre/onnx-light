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

class ReduceLogSum_27_28 final : public Adapter {
public:
  explicit ReduceLogSum_27_28(const std::string &op_name,
                              const std::vector<TensorProto_DataType> &unallowed_types)
      : Adapter(op_name, OpSetID(27), OpSetID(28)), unallowed_types_(unallowed_types) {}

  Node *adapt(std::shared_ptr<Graph> /*graph*/, Node *node) const override {
    if (!node->inputs().empty()) {
      assertAllowed(node->inputs()[0], "input");
    }
    for (const Value *output : node->outputs()) {
      assertAllowed(output, "output");
    }
    return node;
  }

private:
  std::vector<TensorProto_DataType> unallowed_types_;

  void assertAllowed(const Value *value, const char *kind) const {
    ONNX_ASSERTM(std::find(unallowed_types_.begin(), unallowed_types_.end(), value->elemType()) ==
                     unallowed_types_.end(),
                 "DataType (", value->elemType(), ") of ", kind, " of operator '", name(),
                 "' is not supported in Opset Version ",
                 static_cast<int64_t>(target_version().version()), ".");
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

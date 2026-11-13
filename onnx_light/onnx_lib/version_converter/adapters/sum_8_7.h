// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Sum in default domain from version 8 to 7

#pragma once

#include <memory>
#include <vector>

#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Sum_8_7 final : public Adapter {
public:
  explicit Sum_8_7() : Adapter("Sum", OpSetID(8), OpSetID(7)) {}

  void adapt_sum_8_7(const std::shared_ptr<Graph> & /*unused*/, Node *node) const {
    const ArrayRef<Value *> &inputs = node->inputs();
    ONNX_ASSERTM(!inputs.empty(), "Sum requires at least one input.")
    for (const Value *input : inputs) {
      ONNX_ASSERTM(!input->uniqueName().empty(), "Sum inputs must be present.")
    }
    for (size_t i = 1; i < inputs.size(); i++) {
      if (inputs[i - 1] == inputs[i])
        continue;
      ONNX_ASSERTM(inputs[i - 1]->has_sizes() && inputs[i]->has_sizes(),
                   "Sum downgrade requires known input shapes.")
      const auto &first = inputs[i - 1]->sizes();
      const auto &second = inputs[i]->sizes();
      ONNX_ASSERTM(first.size() == second.size(), "Sum downgrade requires equal input shapes.")
      for (size_t dim = 0; dim < first.size(); ++dim) {
        const auto &a = first[dim];
        const auto &b = second[dim];
        const bool same_integer = a.is_int && b.is_int && a.dim == b.dim;
        const bool same_symbol = !a.is_unknown && !b.is_unknown && !a.is_int && !b.is_int &&
                                 !a.param.empty() && a.param == b.param;
        ONNX_ASSERTM(same_integer || same_symbol, "Sum downgrade requires equal input shapes.")
      }
    }
  }

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    adapt_sum_8_7(graph, node);
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

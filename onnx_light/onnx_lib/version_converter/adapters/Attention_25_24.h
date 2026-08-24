// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Attention in default domain from version 25 to 24

#pragma once

#include <cstdint>
#include <memory>

#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Attention_25_24 final : public Adapter {
public:
  explicit Attention_25_24() : Adapter("Attention", OpSetID(25), OpSetID(24)) {}

  void adapt_attention_25_24(const std::shared_ptr<Graph> & /*unused*/, Node *node) const {
    for (Symbol attr : {kleft_window_size, kright_window_size}) {
      if (node->hasAttribute(attr)) {
        const int64_t value = node->i(attr);
        if (value != -1) {
          const char *attr_name =
              attr == kleft_window_size ? "left_window_size" : "right_window_size";
          ONNX_ASSERTM(false, "Attention 25->24 downgrade: ", attr_name,
                       " must be -1 (disabled) for conversion to opset 24, got ", value,
                       ". Windowed attention is not representable in opset 24.");
        }
        node->removeAttribute(attr);
      }
    }
  }

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    adapt_attention_25_24(graph, node);
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

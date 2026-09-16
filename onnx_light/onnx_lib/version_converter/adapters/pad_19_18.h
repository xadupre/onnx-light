// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Pad in default domain from version 19 to 18

#pragma once

#include <memory>

#include "onnx_lib/version_converter/adapters/adapter.h"
#include "onnx_lib/version_converter/internal_symbol.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Pad_19_18 final : public Adapter {
public:
  explicit Pad_19_18() : Adapter("Pad", OpSetID(19), OpSetID(18)) {}

  Node *adapt(std::shared_ptr<Graph> /*graph*/, Node *node) const override {
    if (node->hasAttribute(kmode)) {
      ONNX_ASSERTM(node->s(kmode) != "wrap",
                   "Pad mode='wrap' is not supported when converting from opset 19 "
                   "to opset 18.")
    }
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

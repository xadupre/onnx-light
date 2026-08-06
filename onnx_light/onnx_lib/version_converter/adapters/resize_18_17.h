// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Resize in default domain from version 18 to 17.

#pragma once

#include <memory>

#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Resize_18_17 final : public Adapter {
public:
  explicit Resize_18_17() : Adapter("Resize", OpSetID(18), OpSetID(17)) {}

  Node *adapt(std::shared_ptr<Graph> /*graph*/, Node *node) const override {
    const Symbol antialias("antialias");
    const Symbol keep_aspect_ratio_policy("keep_aspect_ratio_policy");
    const Symbol axes("axes");

    if (node->hasAttribute(antialias)) {
      ONNX_ASSERTM(
          node->i(antialias) == 0,
          "Resize antialias=1 is not supported when converting from opset 18 to opset 17.");
      node->removeAttribute(antialias);
    }

    if (node->hasAttribute(keep_aspect_ratio_policy)) {
      ONNX_ASSERTM(
          node->s(keep_aspect_ratio_policy) == "stretch",
          "Resize keep_aspect_ratio_policy must be 'stretch' when converting from opset 18 "
          "to opset 17.");
      node->removeAttribute(keep_aspect_ratio_policy);
    }

    ONNX_ASSERTM(!node->hasAttribute(axes),
                 "Resize axes is not supported when converting from opset 18 to opset 17.");
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

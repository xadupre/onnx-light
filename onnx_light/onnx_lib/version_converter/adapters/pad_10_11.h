// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Pad in default domain from version 10 to 11

#pragma once

#include <memory>
#include <vector>

#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class Pad_10_11 final : public Adapter {
public:
  explicit Pad_10_11() : Adapter("Pad", OpSetID(10), OpSetID(11)) {}

  void adapt_pad_10_11(const std::shared_ptr<Graph> &graph, Node *node) const {
    // Turn pads attribute into input
    Tensor t_pads;
    t_pads.elem_type() = TensorProto_DataType_INT64;
    auto &data_pads = t_pads.int64s();
    for (int64_t shape : node->is(kpads)) {
      data_pads.emplace_back(shape);
    }
    t_pads.sizes() = std::vector<int64_t>{static_cast<int64_t>(data_pads.size())};
    Value *v_pads = graph->addInitializerAndCreateValue(t_pads);
    node->addInput(v_pads);
    node->removeAttribute(kpads);
    // Turn value attribute into input
    if (!node->hasAttribute(kmode) || node->s(kmode) == "constant") {
      if (!node->hasAttribute(kvalue))
        node->f_(kvalue, 0.);
      ONNX_ASSERTM(!node->inputs().empty() &&
                       node->inputs()[0]->elemType() != TensorProto_DataType_UNDEFINED,
                   "Pad input type must be known to convert the value attribute.")
      const int32_t input_type = node->inputs()[0]->elemType();
      Tensor t_value;
      t_value.elem_type() = TensorProto_DataType_FLOAT;
      auto &data_value = t_value.floats();
      float float_value = static_cast<float>(node->f(kvalue));
      data_value.emplace_back(float_value);
      Node *constant = graph->create(kConstant);
      constant->insertBefore(node);
      constant->t_(kvalue, t_value);
      constant->output()->setElemType(TensorProto_DataType_FLOAT);
      Value *value = constant->output();
      if (input_type != TensorProto_DataType_FLOAT) {
        Node *cast = graph->create(kCast);
        cast->i_(kto, input_type);
        cast->addInput(value);
        cast->insertBefore(node);
        cast->output()->setElemType(input_type);
        value = cast->output();
      }
      node->addInput(value);
      node->removeAttribute(kvalue);
    }
  }

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    adapt_pad_10_11(graph, node);
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

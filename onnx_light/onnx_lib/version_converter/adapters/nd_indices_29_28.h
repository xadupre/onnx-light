// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class NDIndices_29_28 final : public Adapter {
public:
  explicit NDIndices_29_28(const std::string &op_name)
      : Adapter(op_name, OpSetID(29), OpSetID(28)) {}

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    ONNX_ASSERTM(node->inputs().size() >= 2, "ND indexing requires at least two inputs.");
    Value *indices = node->inputs()[1];
    ONNX_ASSERTM(!indices->uniqueName().empty(), "ND indexing requires an indices input.");
    if (indices->type() != nullptr && indices->type()->has_type()) {
      ONNX_ASSERTM(indices->type()->has_tensor_type(), "ND indexing requires tensor indices.");
      const auto &tensor_type = indices->type()->tensor_type();
      if (indices->elemType() == TensorProto_DataType_UNDEFINED && tensor_type.has_elem_type()) {
        indices->setElemType(tensor_type.elem_type());
      }
    }
    ONNX_ASSERTM(indices->elemType() == TensorProto_DataType_UNDEFINED ||
                     indices->elemType() == TensorProto_DataType_INT32 ||
                     indices->elemType() == TensorProto_DataType_INT64,
                 "ND indexing requires int32 or int64 indices.");
    if (indices->elemType() == TensorProto_DataType_UNDEFINED) {
      if (const auto *possible_types = indices->possibleElemTypes()) {
        for (const int32_t type : *possible_types) {
          ONNX_ASSERTM(type == TensorProto_DataType_INT32 || type == TensorProto_DataType_INT64,
                       "ND indexing requires int32 or int64 indices.");
        }
      } else {
        // The original ND operator constrains its indices without identifying one exact type.
        indices->setPossibleElemTypes({TensorProto_DataType_INT32, TensorProto_DataType_INT64});
      }
    }
    if (indices->elemType() != TensorProto_DataType_INT64) {
      // Widening preserves every int32 index, including negative indices.
      // Unknown types are cast too: both valid index types fit in int64.
      Node *cast = graph->create(kCast);
      cast->i_(kto, TensorProto_DataType_INT64);
      cast->addInput(indices);
      cast->insertBefore(node);
      cast->output()->setElemType(TensorProto_DataType_INT64);
      if (indices->has_sizes()) {
        cast->output()->setSizes(indices->sizes());
      }
      node->replaceInput(1, cast->output());
    }
    return node;
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

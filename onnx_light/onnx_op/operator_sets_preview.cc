// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_preview.h"
#include "onnx_op/operator_sets_preview_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace preview {

namespace {

// Mirrors OpSchema::all_float_types_ir4() ordering used by the upstream
// FlexAttention schema: bfloat16, float16, float, double.
std::vector<TensorType> FlexAttentionFloatTypes() {
  return {
      TensorType::kBfloat16,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  };
}

LightOpSchema MakeFlexAttentionSchema() {
  return LightOpSchema(
      "FlexAttention", kOnnxPreviewDomain, 1, MakeFlexAttentionDoc(),
      {
          {"Q", "Query tensor with shape `(batch_size, q_num_heads, q_seq_len, head_size)`.", "T1"},
          {"K", "Key tensor with shape `(batch_size, kv_num_heads, kv_seq_len, head_size)`.", "T1"},
          {"V", "Value tensor with shape `(batch_size, kv_num_heads, kv_seq_len, v_head_size)`.",
           "T1"},
      },
      {
          {"Y", "Output tensor with shape `(batch_size, q_num_heads, q_seq_len, v_head_size)`.",
           "T1"},
      },
      {
          {"T1", FlexAttentionFloatTypes(), "Constrain Q, K, V to float tensors."},
      },
      /*has_function_implementation=*/true);
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpPreviewSchemasWithHistory(const std::string &op_type,
                                                                 bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"FlexAttention", [] { return std::vector<LightOpSchema>{MakeFlexAttentionSchema()}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace preview
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

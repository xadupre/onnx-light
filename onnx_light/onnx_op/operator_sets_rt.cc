// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_rt.h"
#include "onnx_op/operator_sets_rt_doc.h"

#include <algorithm>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_op::rt {

namespace {

std::vector<TensorType> DelayedInitializerTensorTypes() {
  std::vector<TensorType> types = AllTensorTypes();
  types.erase(std::remove(types.begin(), types.end(), TensorType::kString), types.end());
  return types;
}

LightOpSchema MakeDelayedInitializerSchema() {
  return LightOpSchema(
      "DelayedInitializer", kAiRtDomain, 1, MakeDelayedInitializerDoc(), {},
      {
          {"output", "Tensor produced by the delayed initializer.", "T"},
      },
      {
          {"T", DelayedInitializerTensorTypes(),
           "Constrain output to tensor types backed by raw byte storage."},
      },
      {
          {"shape", "Shape of the output tensor.", AttributeType::INTS, /*required=*/true,
           std::monostate{}},
          {"dtype", "Element type of the output tensor, encoded as a TensorProto::DataType value.",
           AttributeType::INT, /*required=*/true, std::monostate{}},
          {"load_device", "Device where the initializer is first loaded.", AttributeType::STRING,
           /*required=*/true, std::monostate{}},
          {"runtime_device", "Device where the initializer is moved at runtime.",
           AttributeType::STRING, /*required=*/true, std::monostate{}},
          {"filename", "Filename containing the serialized tensor payload.", AttributeType::STRING,
           /*required=*/true, std::monostate{}},
          {"offset", "Byte offset of the tensor payload within `filename`.", AttributeType::INT,
           /*required=*/true, std::monostate{}},
      });
}

LightOpSchema MakeQuantizeSchema() {
  return LightOpSchema(
      "Quantize", kAiRtDomain, 1,
      "Encodes a finite floating tensor in the storage layout specified by type "
      "(a TYPE_PROTO attribute containing StructTypeProto). Optional inputs are scales, "
      "zero_points, offsets, codebooks, permutation, forward, inverse and outliers. "
      "Omitted scales are calibrated per block from the source range; cast blocks use one. "
      "Scalar numerical parameters broadcast, otherwise one value is required per block. "
      "Fixed scalar codebooks have profile defaults. Learned codebooks, vector/additive "
      "codebook scales and nonempty transforms/permutations/outlier indices must be supplied. "
      "This does not train GPTQ/AWQ/codebooks. Output preserves the source logical shape and "
      "dtype. With parameter_ref, a model-local fixed parameter set replaces all optional "
      "inputs and calibration; the output carries only local codes/outlier values in a compact "
      "storage schema plus the reference. Otherwise physical storage uses the requested type.",
      {{"X", "Floating tensor to encode.", "T"},
       {"scales", "Optional scalar or per-block scales.", "P1"},
       {"zero_points", "Optional scalar or per-block zero points.", "P2"},
       {"offsets", "Optional scalar or per-block offsets.", "P3"},
       {"codebooks", "Optional concatenated per-block codebooks.", "P4"},
       {"permutation", "Optional gather permutation.", "I"},
       {"forward", "Optional forward transform matrix.", "P5"},
       {"inverse", "Optional inverse transform matrix.", "P6"},
       {"outliers", "Optional original-source outlier indices.", "I"}},
      {{"Y", "EncodedValueProto carrying its local storage type and parameter ownership.", "E"}},
      {{"T",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""},
       {"P1",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""},
       {"P2",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""},
       {"P3",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""},
       {"P4",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""},
       {"P5",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""},
       {"P6",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""},
       {"I", {TensorType::kInt64}, ""},
       {"E", {TensorType::kStruct}, "Structured encoded value."}},
      {{"type", "Required quantization StructTypeProto wrapped in TypeProto.",
        AttributeType::TYPE_PROTO, true, std::monostate{}},
       {"parameter_ref", "Model-local fixed numerical parameter set; excludes optional inputs.",
        AttributeType::STRING, false, std::monostate{}}});
}

LightOpSchema MakeDequantizeSchema() {
  return LightOpSchema(
      "Dequantize", kAiRtDomain, 1,
      "Decodes a supported EncodedValueProto into the requested floating dtype without "
      "changing its logical shape. Rejects malformed layouts, unsupported dtypes, "
      "nonfinite reconstructions and output overflow. Supports the portable quantization "
      "and ORT MatMulNBits layouts, including model-catalogue type references.",
      {{"X", "Structured quantized value.", "E"}},
      {{"Y", "Decoded tensor in the requested dtype.", "T"}},
      {{"E", {TensorType::kStruct}, "Structured encoded value."},
       {"T",
        {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16},
        ""}},
      {{"dtype", "Required output TensorProto dtype: FLOAT, DOUBLE, FLOAT16 or BFLOAT16.",
        AttributeType::INT, true, std::monostate{}}});
}

LightOpSchema MakePagedAttentionSchema() {
  return LightOpSchema(
      "PagedAttention", kAiRtDomain, 1,
      "Appends immutable KV pages and computes grouped-query attention for finite Q/K/V tensors "
      "of shape [batch,heads,L,D]. Q may have a multiple of the K/V head count; K and V have "
      "equal head counts and all inputs have equal batch and new sequence dimensions. The "
      "past/present values are structured caches whose blocks form a dynamic sequence of pages "
      "with INT64 scalar start/length and logical key/value tensors "
      "[batch,kv_heads,capacity,head_size]. Runtime pages may store dense FLOAT/FLOAT16/BFLOAT16 "
      "or inline affine integer data. Shape inference produces Y with Q's batch, head and "
      "sequence dimensions, V's head size and the input floating-point type. Storage policy "
      "belongs to the registered kernel and is not part of the operator attributes.",
      {{"Q", "Queries [batch,q_heads,L,key_head_size].", "T"},
       {"K", "New keys [batch,kv_heads,L,key_head_size].", "T"},
       {"V", "New values [batch,kv_heads,L,value_head_size].", "T"},
       {"past", "Paged cache structured value.", "C"}},
      {{"Y", "Attention result [batch,q_heads,L,value_head_size].", "T"},
       {"present", "Cache after appending the new pages.", "C"}},
      {{"T",
        {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
        "Constrain Q/K/V and Y to one floating-point type."},
       {"C", {TensorType::kStruct}, "Named paged-cache structure."}},
      {{"block_size", "Positive maximum token capacity for each new page.", AttributeType::INT,
        false, int64_t(16)},
       {"max_tokens", "Positive maximum retained token count, including past and new tokens.",
        AttributeType::INT, false, int64_t(4096)},
       {"is_causal", "Whether to apply causal masking; only 0 or 1 is valid.", AttributeType::INT,
        false, int64_t(1)},
       {"left_window_size", "Past-token window; -1 is unbounded and non-negative values limit it.",
        AttributeType::INT, false, int64_t(-1)}});
}

LightOpSchema MakeQuantizePagedCacheSchema() {
  const std::vector<TensorType> storage_types{
      TensorType::kInt8,  TensorType::kUint8,   TensorType::kInt4,
      TensorType::kUint4, TensorType::kInt2,    TensorType::kUint2,
      TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16};
  return LightOpSchema(
      "QuantizePagedCache", kAiRtDomain, 1,
      "Quantizes the key and value payloads of selected immutable paged-cache blocks. "
      "block_indices contains unique zero-based block indices. key_scale and value_scale are "
      "positive scalar FLOAT tensors. The scalar key_zero_point and value_zero_point tensors "
      "select independent INT8, UINT8, INT4, UINT4, INT2, UINT2, FLOAT, FLOAT16 or BFLOAT16 "
      "storage formats. Integer types quantize to affine storage; floating types dequantize to "
      "dense storage. Selected dense or affine blocks are converted; unselected blocks retain "
      "their existing storage and ownership.",
      {{"cache", "Input paged-cache structured value.", "C"},
       {"block_indices", "Unique zero-based indices of blocks to quantize.", "I"},
       {"key_scale", "Scalar FLOAT key quantization scale.", "S"},
       {"key_zero_point", "Scalar key zero point selecting the key storage type.", "ZK"},
       {"value_scale", "Scalar FLOAT value quantization scale.", "S"},
       {"value_zero_point", "Scalar value zero point selecting the value storage type.", "ZV"}},
      {{"quantized_cache", "Cache with the selected blocks requantized.", "C"}},
      {{"C", {TensorType::kStruct}, "Constrain cache input and output to structured values."},
       {"I", {TensorType::kInt64}, "Constrain block indices to INT64."},
       {"S", {TensorType::kFloat}, "Constrain scales to FLOAT."},
       {"ZK", storage_types, "Select the key's affine integer or dense floating storage type."},
       {"ZV", storage_types, "Select the value's affine integer or dense floating storage type."}});
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpRtSchemasWithHistory(const std::string &op_type,
                                                            bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"DelayedInitializer",
       [] { return std::vector<LightOpSchema>{MakeDelayedInitializerSchema()}; }},
      {"Quantize", [] { return std::vector<LightOpSchema>{MakeQuantizeSchema()}; }},
      {"Dequantize", [] { return std::vector<LightOpSchema>{MakeDequantizeSchema()}; }},
      {"PagedAttention", [] { return std::vector<LightOpSchema>{MakePagedAttentionSchema()}; }},
      {"QuantizePagedCache",
       [] { return std::vector<LightOpSchema>{MakeQuantizePagedCacheSchema()}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::rt

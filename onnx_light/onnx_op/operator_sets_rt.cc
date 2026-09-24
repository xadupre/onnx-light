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
      "dtype; physical storage is described by the requested type.",
      {{"X", "Floating tensor to encode.", "T"},
       {"scales", "Optional scalar or per-block scales.", "P1"},
       {"zero_points", "Optional scalar or per-block zero points.", "P2"},
       {"offsets", "Optional scalar or per-block offsets.", "P3"},
       {"codebooks", "Optional concatenated per-block codebooks.", "P4"},
       {"permutation", "Optional gather permutation.", "I"},
       {"forward", "Optional forward transform matrix.", "P5"},
       {"inverse", "Optional inverse transform matrix.", "P6"},
       {"outliers", "Optional original-source outlier indices.", "I"}},
      {{"Y", "EncodedValueProto carrying the storage type and numerical parameters.", "E"}},
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
        AttributeType::TYPE_PROTO, true, std::monostate{}}});
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

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpRtSchemasWithHistory(const std::string &op_type,
                                                            bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"DelayedInitializer",
       [] { return std::vector<LightOpSchema>{MakeDelayedInitializerSchema()}; }},
      {"Quantize", [] { return std::vector<LightOpSchema>{MakeQuantizeSchema()}; }},
      {"Dequantize", [] { return std::vector<LightOpSchema>{MakeDequantizeSchema()}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::rt

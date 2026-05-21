// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_sequence.h"
#include "onnx_op/operator_sets_sequence_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace sequence {

namespace {

std::vector<TensorType> AllTensorSequenceTypes() {
  return {
      TensorType::kSeqUint8,  TensorType::kSeqUint16,    TensorType::kSeqUint32,
      TensorType::kSeqUint64, TensorType::kSeqInt8,      TensorType::kSeqInt16,
      TensorType::kSeqInt32,  TensorType::kSeqInt64,     TensorType::kSeqFloat16,
      TensorType::kSeqFloat,  TensorType::kSeqDouble,    TensorType::kSeqString,
      TensorType::kSeqBool,   TensorType::kSeqComplex64, TensorType::kSeqComplex128,
  };
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpSequenceSchemasWithHistory() {
  return std::vector<LightOpSchema>{
      LightOpSchema(
          "SequenceEmpty", kOnnxDomain, 11, MakeSequenceEmptyDoc(), {},
          {
              {"output", "Empty sequence.", "S"},
          },
          {
              {"S", AllTensorSequenceTypes(), "Constrain output types to any tensor type."},
          }),
      LightOpSchema(
          "SequenceLength", kOnnxDomain, 11, MakeSequenceLengthDoc(),
          {
              {"input_sequence", "Input sequence.", "S"},
          },
          {
              {"length", "Length of input sequence. It must be a scalar(tensor of empty shape).",
               "I"},
          },
          {
              {"S", AllTensorSequenceTypes(), "Constrain to any tensor type."},
              {"I",
               {TensorType::kInt64},
               "Constrain output to integral tensor. It must be a scalar(tensor of empty shape)."},
          }),
  };
}

} // namespace sequence
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

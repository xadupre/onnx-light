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

std::vector<TensorType> AllTensorOrSequenceTypes() {
  std::vector<TensorType> types = AllTensorTypes();
  const std::vector<TensorType> seq_types = AllTensorSequenceTypes();
  types.insert(types.end(), seq_types.begin(), seq_types.end());
  return types;
}

constexpr const char *kSequencePositionDescription =
    "Position of the tensor in the sequence. "
    "Negative value means counting positions from the back. "
    "Accepted range in `[-n, n - 1]`, "
    "where `n` is the number of tensors in 'input_sequence'. "
    "It is an error if any of the index values are out of bounds. "
    "It must be a scalar(tensor of empty shape).";

constexpr const char *kSequenceInsertPositionDescription =
    "Position in the sequence where the new tensor is inserted. "
    "It is optional and default is to insert to the back of the sequence. "
    "Negative value means counting positions from the back. "
    "Accepted range in `[-n, n]`, "
    "where `n` is the number of tensors in 'input_sequence'. "
    "It is an error if any of the index values are out of bounds. "
    "It must be a scalar(tensor of empty shape).";

constexpr const char *kIntegralPositionConstraint =
    "Constrain position to integral tensor. It must be a scalar(tensor of empty shape).";

constexpr const char *kSplitToSequenceSplitDescription =
    "Length of each output. "
    "It can be either a scalar(tensor of empty shape), or a 1-D tensor. All values must "
    "be >= 0. ";

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpSequenceSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      LightOpSchema(
          "ConcatFromSequence", kOnnxDomain, 11, MakeConcatFromSequenceDoc(),
          {
              {"input_sequence", "Sequence of tensors for concatenation", "S"},
          },
          {
              {"concat_result", "Concatenated tensor", "T"},
          },
          {
              {"S", AllTensorSequenceTypes(), "Constrain input types to any tensor type."},
              {"T", AllTensorTypes(), "Constrain output types to any tensor type."},
          },
          {
              AttributeParam{"axis",
                             "Which axis to concat on. Accepted range in `[-r, r - 1]`, "
                             "where `r` is the rank of input tensors. "
                             "When `new_axis` is 1, accepted range is `[-r - 1, r]`. ",
                             AttributeType::INT, /*required=*/true, std::monostate{}},
              AttributeParam{"new_axis",
                             "Insert and concatenate on a new axis or not, "
                             "default 0 means do not insert new axis.",
                             AttributeType::INT, /*required=*/false, int64_t{0}},
          }),
      LightOpSchema(
          "SequenceAt", kOnnxDomain, 11, MakeSequenceAtDoc(),
          {
              {"input_sequence", "Input sequence.", "S"},
              {"position", kSequencePositionDescription, "I"},
          },
          {
              {"tensor", "Output tensor at the specified position in the input sequence.", "T"},
          },
          {
              {"S", AllTensorSequenceTypes(), "Constrain to any tensor type."},
              {"T", AllTensorTypes(), "Constrain to any tensor type."},
              {"I", {TensorType::kInt32, TensorType::kInt64}, kIntegralPositionConstraint},
          }),
      LightOpSchema(
          "SequenceConstruct", kOnnxDomain, 11, MakeSequenceConstructDoc(),
          {
              {"inputs", "Tensors.", "T"},
          },
          {
              {"output_sequence", "Sequence enclosing the input tensors.", "S"},
          },
          {
              {"T", AllTensorTypes(), "Constrain input types to any tensor type."},
              {"S", AllTensorSequenceTypes(), "Constrain output types to any tensor type."},
          }),
      LightOpSchema(
          "SequenceEmpty", kOnnxDomain, 11, MakeSequenceEmptyDoc(), {},
          {
              {"output", "Empty sequence.", "S"},
          },
          {
              {"S", AllTensorSequenceTypes(), "Constrain output types to any tensor type."},
          }),
      LightOpSchema(
          "SequenceErase", kOnnxDomain, 11, MakeSequenceEraseDoc(),
          {
              {"input_sequence", "Input sequence.", "S"},
              {"position", kSequencePositionDescription, "I"},
          },
          {
              {"output_sequence",
               "Output sequence that has the tensor at the specified position removed.", "S"},
          },
          {
              {"S", AllTensorSequenceTypes(), "Constrain to any tensor type."},
              {"I", {TensorType::kInt32, TensorType::kInt64}, kIntegralPositionConstraint},
          }),
      LightOpSchema(
          "SequenceInsert", kOnnxDomain, 11, MakeSequenceInsertDoc(),
          {
              {"input_sequence", "Input sequence.", "S"},
              {"tensor", "Input tensor to be inserted into the input sequence.", "T"},
              {"position", kSequenceInsertPositionDescription, "I"},
          },
          {
              {"output_sequence",
               "Output sequence that contains the inserted tensor at given position.", "S"},
          },
          {
              {"T", AllTensorTypes(), "Constrain to any tensor type."},
              {"S", AllTensorSequenceTypes(), "Constrain to any tensor type."},
              {"I", {TensorType::kInt32, TensorType::kInt64}, kIntegralPositionConstraint},
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
      LightOpSchema(
          "SequenceMap", kOnnxDomain, 17, MakeSequenceMapDoc(),
          {
              {"input_sequence", "Input sequence.", "S"},
              {"additional_inputs", "Additional inputs to the graph", "V"},
          },
          {
              {"out_sequence", "Output sequence(s)", "S"},
          },
          {
              {"S", AllTensorSequenceTypes(), "Constrain input types to any sequence type."},
              {"V", AllTensorOrSequenceTypes(), "Constrain to any tensor or sequence type."},
          },
          /*has_function_implementation=*/true),
      LightOpSchema(
          "SplitToSequence", kOnnxDomain, 11, MakeSplitToSequenceDoc(),
          {
              {"input", "The tensor to split", "T"},
              {"split", kSplitToSequenceSplitDescription, "I"},
          },
          {
              {"output_sequence",
               "One or more outputs forming a sequence of tensors after splitting", "S"},
          },
          {
              {"T", AllTensorTypes(), "Constrain input types to all tensor types."},
              {"I",
               {TensorType::kInt32, TensorType::kInt64},
               "Constrain split size to integral tensor."},
              {"S", AllTensorSequenceTypes(), "Constrain output types to all tensor types."},
          }),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace sequence
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

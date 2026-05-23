// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_nn.h"
#include "onnx_op/operator_sets_nn_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace nn {

namespace {

const char *const kPoolingInputDescription =
    "Input data tensor from the previous operator; "
    "dimensions for image case are (N x C x H x W), "
    "where N is the batch size, C is the number of "
    "channels, and H and W are the height and the "
    "width of the data. For non image case, the "
    "dimensions are in the form of "
    "(N x C x D1 x D2 ... Dn), where N is the batch "
    "size. Optionally, if dimension denotation is "
    "in effect, the operation expects the input "
    "data tensor to arrive with the dimension denotation "
    "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].";

const char *const kPoolingOutputDescription =
    "Output data tensor from average or max pooling across "
    "the input tensor. Dimensions will vary based "
    "on various kernel, stride, and pad sizes. Floor value of "
    "the dimension is used";

std::vector<TensorType> AveragePoolTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeAveragePoolSchema(int since_version) {
  return LightOpSchema("AveragePool", kOnnxDomain, since_version, MakeAveragePoolDoc(since_version),
                       {
                           {"X", kPoolingInputDescription, "T"},
                       },
                       {
                           {"Y", kPoolingOutputDescription, "T"},
                       },
                       {
                           {"T", AveragePoolTypes(since_version),
                            "Constrain input and output types to float tensors."},
                       });
}

// --- Recurrent operators (RNN, GRU, LSTM) ------------------------------------

// Inputs/outputs and type-constraint descriptions are reproduced verbatim from
// upstream ONNX so that the LightOpSchema parity test in test_onnx_ops.cc
// passes.

const char *const kRecurrentXDescription =
    "The input sequences packed (and potentially padded) into one 3-D "
    "tensor with the shape of `[seq_length, batch_size, input_size]`.";

const char *const kRecurrentSequenceLensDescription =
    "Optional tensor specifying lengths of the sequences in a batch. "
    "If not specified - assumed all sequences in the batch to have "
    "length `seq_length`. It has shape `[batch_size]`.";

const char *const kRecurrentInitialHDescription =
    "Optional initial value of the hidden. If not specified - assumed "
    "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.";

const char *const kRecurrentYDescriptionVer1 =
    "A tensor that concats all the intermediate output values of the hidden. "
    "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. "
    "It is optional if `output_sequence` is 0.";

const char *const kRecurrentYDescriptionVer7 =
    "A tensor that concats all the intermediate output values of the hidden. "
    "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ";

const char *const kRecurrentYhDescription = "The last output value of the hidden. It has shape "
                                            "`[num_directions, batch_size, hidden_size]`.";

// RNN-specific descriptions
const char *const kRNNWDescription =
    "The weight tensor for input gate. Concatenation of `Wi` and `WBi` "
    "(if bidirectional). The tensor has shape "
    "`[num_directions, hidden_size, input_size]`.";

const char *const kRNNRDescription =
    "The recurrence weight tensor. Concatenation of `Ri` and `RBi` "
    "(if bidirectional). The tensor has shape "
    "`[num_directions, hidden_size, hidden_size]`.";

const char *const kRNNBDescription =
    "The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` "
    "and `[WBbi, RBbi]` (if bidirectional). The tensor has shape "
    "`[num_directions, 2*hidden_size]`. Optional: If not specified - assumed "
    "to be 0.";

// GRU-specific descriptions
const char *const kGRUWDescription =
    "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
    "(if bidirectional) along dimension 0. This tensor has shape "
    "`[num_directions, 3*hidden_size, input_size]`.";

const char *const kGRURDescription =
    "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
    "(if bidirectional) along dimension 0. This tensor has shape "
    "`[num_directions, 3*hidden_size, hidden_size]`.";

const char *const kGRUBDescription =
    "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
    "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
    "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
    "- assumed to be 0";

// LSTM-specific descriptions
const char *const kLSTMWDescription =
    "The weight tensor for the gates. Concatenation of `W[iofc]` and "
    "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
    "`[num_directions, 4*hidden_size, input_size]`.";

const char *const kLSTMRDescription =
    "The recurrence weight tensor. Concatenation of `R[iofc]` and "
    "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
    "`[num_directions, 4*hidden_size, hidden_size]`.";

const char *const kLSTMBDescription =
    "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
    "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
    "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
    "specified - assumed to be 0.";

const char *const kLSTMInitialCDescription =
    "Optional initial value of the cell. If not specified - assumed "
    "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.";

const char *const kLSTMPDescription =
    "The weight tensor for peepholes. Concatenation of `P[iof]` and "
    "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
    "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
    "assumed to be 0.";

const char *const kLSTMYcDescription = "The last output value of the cell. It has shape "
                                       "`[num_directions, batch_size, hidden_size]`.";

std::vector<TensorType> RecurrentTTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

std::vector<TypeConstraintParam> RecurrentTypeConstraints(int since_version) {
  return {
      {"T", RecurrentTTypes(since_version), "Constrain input and output types to float tensors."},
      {"T1", {TensorType::kInt32}, "Constrain seq_lens to integer tensor."},
  };
}

const char *RecurrentYDescription(int since_version) {
  // RNN/LSTM v1 and GRU v1/v3 use the older Y description that mentions the
  // deprecated `output_sequence` attribute. The opset1_to_6 RNN doc generator
  // is shared across both, so all versions <= 6 receive the v1 description.
  return since_version <= 6 ? kRecurrentYDescriptionVer1 : kRecurrentYDescriptionVer7;
}

LightOpSchema MakeRNNSchema(int since_version) {
  return LightOpSchema("RNN", kOnnxDomain, since_version, MakeRNNDoc(since_version),
                       {
                           {"X", kRecurrentXDescription, "T"},
                           {"W", kRNNWDescription, "T"},
                           {"R", kRNNRDescription, "T"},
                           {"B", kRNNBDescription, "T"},
                           {"sequence_lens", kRecurrentSequenceLensDescription, "T1"},
                           {"initial_h", kRecurrentInitialHDescription, "T"},
                       },
                       {
                           {"Y", RecurrentYDescription(since_version), "T"},
                           {"Y_h", kRecurrentYhDescription, "T"},
                       },
                       RecurrentTypeConstraints(since_version));
}

LightOpSchema MakeGRUSchema(int since_version) {
  return LightOpSchema("GRU", kOnnxDomain, since_version, MakeGRUDoc(since_version),
                       {
                           {"X", kRecurrentXDescription, "T"},
                           {"W", kGRUWDescription, "T"},
                           {"R", kGRURDescription, "T"},
                           {"B", kGRUBDescription, "T"},
                           {"sequence_lens", kRecurrentSequenceLensDescription, "T1"},
                           {"initial_h", kRecurrentInitialHDescription, "T"},
                       },
                       {
                           {"Y", RecurrentYDescription(since_version), "T"},
                           {"Y_h", kRecurrentYhDescription, "T"},
                       },
                       RecurrentTypeConstraints(since_version));
}

LightOpSchema MakeLSTMSchema(int since_version) {
  return LightOpSchema("LSTM", kOnnxDomain, since_version, MakeLSTMDoc(since_version),
                       {
                           {"X", kRecurrentXDescription, "T"},
                           {"W", kLSTMWDescription, "T"},
                           {"R", kLSTMRDescription, "T"},
                           {"B", kLSTMBDescription, "T"},
                           {"sequence_lens", kRecurrentSequenceLensDescription, "T1"},
                           {"initial_h", kRecurrentInitialHDescription, "T"},
                           {"initial_c", kLSTMInitialCDescription, "T"},
                           {"P", kLSTMPDescription, "T"},
                       },
                       {
                           {"Y", RecurrentYDescription(since_version), "T"},
                           {"Y_h", kRecurrentYhDescription, "T"},
                           {"Y_c", kLSTMYcDescription, "T"},
                       },
                       RecurrentTypeConstraints(since_version));
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpNnSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      MakeAveragePoolSchema(22), MakeAveragePoolSchema(19), MakeAveragePoolSchema(11),
      MakeAveragePoolSchema(10), MakeAveragePoolSchema(7),  MakeAveragePoolSchema(1),
      MakeGRUSchema(22),         MakeGRUSchema(14),         MakeGRUSchema(7),
      MakeGRUSchema(3),          MakeGRUSchema(1),          MakeLSTMSchema(22),
      MakeLSTMSchema(14),        MakeLSTMSchema(7),         MakeLSTMSchema(1),
      MakeRNNSchema(22),         MakeRNNSchema(14),         MakeRNNSchema(7),
      MakeRNNSchema(1),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

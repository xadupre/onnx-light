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

// --- BatchNormalization -----------------------------------------------------

// Input/output and type-constraint descriptions reproduced verbatim from
// the upstream ONNX schemas so the LightOpSchema parity test passes.

const char *const kBNXDescriptionVer1 = "The input 4-dimensional tensor of shape NCHW.";

const char *const kBNXDescriptionVer6 = "Input data tensor from the previous operator; "
                                        "dimensions for image case are (N x C x H x W), "
                                        "where N is the batch size, C is the number of "
                                        "channels, and H and W are the height and the "
                                        "width of the data. For non image case, the "
                                        "dimensions are in the form of "
                                        "(N x C x D1 x D2 ... Dn), where N is the batch "
                                        "size.";

const char *const kBNXDescriptionVer9 =
    "Input data tensor from the previous operator; "
    "dimensions are in the form of (N x C x D1 x D2 ... Dn), "
    "where N is the batch size, C is the number of channels. "
    "Statistics are computed for every channel of C over N and D1 to Dn dimensions. "
    "For image data, input dimensions become (N x C x H x W). "
    "The op also accepts single dimension input of size N in which case C is assumed to be 1";

const char *const kBNScaleDescriptionVer1 =
    "The scale as a 1-dimensional tensor of size C to be applied to the output.";

const char *const kBNBiasDescriptionVer1 =
    "The bias as a 1-dimensional tensor of size C to be applied to the output.";

const char *const kBNMeanDescriptionVer1 =
    "The running mean (training) or the estimated mean (testing) "
    "as a 1-dimensional tensor of size C.";

const char *const kBNVarDescriptionVer1 = "The running variance (training) or the estimated "
                                          "variance (testing) as a 1-dimensional tensor of size C.";

const char *const kBNScaleDescriptionVer7 = "If spatial is true, the dimension of scale is (C). "
                                            "If spatial is false, the dimensions of scale are "
                                            "(C x D1 x ... x Dn)";

const char *const kBNBiasDescriptionVer7 = "If spatial is true, the dimension of bias is (C). "
                                           "If spatial is false, the dimensions of bias are "
                                           "(C x D1 x ... x Dn)";

const char *const kBNMeanDescriptionVer7 =
    "If spatial is true, the dimension of the running mean "
    "(training) or the estimated mean (testing) is (C). "
    "If spatial is false, the dimensions of the running mean "
    "(training) or the estimated mean (testing) are (C x D1 x ... x Dn).";

const char *const kBNVarDescriptionVer7 =
    "If spatial is true, the dimension of the running variance"
    "(training) or the estimated variance (testing) is (C). "
    "If spatial is false, the dimensions of the running variance"
    "(training) or the estimated variance (testing) are (C x D1 x ... x Dn).";

const char *const kBNScaleDescriptionVer14 = "Scale tensor of shape (C).";
const char *const kBNBiasDescriptionVer14 = "Bias tensor of shape (C).";

const char *const kBNInputMeanDescriptionVer14 =
    "running (training) or estimated (testing) mean tensor of shape (C).";

const char *const kBNInputVarDescriptionVer14 =
    "running (training) or estimated (testing) variance tensor of shape (C).";

const char *const kBNYDescriptionVer1 = "The output 4-dimensional tensor of the same shape as X.";
const char *const kBNYDescriptionVer6 = "The output tensor of the same shape as X.";
const char *const kBNYDescriptionVer7 = "The output tensor of the same shape as X";

const char *const kBNMeanOutputDescriptionVer1 =
    "The running mean after the BatchNormalization operator. Must be in-place "
    "with the input mean. Should not be used for testing.";

const char *const kBNVarOutputDescriptionVer1 =
    "The running variance after the BatchNormalization operator. Must be "
    "in-place with the input var. Should not be used for testing.";

const char *const kBNSavedMeanDescriptionVer1 =
    "Saved mean used during training to speed up gradient "
    "computation. Should not be used for testing.";

const char *const kBNSavedVarDescriptionVer1 =
    "Saved variance used during training to speed up "
    "gradient computation. Should not be used for testing.";

const char *const kBNMeanOutputDescriptionVer7 =
    "The running mean after the BatchNormalization operator.";

const char *const kBNVarOutputDescriptionVer7 =
    "The running variance after the BatchNormalization operator.";

const char *const kBNSavedMeanDescriptionVer7 =
    "Saved mean used during training to speed up gradient "
    "computation.";

const char *const kBNSavedVarDescriptionVer7 = "Saved variance used during training to speed up "
                                               "gradient computation.";

const char *const kBNRunningVarDescriptionVer14 =
    "The running variance after the BatchNormalization operator. This op uses the "
    "population size (N) for "
    "calculating variance, and not the sample size N-1.";

const char *const kBNTConstraintDescription = "Constrain input and output types to float tensors.";

const char *const kBNUConstraintDescription =
    "Constrain mean and variance types to float tensors. It allows all float type for U.";

const char *const kBNT1ConstraintDescription = "Constrain scale and bias types to float tensors.";
const char *const kBNT2ConstraintDescription =
    "Constrain mean and variance types to float tensors.";

std::vector<TensorType> BatchNormalizationFloatTypes(int since_version) {
  if (since_version >= 14) {
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeBatchNormalizationSchema(int since_version) {
  if (since_version == 1) {
    return LightOpSchema(
        "BatchNormalization", kOnnxDomain, since_version, MakeBatchNormalizationDoc(since_version),
        {
            {"X", kBNXDescriptionVer1, "T"},
            {"scale", kBNScaleDescriptionVer1, "T"},
            {"B", kBNBiasDescriptionVer1, "T"},
            {"mean", kBNMeanDescriptionVer1, "T"},
            {"var", kBNVarDescriptionVer1, "T"},
        },
        {
            {"Y", kBNYDescriptionVer1, "T"},
            {"mean", kBNMeanOutputDescriptionVer1, "T"},
            {"var", kBNVarOutputDescriptionVer1, "T"},
            {"saved_mean", kBNSavedMeanDescriptionVer1, "T"},
            {"saved_var", kBNSavedVarDescriptionVer1, "T"},
        },
        {
            {"T", BatchNormalizationFloatTypes(since_version), kBNTConstraintDescription},
        });
  }
  if (since_version == 6) {
    return LightOpSchema(
        "BatchNormalization", kOnnxDomain, since_version, MakeBatchNormalizationDoc(since_version),
        {
            {"X", kBNXDescriptionVer6, "T"},
            {"scale", kBNScaleDescriptionVer1, "T"},
            {"B", kBNBiasDescriptionVer1, "T"},
            {"mean", kBNMeanDescriptionVer1, "T"},
            {"var", kBNVarDescriptionVer1, "T"},
        },
        {
            {"Y", kBNYDescriptionVer6, "T"},
            {"mean", kBNMeanOutputDescriptionVer1, "T"},
            {"var", kBNVarOutputDescriptionVer1, "T"},
            {"saved_mean", kBNSavedMeanDescriptionVer1, "T"},
            {"saved_var", kBNSavedVarDescriptionVer1, "T"},
        },
        {
            {"T", BatchNormalizationFloatTypes(since_version), kBNTConstraintDescription},
        });
  }
  if (since_version == 7) {
    return LightOpSchema(
        "BatchNormalization", kOnnxDomain, since_version, MakeBatchNormalizationDoc(since_version),
        {
            {"X", kBNXDescriptionVer6, "T"},
            {"scale", kBNScaleDescriptionVer7, "T"},
            {"B", kBNBiasDescriptionVer7, "T"},
            {"mean", kBNMeanDescriptionVer7, "T"},
            {"var", kBNVarDescriptionVer7, "T"},
        },
        {
            {"Y", kBNYDescriptionVer7, "T"},
            {"mean", kBNMeanOutputDescriptionVer7, "T"},
            {"var", kBNVarOutputDescriptionVer7, "T"},
            {"saved_mean", kBNSavedMeanDescriptionVer7, "T"},
            {"saved_var", kBNSavedVarDescriptionVer7, "T"},
        },
        {
            {"T", BatchNormalizationFloatTypes(since_version), kBNTConstraintDescription},
        });
  }
  if (since_version == 9) {
    return LightOpSchema(
        "BatchNormalization", kOnnxDomain, since_version, MakeBatchNormalizationDoc(since_version),
        {
            {"X", kBNXDescriptionVer9, "T"},
            {"scale", kBNScaleDescriptionVer14, "T"},
            {"B", kBNBiasDescriptionVer14, "T"},
            {"mean", kBNInputMeanDescriptionVer14, "T"},
            {"var", kBNInputVarDescriptionVer14, "T"},
        },
        {
            {"Y", kBNYDescriptionVer7, "T"},
            {"mean", kBNMeanOutputDescriptionVer7, "T"},
            {"var", kBNVarOutputDescriptionVer7, "T"},
            {"saved_mean", kBNSavedMeanDescriptionVer7, "T"},
            {"saved_var", kBNSavedVarDescriptionVer7, "T"},
        },
        {
            {"T", BatchNormalizationFloatTypes(since_version), kBNTConstraintDescription},
        });
  }
  if (since_version == 14) {
    return LightOpSchema(
        "BatchNormalization", kOnnxDomain, since_version, MakeBatchNormalizationDoc(since_version),
        {
            {"X", kBNXDescriptionVer9, "T"},
            {"scale", kBNScaleDescriptionVer14, "T"},
            {"B", kBNBiasDescriptionVer14, "T"},
            {"input_mean", kBNInputMeanDescriptionVer14, "U"},
            {"input_var", kBNInputVarDescriptionVer14, "U"},
        },
        {
            {"Y", kBNYDescriptionVer7, "T"},
            {"running_mean", kBNMeanOutputDescriptionVer7, "U"},
            {"running_var", kBNRunningVarDescriptionVer14, "U"},
        },
        {
            {"T", BatchNormalizationFloatTypes(since_version), kBNTConstraintDescription},
            {"U", BatchNormalizationFloatTypes(since_version), kBNUConstraintDescription},
        });
  }
  // since_version == 15
  return LightOpSchema(
      "BatchNormalization", kOnnxDomain, since_version, MakeBatchNormalizationDoc(since_version),
      {
          {"X", kBNXDescriptionVer9, "T"},
          {"scale", kBNScaleDescriptionVer14, "T1"},
          {"B", kBNBiasDescriptionVer14, "T1"},
          {"input_mean", kBNInputMeanDescriptionVer14, "T2"},
          {"input_var", kBNInputVarDescriptionVer14, "T2"},
      },
      {
          {"Y", kBNYDescriptionVer7, "T"},
          {"running_mean", kBNMeanOutputDescriptionVer7, "T2"},
          {"running_var", kBNRunningVarDescriptionVer14, "T2"},
      },
      {
          {"T", BatchNormalizationFloatTypes(since_version), kBNTConstraintDescription},
          {"T1", BatchNormalizationFloatTypes(since_version), kBNT1ConstraintDescription},
          {"T2", BatchNormalizationFloatTypes(since_version), kBNT2ConstraintDescription},
      });
}

// --- Attention --------------------------------------------------------------

// Input/output descriptions reproduced verbatim from the upstream ONNX
// Attention schemas (v23 in onnx_lib/defs/nn/old.cc and v24 in
// onnx_lib/defs/nn/defs.cc) so the LightOpSchema parity test passes.

const char *const kAttentionQDescription =
    "Query tensor. "
    "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, head_size)` or "
    "3D tensor with shape `(batch_size, q_sequence_length, q_hidden_size)`. "
    "For cases with a 3D input tensor, `q_hidden_size = q_num_heads * head_size`";

const char *const kAttentionKDescription =
    "Key tensor. "
    "4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, head_size)` "
    "or 3D tensor with shape `(batch_size, kv_sequence_length, k_hidden_size)`. "
    "For cases with a 3D input tensor, `k_hidden_size = kv_num_heads * head_size`";

const char *const kAttentionVDescription =
    "Value tensor. "
    "4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)` "
    "or 3D tensor with shape `(batch_size, kv_sequence_length, v_hidden_size)`. "
    "For cases with a 3D input tensor, `v_hidden_size = kv_num_heads * v_head_size`";

const char *const kAttentionAttnMaskDescriptionVer23 =
    "Attention mask. "
    "Shape must be broadcastable to "
    "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, "
    "total_sequence_length)` "
    "where `total_sequence_length = past_sequence_length + kv_sequence_length.` "
    "Two types of masks are supported. A boolean mask where a value of `True` indicates "
    "that the element should take part in attention. "
    "Also supports a float mask of the same type as query, key, value that is added to "
    "the attention score.";

const char *const kAttentionAttnMaskDescriptionVer24 =
    "Attention mask. "
    "Shape must be broadcastable to `(batch_size, q_num_heads, q_sequence_length, "
    "total_sequence_length)` "
    "where `total_sequence_length = past_sequence_length + kv_sequence_length.` "
    "The last dimension can also be shorter than `total_sequence_length` and will be "
    "padded to `total_sequence_length` with negative infinity. "
    "Two types of masks are supported: a boolean mask where a value of `True` indicates "
    "that the element should take part in attention, "
    "or a float mask of the same type as query, key, value that is added to the "
    "attention score.";

const char *const kAttentionPastKeyDescription =
    "past state cache for key with shape `(batch_size, kv_num_heads, "
    "past_sequence_length, head_size)`";

const char *const kAttentionPastValueDescription =
    "past state cache for value with shape `(batch_size, kv_num_heads, "
    "past_sequence_length, v_head_size)`";

const char *const kAttentionNonpadKvSeqlenDescription =
    "A vector of integers of shape `(batch_size,)` that indicates the number of valid "
    "(ie, non-padding) "
    "tokens in each sample. A padding mask can be derived from this. This should not be "
    "used together with "
    "`past_key` and `past_value` inputs or `present_key` and `present_value` outputs "
    "(See the KV cache use cases in the operator description).";

const char *const kAttentionYDescription =
    "The output tensor . "
    "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` "
    "or 3D tensor with shape `(batch_size, q_sequence_length, hidden_size)`. "
    "For cases with a 3D input tensor, `hidden_size = q_num_heads * v_head_size`";

const char *const kAttentionPresentKeyDescription =
    "Updated key cache with shape `(batch_size, kv_num_heads, total_sequence_length, "
    "head_size)` "
    "where `total_sequence_length = past_sequence_length + kv_sequence_length`.";

const char *const kAttentionPresentValueDescription =
    "Updated value cache with shape `(batch_size, kv_num_heads, total_sequence_length, "
    "v_head_size)` "
    "where `total_sequence_length = past_sequence_length + kv_sequence_length`.";

const char *const kAttentionQkMatmulOutputDescription =
    "The output of QK matmul. "
    "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, "
    "total_sequence_length)` "
    "where `total_sequence_length = past_sequence_length + kv_sequence_length`.";

std::vector<TensorType> AttentionFloatTypes() {
  // Mirrors OpSchema::all_float_types_ir4() ordering used by upstream.
  return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

std::vector<TensorType> AttentionMaskTypes() {
  // Mirrors OpSchema::all_non_complex_numeric_types_plus_bool_ir4() used by
  // upstream for the "U" constraint on attn_mask.
  return {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
      TensorType::kBool,
  };
}

LightOpSchema MakeAttentionSchema(int since_version) {
  std::vector<FormalParameter> inputs = {
      {"Q", kAttentionQDescription, "T1"},
      {"K", kAttentionKDescription, "T1"},
      {"V", kAttentionVDescription, "T2"},
      {"attn_mask",
       since_version >= 24 ? kAttentionAttnMaskDescriptionVer24
                           : kAttentionAttnMaskDescriptionVer23,
       "U"},
      {"past_key", kAttentionPastKeyDescription, "T1"},
      {"past_value", kAttentionPastValueDescription, "T2"},
  };
  if (since_version >= 24) {
    inputs.push_back({"nonpad_kv_seqlen", kAttentionNonpadKvSeqlenDescription, "tensor(int64)"});
  }
  std::vector<FormalParameter> outputs = {
      {"Y", kAttentionYDescription, "T1"},
      {"present_key", kAttentionPresentKeyDescription, "T1"},
      {"present_value", kAttentionPresentValueDescription, "T2"},
      {"qk_matmul_output", kAttentionQkMatmulOutputDescription, "T1"},
  };
  return LightOpSchema(
      "Attention", kOnnxDomain, since_version, MakeAttentionDoc(since_version), std::move(inputs),
      std::move(outputs),
      {
          {"T1", AttentionFloatTypes(), "Constrain Q and K inputs types to float tensors."},
          {"T2", AttentionFloatTypes(), "Constrain V input types to float tensors."},
          {"U", AttentionMaskTypes(),
           "Constrain output 'mask' types to boolean tensors and input types."},
      },
      /*has_function_implementation=*/true);
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpNnSchemasWithHistory(const std::string &op_type,
                                                            bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"Attention",
       [] {
         return std::vector<LightOpSchema>{
             MakeAttentionSchema(24),
             MakeAttentionSchema(23),
         };
       }},
      {"AveragePool",
       [] {
         return std::vector<LightOpSchema>{
             MakeAveragePoolSchema(22), MakeAveragePoolSchema(19), MakeAveragePoolSchema(11),
             MakeAveragePoolSchema(10), MakeAveragePoolSchema(7),  MakeAveragePoolSchema(1),
         };
       }},
      {"BatchNormalization",
       [] {
         return std::vector<LightOpSchema>{
             MakeBatchNormalizationSchema(15), MakeBatchNormalizationSchema(14),
             MakeBatchNormalizationSchema(9),  MakeBatchNormalizationSchema(7),
             MakeBatchNormalizationSchema(6),  MakeBatchNormalizationSchema(1),
         };
       }},
      {"GRU",
       [] {
         return std::vector<LightOpSchema>{
             MakeGRUSchema(22), MakeGRUSchema(14), MakeGRUSchema(7),
             MakeGRUSchema(3),  MakeGRUSchema(1),
         };
       }},
      {"LSTM",
       [] {
         return std::vector<LightOpSchema>{
             MakeLSTMSchema(22),
             MakeLSTMSchema(14),
             MakeLSTMSchema(7),
             MakeLSTMSchema(1),
         };
       }},
      {"RNN",
       [] {
         return std::vector<LightOpSchema>{
             MakeRNNSchema(22),
             MakeRNNSchema(14),
             MakeRNNSchema(7),
             MakeRNNSchema(1),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

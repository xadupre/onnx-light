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

const char *const kMaxPoolIndicesDescription =
    "Indices tensor from max pooling across the input tensor. "
    "The dimensions of indices are the same as output tensor. "
    "The values in indices of are the indices of the selected values during pooling. "
    "The indices are computed as flatten 1-D tensor, "
    "and the indices do not consider padding. "
    "So the values in indices are in [0, N x C x D1 x ... x Dn).";

std::vector<TensorType> MaxPoolTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,
            TensorType::kDouble,   TensorType::kInt8,    TensorType::kUint8};
  }
  if (since_version >= 12) {
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kInt8,
            TensorType::kUint8};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeMaxPoolSchema(int since_version) {
  // v1 has only the primary output Y; v8+ adds the optional Indices output.
  std::vector<FormalParameter> outputs = {
      {"Y", kPoolingOutputDescription, "T"},
  };
  std::vector<TypeConstraintParam> tcs = {
      {"T", MaxPoolTypes(since_version),
       since_version >= 12 ? "Constrain input and output types to float and 8 bit tensors."
                           : "Constrain input and output types to float tensors."},
  };
  if (since_version >= 8) {
    outputs.push_back({"Indices", kMaxPoolIndicesDescription, "I"});
    tcs.push_back({"I", {TensorType::kInt64}, "Constrain index tensor to int64"});
  }
  return LightOpSchema("MaxPool", kOnnxDomain, since_version, MakeMaxPoolDoc(since_version),
                       {
                           {"X", kPoolingInputDescription, "T"},
                       },
                       std::move(outputs), std::move(tcs));
}

const char *const kMaxUnpoolXDescription =
    "Input data tensor that has to be unpooled. "
    "This tensor is typically the first output of the MaxPool op."
    "Dimensions for image case are (N x C x H x W), "
    "where N is the batch size, C is the number of "
    "channels, and H and W are the height and the "
    "width of the data. For non-image case, the "
    "dimensions are in the form of "
    "(N x C x D1 x D2 ... Dn), where N is the batch "
    "size. Optionally, if dimension denotation is "
    "in effect, the operation expects the input "
    "data tensor to arrive with the dimension denotation "
    "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].";

const char *const kMaxUnpoolIDescription =
    "Input data tensor containing the indices corresponding to "
    "elements in the first input tensor X."
    "This tensor is typically the second output of the MaxPool op."
    "Dimensions must be the same as input tensor X. "
    "The indices are linear, i.e. computed considering the tensor as flattened 1-D tensor, "
    "assuming row-major storage. Also, the linear indices should not consider padding. "
    "So the values in indices are in the range [0, N x C x D1 x ... x Dn).";

const char *const kMaxUnpoolOutputShapeDescription =
    "The shape of the output can be explicitly set which will cause pads values to be "
    "auto generated. If 'output_shape' is specified, "
    "'pads' values are ignored.";

const char *const kMaxUnpoolOutputDescription =
    "Output data tensor that contains the result of the unpooling.";

std::vector<TensorType> MaxUnpoolFloatTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeMaxUnpoolSchema(int since_version) {
  return LightOpSchema("MaxUnpool", kOnnxDomain, since_version, MakeMaxUnpoolDoc(since_version),
                       {
                           {"X", kMaxUnpoolXDescription, "T1"},
                           {"I", kMaxUnpoolIDescription, "T2"},
                           {"output_shape", kMaxUnpoolOutputShapeDescription, "T2"},
                       },
                       {
                           {"output", kMaxUnpoolOutputDescription, "T1"},
                       },
                       {
                           {"T1", MaxUnpoolFloatTypes(since_version),
                            "Constrain input and output types to float tensors."},
                           {"T2", {TensorType::kInt64}, "Constrain index tensor to int64"},
                       });
}

// --- MaxRoiPool --------------------------------------------------------------

std::vector<TensorType> MaxRoiPoolFloatTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeMaxRoiPoolSchema(int since_version) {
  return LightOpSchema(
      "MaxRoiPool", kOnnxDomain, since_version, MakeMaxRoiPoolDoc(since_version),
      {
          {"X",
           "Input data tensor from the previous operator; "
           "dimensions for image case are (N x C x H x W), "
           "where N is the batch size, C is the number of "
           "channels, and H and W are the height and the "
           "width of the data.",
           "T"},
          {"rois",
           "RoIs (Regions of Interest) to pool over. Should "
           "be a 2-D tensor of shape (num_rois, 5) given as "
           "[[batch_id, x1, y1, x2, y2], ...].",
           "T"},
      },
      {
          {"Y",
           "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], "
           "pooled_shape[1]).",
           "T"},
      },
      {
          {"T", MaxRoiPoolFloatTypes(since_version),
           "Constrain input and output types to float tensors."},
      },
      {
          {"pooled_shape", "ROI pool output shape (height, width).", AttributeType::INTS,
           /*required=*/true, std::monostate{}},
          {"spatial_scale",
           "Multiplicative spatial scale factor to translate ROI coordinates from their input "
           "scale to the scale used when pooling.",
           AttributeType::FLOAT, /*required=*/false, 1.0f},
      });
}

const char *const kDropoutDataDescription = "The input data as Tensor.";
const char *const kDropoutOutputDescription = "The output.";
const char *const kDropoutMaskDescriptionVer1And6 =
    "The output mask. If is_test is nonzero, this output is not filled.";
const char *const kDropoutMaskDescription = "The output mask.";
const char *const kDropoutRatioDescriptionVer12And13 =
    "The ratio of random dropout, with value in [0, 1). If this input was not set, "
    "or if it was set to 0, the output would be a simple copy of the input. "
    "If it's non-zero, output will be a random dropout of the scaled input, which is "
    "typically "
    "the case during training. It is an optional value, if not specified it will "
    "default to 0.5.";
const char *const kDropoutRatioDescriptionVer22 =
    "The ratio of random dropout, with value in [0, 1). If set to 0, "
    "the output would be a simple copy of the input. "
    "If it's non-zero, output will be a random dropout of the scaled input, which is "
    "typically "
    "the case during training. It is an optional value, if not specified it will "
    "default to 0.5.";
const char *const kDropoutTrainingModeDescription =
    "If set to true then it indicates dropout is being used for training. It is an "
    "optional value hence unless "
    "specified explicitly, it is false. If it is false, ratio is ignored and the "
    "operation mimics inference mode where "
    "nothing will be dropped from the input data and if mask is requested as output it "
    "will contain all ones.";

std::vector<TensorType> DropoutTypes13() {
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16};
}

std::vector<TensorType> DropoutTypes22() {
  return {
      TensorType::kBfloat16,   TensorType::kFloat16,        TensorType::kFloat,
      TensorType::kDouble,     TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz,
      TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
  };
}

LightOpSchema MakeDropoutSchema(int since_version) {
  if (since_version <= 7) {
    return LightOpSchema(
        "Dropout", kOnnxDomain, since_version, "",
        {
            {"data", kDropoutDataDescription, "T"},
        },
        {
            {"output", kDropoutOutputDescription, "T"},
            {"mask", since_version <= 6 ? kDropoutMaskDescriptionVer1And6 : kDropoutMaskDescription,
             "T"},
        },
        {
            {"T", FloatTypes(), "Constrain input and output types to float tensors."},
        });
  }
  if (since_version == 10) {
    return LightOpSchema(
        "Dropout", kOnnxDomain, since_version, "",
        {
            {"data", kDropoutDataDescription, "T"},
        },
        {
            {"output", kDropoutOutputDescription, "T"},
            {"mask", kDropoutMaskDescription, "T1"},
        },
        {
            {"T", FloatTypes(), "Constrain input and output types to float tensors."},
            {"T1", {TensorType::kBool}, "Constrain output mask types to boolean tensors."},
        });
  }
  return LightOpSchema(
      "Dropout", kOnnxDomain, since_version, "",
      {
          {"data", kDropoutDataDescription, "T"},
          {"ratio",
           since_version >= 22 ? kDropoutRatioDescriptionVer22 : kDropoutRatioDescriptionVer12And13,
           "T1"},
          {"training_mode", kDropoutTrainingModeDescription, "T2"},
      },
      {
          {"output", kDropoutOutputDescription, "T"},
          {"mask", kDropoutMaskDescription, "T2"},
      },
      {
          {"T",
           since_version >= 22 ? DropoutTypes22()
                               : (since_version >= 13 ? DropoutTypes13() : FloatTypes()),
           "Constrain input and output types to float tensors."},
          {"T1", since_version >= 22 ? DropoutTypes22() : FloatTypes(),
           "Constrain input 'ratio' types to float tensors."},
          {"T2", {TensorType::kBool}, "Constrain output 'mask' types to boolean tensors."},
      });
}

// --- GlobalAveragePool / GlobalMaxPool / GlobalLpPool -------------------------

const char *const kGlobalPoolInputDescription =
    "Input data tensor from the previous operator; "
    "dimensions for image case are (N x C x H x W), "
    "where N is the batch size, C is the number of "
    "channels, and H and W are the height and the width "
    "of the data. For non image case, the dimensions are "
    "in the form of (N x C x D1 x D2 ... Dn), "
    "where N is the batch size.";

const char *const kGlobalPoolOutputDescription =
    "Output data tensor from pooling across the input "
    "tensor. The output tensor has the same rank as the input. "
    "The first two dimensions of output shape are the same as "
    "the input (N x C), while the other dimensions are all 1.";

std::vector<TensorType> GlobalPoolTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeGlobalAveragePoolSchema(int since_version) {
  return LightOpSchema("GlobalAveragePool", kOnnxDomain, since_version,
                       MakeGlobalAveragePoolDoc(since_version),
                       {
                           {"X", kGlobalPoolInputDescription, "T"},
                       },
                       {
                           {"Y", kGlobalPoolOutputDescription, "T"},
                       },
                       {
                           {"T", GlobalPoolTypes(since_version),
                            "Constrain input and output types to float tensors."},
                       });
}

LightOpSchema MakeGlobalMaxPoolSchema(int since_version) {
  return LightOpSchema("GlobalMaxPool", kOnnxDomain, since_version,
                       MakeGlobalMaxPoolDoc(since_version),
                       {
                           {"X", kGlobalPoolInputDescription, "T"},
                       },
                       {
                           {"Y", kGlobalPoolOutputDescription, "T"},
                       },
                       {
                           {"T", GlobalPoolTypes(since_version),
                            "Constrain input and output types to float tensors."},
                       });
}

LightOpSchema MakeGlobalLpPoolSchema(int since_version) {
  // GlobalLpPool v1 has slightly different input/output descriptions from
  // upstream ONNX that must match exactly for the schema parity test.
  // Note: "the dimension are" (without 's') is an intentional verbatim copy of
  // the upstream ONNX v1 schema typo in onnx_lib/defs/nn/old.cc.
  const char *input_desc = since_version == 1
                               ? "Input data tensor from the previous operator; "
                                 "dimensions for image case are (N x C x H x W), "
                                 "where N is the batch size, C is the number of "
                                 "channels, and H and W are the height and the width "
                                 "of the data. For non image case, the dimension are "
                                 "in the form of (N x C x D1 x D2 ... Dn), "
                                 "where N is the batch size."
                               : kGlobalPoolInputDescription;
  const char *output_desc = since_version == 1 ? "Output data tensor from pooling across the input "
                                                 "tensor. Dimensions will be N x C x 1 x 1"
                                               : kGlobalPoolOutputDescription;
  return LightOpSchema("GlobalLpPool", kOnnxDomain, since_version,
                       MakeGlobalLpPoolDoc(since_version),
                       {
                           {"X", input_desc, "T"},
                       },
                       {
                           {"Y", output_desc, "T"},
                       },
                       {
                           {"T", GlobalPoolTypes(since_version),
                            "Constrain input and output types to float tensors."},
                       });
}

// --- Flatten -----------------------------------------------------------------

std::vector<TensorType> FlattenTypesIr4() {
  return {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32,    TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,     TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,
      TensorType::kString,   TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<TensorType> FlattenTypesIr9() {
  std::vector<TensorType> types = FlattenTypesIr4();
  types.push_back(TensorType::kFloat8e4m3fn);
  types.push_back(TensorType::kFloat8e4m3fnuz);
  types.push_back(TensorType::kFloat8e5m2);
  types.push_back(TensorType::kFloat8e5m2fnuz);
  return types;
}

std::vector<TensorType> FlattenTypesIr10() {
  std::vector<TensorType> types = FlattenTypesIr9();
  types.push_back(TensorType::kUint4);
  types.push_back(TensorType::kInt4);
  return types;
}

std::vector<TensorType> FlattenTypesIr11() {
  std::vector<TensorType> types = FlattenTypesIr10();
  types.push_back(TensorType::kFloat4e2m1);
  return types;
}

std::vector<TensorType> FlattenTypesIr12() {
  std::vector<TensorType> types = FlattenTypesIr11();
  types.push_back(TensorType::kFloat8e8m0);
  return types;
}

std::vector<TensorType> FlattenTypesIr13() {
  std::vector<TensorType> types = FlattenTypesIr12();
  types.push_back(TensorType::kUint2);
  types.push_back(TensorType::kInt2);
  return types;
}

std::vector<TensorType> FlattenTypes(int since_version) {
  switch (since_version) {
  case 25:
    return FlattenTypesIr13();
  case 24:
    return FlattenTypesIr12();
  case 23:
    return FlattenTypesIr11();
  case 21:
    return FlattenTypesIr10();
  case 13:
    return FlattenTypesIr4();
  case 11:
  case 9:
    return AllTensorTypes();
  case 1:
  default:
    return FloatTypes();
  }
}

const char *const kFlattenInputDescription = "A tensor of rank >= axis.";

const char *const kFlattenOutputDescription =
    "A 2D tensor with the contents of the input tensor, "
    "with input dimensions up to axis flattened to the outer dimension "
    "of the output and remaining input dimensions flattened into the inner "
    "dimension of the output.";

const char *const kFlattenAxisDescriptionLegacy =
    "Indicate up to which input dimensions "
    "(exclusive) should be flattened to the outer dimension of the output. "
    "The value for axis must be in the range [0, R], where R is the rank of the input "
    "tensor. "
    "When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n), "
    "where the shape of the input tensor is (d_0, d_1, ... d_n). ";

const char *const kFlattenAxisDescription =
    "Indicate up to which input dimensions "
    "(exclusive) should be flattened to the outer dimension of the output. "
    "The value for axis must be in the range [-r, r], where r is the rank of the input "
    "tensor. "
    "Negative value means counting dimensions from the back. "
    "When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n), "
    "where the shape of the input tensor is (d_0, d_1, ... d_n). ";

const char *FlattenTypeConstraintDescription(int since_version) {
  switch (since_version) {
  case 1:
    return "Constrain input and output types to float tensors.";
  case 21:
  case 23:
    // Upstream uses the description "...up to IRv10." for both v21 and v23,
    // even though v23 actually constrains to IRv11 types. The text is
    // preserved verbatim for schema parity.
    return "Constrain input and output to all tensor types up to IRv10.";
  case 24:
    return "Constrain input and output to all tensor types up to IRv12.";
  case 25:
    return "Constrain input and output to all tensor types up to IRv13.";
  case 9:
  case 11:
  case 13:
  default:
    return "Constrain input and output to all tensor types.";
  }
}

LightOpSchema MakeFlattenSchema(int since_version) {
  const char *axis_desc =
      since_version <= 9 ? kFlattenAxisDescriptionLegacy : kFlattenAxisDescription;
  return LightOpSchema(
      "Flatten", kOnnxDomain, since_version, MakeFlattenDoc(since_version),
      {
          {"input", kFlattenInputDescription, "T"},
      },
      {
          {"output", kFlattenOutputDescription, "T"},
      },
      {
          {"T", FlattenTypes(since_version), FlattenTypeConstraintDescription(since_version)},
      },
      {
          {"axis", axis_desc, AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
      });
}

// --- LRN ---------------------------------------------------------------------

std::vector<TensorType> LRNTypes(int since_version) {
  if (since_version >= 13) {
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

const char *const kLRNXDescription =
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

LightOpSchema MakeLRNSchema(int since_version) {
  return LightOpSchema(
      "LRN", kOnnxDomain, since_version, MakeLRNDoc(since_version),
      {
          {"X", kLRNXDescription, "T"},
      },
      {
          {"Y", "Output tensor, which has the shape and type as input tensor", "T"},
      },
      {
          {"T", LRNTypes(since_version),
           "Constrain input and output "
           " types to float tensors."},
      },
      {
          {"alpha", "Scaling parameter.", AttributeType::FLOAT, /*required=*/false, 0.0001f},
          {"beta", "The exponent.", AttributeType::FLOAT, /*required=*/false, 0.75f},
          {"bias", "", AttributeType::FLOAT, /*required=*/false, 1.0f},
          {"size", "The number of channels to sum over", AttributeType::INT, /*required=*/true,
           std::monostate{}},
      });
}

// --- LpNormalization ---------------------------------------------------------

std::vector<TensorType> LpNormalizationTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeLpNormalizationSchema(int since_version) {
  return LightOpSchema(
      "LpNormalization", kOnnxDomain, since_version, MakeLpNormalizationDoc(since_version),
      {
          {"input", "Input matrix", "T"},
      },
      {
          {"output", "Matrix after normalization", "T"},
      },
      {
          {"T", LpNormalizationTypes(since_version),
           "Constrain input and output types to float tensors."},
      },
      {
          {"axis", "The axis on which to apply normalization, -1 mean last axis.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(-1)},
          {"p", "The order of the normalization, only 1 or 2 are supported.", AttributeType::INT,
           /*required=*/false, static_cast<int64_t>(2)},
      });
}

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

// --- RotaryEmbedding ---------------------------------------------------------

// Input/output descriptions reproduced verbatim from the upstream ONNX
// RotaryEmbedding schema (v23 in onnx_lib/defs/nn/defs.cc) so the
// LightOpSchema parity test passes.

const char *const kRotaryEmbeddingXDescription =
    "The input tensor representing the token embeddings. "
    "4D tensor with shape `(batch_size, num_heads, sequence_length, head_size)` or 3D "
    "tensor with shape `(batch_size, sequence_length, hidden_size)`. "
    "For cases with a 4D input tensor, `head_size` has to be even. For cases with a 3D "
    "input tensor, `num_heads` attribute must be provided and "
    "`hidden_size` must be an even multiple of `num_heads` where `hidden_size = "
    "num_heads * head_size`";

const char *const kRotaryEmbeddingCosCacheDescription =
    "The cosine values for the rotation. "
    "2D tensor with shape `(max_position_id_plus_1, head_size / 2)` for full rotation "
    "or `(max_position_id_plus_1, rotary_embedding_dim / 2)` "
    "for partial rotation when `position_ids` are provided. 3D tensor with shape "
    "`(batch_size, sequence_length, head_size / 2)` "
    "for full rotation or `(batch_size, sequence_length, rotary_embedding_dim / 2)` for "
    "partial rotation when `position_ids` are not provided. "
    "`max_position_id_plus_1` is a parameter to the model.";

const char *const kRotaryEmbeddingSinCacheDescription =
    "The sine values for the rotation. "
    "2D tensor with shape `(max_position_id_plus_1, head_size / 2)` for full rotation "
    "or `(max_position_id_plus_1, rotary_embedding_dim / 2)` "
    "for partial rotation when `position_ids` are provided. 3D tensor with shape "
    "`(batch_size, sequence_length, head_size / 2)` "
    "for full rotation or `(batch_size, sequence_length, rotary_embedding_dim / 2)` for "
    "partial rotation when `position_ids` are not provided. "
    "`max_position_id_plus_1` is a parameter to the model.";

const char *const kRotaryEmbeddingPositionIdsDescription =
    "The position indices for the tokens. 2D tensor with shape `(batch_size, "
    "sequence_length)`";

LightOpSchema MakeRotaryEmbeddingSchema(int since_version) {
  std::vector<FormalParameter> inputs = {
      {"X", kRotaryEmbeddingXDescription, "T"},
      {"cos_cache", kRotaryEmbeddingCosCacheDescription, "T"},
      {"sin_cache", kRotaryEmbeddingSinCacheDescription, "T"},
      {"position_ids", kRotaryEmbeddingPositionIdsDescription, "M"},
  };
  std::vector<FormalParameter> outputs = {
      {"Y", "Tensor with same shape as input.", "T"},
  };
  std::vector<AttributeParam> attrs = {
      {"interleaved", "Rotate using interleaved pattern. Default value is 0 (False).",
       AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
      {"rotary_embedding_dim",
       "Rotary embedding dimension used to apply partial rotary embeddings.", AttributeType::INT,
       /*required=*/false, static_cast<int64_t>(0)},
      {"num_heads", "Number of attention heads. Must be provided when input is a 3D tensor. ",
       AttributeType::INT, /*required=*/false, std::monostate{}},
  };
  return LightOpSchema(
      "RotaryEmbedding", kOnnxDomain, since_version, MakeRotaryEmbeddingDoc(since_version),
      std::move(inputs), std::move(outputs),
      {
          {"T",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
          {"M", {TensorType::kInt64}, "Constrain input and output types to integer tensors."},
      },
      std::move(attrs),
      /*has_function_implementation=*/true);
}

// --- RMSNormalization --------------------------------------------------------

// Input/output descriptions reproduced verbatim from the upstream ONNX
// RMSNormalization schema (v23 in onnx_lib/defs/nn/defs.cc) so the
// LightOpSchema parity test passes.

const char *const kRMSNormalizationXDescription =
    "The input tensor to be normalized. "
    "In general, the shape is (D1, D2, ... , Dn) for n-dimensional data, where "
    "the root mean squared norm is taken over the last D dimensions, D is determined by "
    "the axis attribute.";

const char *const kRMSNormalizationScaleDescription =
    "Scale tensor. Scale tensor shape should be broadcastable to the normalized shape.";

const char *const kRMSNormalizationYDescription = "Output data tensor. Same shape as X";

const char *const kRMSNormalizationTConstraintDesc = "Constrain input X type to float tensors.";

const char *const kRMSNormalizationVConstraintDesc =
    "Constrain output Y and scale type to float tensors.";

LightOpSchema MakeRMSNormalizationSchema(int since_version) {
  std::vector<FormalParameter> inputs = {
      {"X", kRMSNormalizationXDescription, "T"},
      {"scale", kRMSNormalizationScaleDescription, "V"},
  };
  std::vector<FormalParameter> outputs = {
      {"Y", kRMSNormalizationYDescription, "V"},
  };
  std::vector<AttributeParam> attrs = {
      {"axis",
       "The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). "
       "Negative value means counting dimensions from the back.",
       AttributeType::INT, /*required=*/false, static_cast<int64_t>(-1)},
      {"epsilon", "The epsilon value to use to avoid division by zero.", AttributeType::FLOAT,
       /*required=*/false, static_cast<double>(1e-5f)},
      {"stash_type", "The floating-point precision used in stage one of the computation.",
       AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
  };
  return LightOpSchema(
      "RMSNormalization", kOnnxDomain, since_version, MakeRMSNormalizationDoc(since_version),
      std::move(inputs), std::move(outputs),
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           kRMSNormalizationTConstraintDesc},
          {"V",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           kRMSNormalizationVConstraintDesc},
      },
      std::move(attrs), /*has_function_implementation=*/true);
}

// --- Conv --------------------------------------------------------------------
// Floating-point Conv operator. Type constraint ``T`` was widened in opset 22
// to also allow ``tensor(bfloat16)`` via ``OpSchema::all_float_types_ir4()``.
// Opsets 11 and 1 expose ``[float16, float, double]``.

namespace {

constexpr const char *kConvInputXDescV11Plus =
    "Input data tensor from previous layer; "
    "has size (N x C x H x W), where N is the batch size, "
    "C is the number of channels, and H and W are the "
    "height and width. Note that this is for the 2D image. "
    "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
    "Optionally, if dimension denotation is "
    "in effect, the operation expects input data tensor "
    "to arrive with the dimension denotation of [DATA_BATCH, "
    "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].";

constexpr const char *kConvInputWDescV11Plus =
    "The weight tensor that will be used in the "
    "convolutions; has size (M x C/group x kH x kW), where C "
    "is the number of channels, and kH and kW are the "
    "height and width of the kernel, and M is the number "
    "of feature maps. For more than 2 dimensions, the "
    "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
    "where (k1 x k2 x ... kn) is the dimension of the kernel. "
    "Optionally, if dimension denotation is in effect, "
    "the operation expects the weight tensor to arrive "
    "with the dimension denotation of [FILTER_OUT_CHANNEL, "
    "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
    "Assuming zero based indices for the shape array, "
    "X.shape[1] == (W.shape[1] * group) == C and "
    "W.shape[0] mod G == 0. Or in other words "
    "FILTER_IN_CHANNEL multiplied by the number of groups "
    "should be equal to DATA_CHANNEL and the number of "
    "feature maps M should be a multiple of the number of "
    "groups G.";

constexpr const char *kConvInputWDescV1 =
    "The weight tensor that will be used in the "
    "convolutions; has size (M x C/group x kH x kW), where C "
    "is the number of channels, and kH and kW are the "
    "height and width of the kernel, and M is the number "
    "of feature maps. For more than 2 dimensions, the "
    "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
    "where (k1 x k2 x ... kn) is the dimension of the kernel. "
    "Optionally, if dimension denotation is in effect, "
    "the operation expects the weight tensor to arrive "
    "with the dimension denotation of [FILTER_OUT_CHANNEL, "
    "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
    "X.shape[1] == (W.shape[1] * group) == C "
    "(assuming zero based indices for the shape array). "
    "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ";

constexpr const char *kConvInputBDesc =
    "Optional 1D bias to be added to the convolution, has size of M.";

constexpr const char *kConvOutputYDesc = "Output data tensor that contains the result of the "
                                         "convolution. The output dimensions are functions "
                                         "of the kernel size, stride size, and pad lengths.";

std::vector<TensorType> ConvFloatTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

// The upstream Conv@11 schema accidentally lists ``tensor(float16)`` twice in
// its T type constraint (see onnx_lib/defs/nn/old.cc); replicate that here so
// the schema parity test succeeds.
std::vector<TensorType> ConvTypesForSchema(int since_version) {
  if (since_version == 11) {
    return {TensorType::kFloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return ConvFloatTypes(since_version);
}

} // namespace

LightOpSchema MakeConvSchema(int since_version) {
  const char *const w_desc = since_version >= 11 ? kConvInputWDescV11Plus : kConvInputWDescV1;
  return LightOpSchema("Conv", kOnnxDomain, since_version, MakeConvDoc(since_version),
                       {
                           {"X", kConvInputXDescV11Plus, "T"},
                           {"W", w_desc, "T"},
                           {"B", kConvInputBDesc, "T"},
                       },
                       {
                           {"Y", kConvOutputYDesc, "T"},
                       },
                       {
                           {"T", ConvTypesForSchema(since_version),
                            "Constrain input and output types to float tensors."},
                       });
}

// --- ConvInteger -------------------------------------------------------------
// Quantized integer convolution introduced at opset 10.

namespace {

constexpr const char *kConvIntegerInputXDesc =
    "Input data tensor from previous layer; "
    "has size (N x C x H x W), where N is the batch size, "
    "C is the number of channels, and H and W are the "
    "height and width. Note that this is for the 2D image. "
    "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
    "Optionally, if dimension denotation is "
    "in effect, the operation expects input data tensor "
    "to arrive with the dimension denotation of [DATA_BATCH, "
    "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].";

constexpr const char *kConvIntegerInputWDesc =
    "The weight tensor that will be used in the "
    "convolutions; has size (M x C/group x kH x kW), where C "
    "is the number of channels, and kH and kW are the "
    "height and width of the kernel, and M is the number "
    "of feature maps. For more than 2 dimensions, the "
    "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
    "where (k1 x k2 x ... kn) is the dimension of the kernel. "
    "Optionally, if dimension denotation is in effect, "
    "the operation expects the weight tensor to arrive "
    "with the dimension denotation of [FILTER_OUT_CHANNEL, "
    "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
    "X.shape[1] == (W.shape[1] * group) == C "
    "(assuming zero based indices for the shape array). "
    "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ";

constexpr const char *kConvIntegerInputXZeroPointDesc =
    "Zero point tensor for input 'x'. It's optional and default value is 0. It's a "
    "scalar, which means a per-tensor/layer quantization.";

constexpr const char *kConvIntegerInputWZeroPointDesc =
    "Zero point tensor for input 'w'. It's optional and default value is 0.  It could "
    "be a scalar or a 1-D tensor, "
    "which means a per-tensor/layer or per output channel quantization. If it's a 1-D "
    "tensor, its number "
    "of elements should be equal to the number of output channels (M)";

constexpr const char *kConvIntegerOutputYDesc =
    "Output data tensor that contains the result of the "
    "convolution. The output dimensions are functions "
    "of the kernel size, stride size, and pad lengths.";

} // namespace

LightOpSchema MakeConvIntegerSchema(int since_version) {
  return LightOpSchema(
      "ConvInteger", kOnnxDomain, since_version, MakeConvIntegerDoc(since_version),
      {
          {"x", kConvIntegerInputXDesc, "T1"},
          {"w", kConvIntegerInputWDesc, "T2"},
          {"x_zero_point", kConvIntegerInputXZeroPointDesc, "T1"},
          {"w_zero_point", kConvIntegerInputWZeroPointDesc, "T2"},
      },
      {
          {"y", kConvIntegerOutputYDesc, "T3"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain input x and its zero point data type to 8-bit integer tensor."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain input w and its zero point data type to 8-bit integer tensor."},
          {"T3", {TensorType::kInt32}, "Constrain output y data type to 32-bit integer tensor."},
      });
}

// --- ConvTranspose -----------------------------------------------------------
// Transposed convolution. ``T`` was widened in opset 22 to also allow
// ``tensor(bfloat16)``.

namespace {

constexpr const char *kConvTransposeInputXDesc =
    "Input data tensor from previous layer; has size (N x C x H x W)"
    ", where N is the batch size, C is the number of channels, and"
    " H and W are the height and width. Note that this is for the 2D image. "
    "Otherwise the size is (N x C x D1 x D2 ... x Dn)";

constexpr const char *kConvTransposeInputWDesc =
    "The weight tensor that will be used in the "
    "convolutions; has size (C x M/group x kH x kW), where C "
    "is the number of channels, and kH and kW are the "
    "height and width of the kernel, and M is the number "
    "of feature maps. For more than 2 dimensions, the "
    "weight shape will be (C x M/group x k1 x k2 x ... x kn), "
    "where (k1 x k2 x ... x kn) is the dimension of the kernel. "
    "The number of channels in the output should be equal to W.shape[1] * group "
    "(assuming zero based indices of the shape array)";

constexpr const char *kConvTransposeInputBDesc =
    "Optional 1D bias to be added to the convolution, has size of M.";

constexpr const char *kConvTransposeOutputYDesc =
    "Output data tensor that contains the result of the convolution. The "
    "output dimensions are functions of the kernel size, stride size, "
    "pad lengths and group count. "
    "The number of channels in the output should be equal to W.shape[1] * group "
    "(assuming zero based indices of the shape array)";

} // namespace

LightOpSchema MakeConvTransposeSchema(int since_version) {
  return LightOpSchema("ConvTranspose", kOnnxDomain, since_version,
                       MakeConvTransposeDoc(since_version),
                       {
                           {"X", kConvTransposeInputXDesc, "T"},
                           {"W", kConvTransposeInputWDesc, "T"},
                           {"B", kConvTransposeInputBDesc, "T"},
                       },
                       {
                           {"Y", kConvTransposeOutputYDesc, "T"},
                       },
                       {
                           {"T", ConvFloatTypes(since_version),
                            "Constrain input and output types to float tensors."},
                       });
}

// --- DeformConv --------------------------------------------------------------

std::vector<TensorType> DeformConvFloatTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeDeformConvSchema(int since_version) {
  return LightOpSchema(
      "DeformConv", kOnnxDomain, since_version, MakeDeformConvDoc(since_version),
      {
          {"X",
           "Input data tensor. For 2D image data, it has shape (N, C, H, W) where N is the "
           "batch size, "
           "C is the number of input channels, and H and W are the height and width. "
           "In general, the shape is (N, C, D1, D2, ... , Dn) for n-dimensional data, where "
           "D1 to Dn are the spatial dimension sizes. Most common use cases have n = 2 or 3.",
           "T"},
          {"W",
           "Weight tensor that will be used in the convolutions. It has shape (oC, C/group, "
           "kH, kW), "
           "where oC is the number of output channels and kH and kW are the kernel height and "
           "width. "
           "For more than 2 dimensions, it has shape (oC, C/group, k1, k2, ... , kn).",
           "T"},
          {"offset",
           "Offset tensor denoting the offset for the sampling locations in the convolution "
           "kernel. "
           "It has shape (N, offset_group * kH * kW * 2, oH, oW) for 2D data or "
           "(N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Use "
           "linear interpolation"
           "for fractional offset values. Sampling locations outside of the padded input "
           "tensor gives zero.",
           "T"},
          {"B",
           "Optional 1D bias of length oC to be added to the convolution. Default is a tensor "
           "of zeros.",
           "T"},
          {"mask",
           "The mask tensor to be applied to each position in the convolution kernel. "
           "It has shape (N, offset_group * kH * kW, oH, oW) for 2D data or "
           "(N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Default "
           "is a "
           "tensor of ones.",
           "T"},
      },
      {
          {"Y",
           "Output data tensor that contains the result of convolution. It has shape (N, oC, "
           "oH, oW) "
           "for 2D data or (N, oC, o1, o2, ..., on) for nD data",
           "T"},
      },
      {
          {"T", DeformConvFloatTypes(since_version),
           "Constrain input and output types to float tensors."},
      });
}

// --- Col2Im ------------------------------------------------------------------

LightOpSchema MakeCol2ImSchema(int since_version) {
  return LightOpSchema(
      "Col2Im", kOnnxDomain, since_version, MakeCol2ImDoc(since_version),
      {
          {"input",
           "Input data tensor to be rearranged from column blocks back into an image."
           " This is a 3-dimensional tensor containing [N, C * n-ary-product(block_shape), L],"
           " where N is batch dimension, C is image channel dimension and L is number of blocks."
           "The blocks are enumerated in increasing lexicographic-order of their indices."
           "For example, with an image-size 10*20 and block-size 9*18, there would be 2*3 blocks,"
           " enumerated in the order block(0, 0), block(0, 1), block(0, 2), block(1, 0), block(1, "
           "1), block(1, 2).",
           "T"},
          {"image_shape",
           "The shape of the spatial dimensions of the image after rearranging the column blocks."
           "This is a 1-dimensional tensor with size of at least 2, containing the value [H_img, "
           "W_img] "
           " for a 2-D image or [dim_i1, dim_i2, ..., dim_iN] for a N-D image.",
           "tensor(int64)"},
          {"block_shape",
           "The shape of the block to apply on the input."
           "This is a 1-dimensional tensor of size of at least 2, containing the value "
           "[H_block, W_block] "
           " for a 2-D image or [dim_b1, dim_b2, ..., dim_bN] for a N-D block."
           "This is the block-shape before dilation is applied to it.",
           "tensor(int64)"},
      },
      {
          {"output", "Output tensor produced by rearranging blocks into an image.", "T"},
      },
      {
          {"T", ConcatTypesVer13(),
           "Constrain input and output types to all numeric tensor types."},
      },
      {
          {"dilations",
           "1-dimensional tensor with dilation value along each spatial axis of the image. "
           "If not present, the dilation defaults to 1 along each spatial axis of the image.",
           AttributeType::INTS, /*required=*/false},
          {"pads",
           "1-dimensional tensor with padding value for the beginning and ending along each "
           "spatial axis, "
           "it can take any value greater than or equal to 0. "
           "The value represent the number of pixels added to the beginning "
           "and end part of the corresponding axis. `pads` format should be as follow "
           "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin is the number of pixels "
           "added at the beginning of axis `i` and xi_end is the number of pixels added at the "
           "end of axis `i`. "
           "If not present, the padding defaults to 0 along start and end of each spatial axis.",
           AttributeType::INTS, /*required=*/false},
          {"strides",
           "1-dimensional tensor with stride value along each spatial axis. "
           "If not present, the stride defaults to 1 along each spatial axis.",
           AttributeType::INTS, /*required=*/false},
      });
}

// --- InstanceNormalization --------------------------------------------------

const char *const kInstanceNormalizationXDescVer1 = "The input 4-dimensional tensor of shape NCHW.";

const char *const kInstanceNormalizationXDescVer6 =
    "Input data tensor from the previous operator; "
    "dimensions for image case are (N x C x H x W), "
    "where N is the batch size, C is the number of "
    "channels, and H and W are the height and the "
    "width of the data. For non image case, the "
    "dimensions are in the form of "
    "(N x C x D1 x D2 ... Dn), where N is the batch "
    "size.";

const char *const kInstanceNormalizationScaleDesc =
    "The input 1-dimensional scale tensor of size C.";

const char *const kInstanceNormalizationBiasDesc = "The input 1-dimensional bias tensor of size C.";

const char *const kInstanceNormalizationYDescVer1 =
    "The output 4-dimensional tensor of the same shape as input.";

const char *const kInstanceNormalizationYDescVer6 = "The output tensor of the same shape as input.";

const char *const kInstanceNormalizationTConstraintDesc =
    "Constrain input and output types to float tensors.";

std::vector<TensorType> InstanceNormalizationFloatTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeInstanceNormalizationSchema(int since_version) {
  std::vector<AttributeParam> attrs;
  if (since_version == 1) {
    attrs.push_back(AttributeParam{"consumed_inputs", "legacy optimization attribute.",
                                   AttributeType::INTS, /*required=*/false, std::monostate{}});
  }
  attrs.push_back(AttributeParam{"epsilon", "The epsilon value to use to avoid division by zero.",
                                 AttributeType::FLOAT, /*required=*/false,
                                 static_cast<double>(1e-5f)});

  const char *x_desc =
      since_version == 1 ? kInstanceNormalizationXDescVer1 : kInstanceNormalizationXDescVer6;
  const char *y_desc =
      since_version == 1 ? kInstanceNormalizationYDescVer1 : kInstanceNormalizationYDescVer6;

  return LightOpSchema("InstanceNormalization", kOnnxDomain, since_version,
                       MakeInstanceNormalizationDoc(since_version),
                       {
                           {"input", x_desc, "T"},
                           {"scale", kInstanceNormalizationScaleDesc, "T"},
                           {"B", kInstanceNormalizationBiasDesc, "T"},
                       },
                       {
                           {"output", y_desc, "T"},
                       },
                       {
                           {"T", InstanceNormalizationFloatTypes(since_version),
                            kInstanceNormalizationTConstraintDesc},
                       },
                       std::move(attrs));
}

// --- GroupNormalization -----------------------------------------------------

const char *const kGroupNormalizationXDesc =
    "Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where `N` is "
    "the batch size, "
    "`C` is the number of channels, and `H` and `W` are the height and width of the "
    "data. Statistics are "
    "computed for every group of channels over `C`, `H`, and `W`. For non-image cases, "
    "the dimensions are "
    "in the form of `(N x C x D1 x D2 ... Dn)`.";

const char *const kGroupNormalizationYDesc = "The output tensor of the same shape as `X`.";

const char *const kGroupNormalizationScaleDescVer18 = "Scale tensor of shape `(num_groups)`.";
const char *const kGroupNormalizationBiasDescVer18 = "Bias tensor of shape `(num_groups)`.";
const char *const kGroupNormalizationScaleDescVer21 = "Scale tensor of shape `(C)`.";
const char *const kGroupNormalizationBiasDescVer21 = "Bias tensor of shape `(C)`.";

const char *const kGroupNormalizationTConstraintDesc =
    "Constrain input and output types to float tensors.";

std::vector<TensorType> GroupNormalizationFloatTypes(int since_version) {
  if (since_version >= 21) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16};
}

LightOpSchema MakeGroupNormalizationSchema(int since_version) {
  std::vector<AttributeParam> attrs;
  attrs.push_back(AttributeParam{"epsilon", "The epsilon value to use to avoid division by zero.",
                                 AttributeType::FLOAT, /*required=*/false,
                                 static_cast<double>(1e-5f)});
  attrs.push_back(AttributeParam{
      "num_groups",
      "The number of groups of channels. It should be a divisor of the number of channels `C`.",
      AttributeType::INT, /*required=*/true, std::monostate{}});
  if (since_version >= 21) {
    attrs.push_back(AttributeParam{
        "stash_type", "The floating-point precision used in stage one of the computation.",
        AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)});
  }

  const char *scale_desc =
      since_version >= 21 ? kGroupNormalizationScaleDescVer21 : kGroupNormalizationScaleDescVer18;
  const char *bias_desc =
      since_version >= 21 ? kGroupNormalizationBiasDescVer21 : kGroupNormalizationBiasDescVer18;

  LightOpSchema schema(
      "GroupNormalization", kOnnxDomain, since_version, MakeGroupNormalizationDoc(since_version),
      {
          {"X", kGroupNormalizationXDesc, "T"},
          {"scale", scale_desc, "T"},
          {"bias", bias_desc, "T"},
      },
      {
          {"Y", kGroupNormalizationYDesc, "T"},
      },
      {
          {"T", GroupNormalizationFloatTypes(since_version), kGroupNormalizationTConstraintDesc},
      },
      std::move(attrs), /*has_function_implementation=*/true);
  if (since_version == 18) {
    schema.set_deprecated(true);
  }
  return schema;
}

// --- MeanVarianceNormalization ----------------------------------------------

const char *const kMeanVarianceNormalizationAxesDesc =
    "A list of integers, along which to reduce. The default is to calculate along axes [0,2,3] "
    "for calculating mean and variance along each channel. Two variables with the same "
    "C-coordinate are associated with the same mean and variance.";

std::vector<TensorType> MeanVarianceNormalizationTypes(int since_version) {
  if (since_version >= 13) {
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeMeanVarianceNormalizationSchema(int since_version) {
  return LightOpSchema("MeanVarianceNormalization", kOnnxDomain, since_version,
                       MakeMeanVarianceNormalizationDoc(since_version),
                       {
                           {"X", "Input tensor", "T"},
                       },
                       {
                           {"Y", "Output tensor", "T"},
                       },
                       {
                           {"T", MeanVarianceNormalizationTypes(since_version),
                            "Constrain input and output types to all numeric tensors."},
                       },
                       {
                           {"axes", kMeanVarianceNormalizationAxesDesc, AttributeType::INTS,
                            /*required=*/false, std::vector<int64_t>{0, 2, 3}},
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
      {"Col2Im",
       [] {
         return std::vector<LightOpSchema>{
             MakeCol2ImSchema(18),
         };
       }},
      {"Conv",
       [] {
         return std::vector<LightOpSchema>{
             MakeConvSchema(22),
             MakeConvSchema(11),
             MakeConvSchema(1),
         };
       }},
      {"ConvInteger",
       [] {
         return std::vector<LightOpSchema>{
             MakeConvIntegerSchema(10),
         };
       }},
      {"ConvTranspose",
       [] {
         return std::vector<LightOpSchema>{
             MakeConvTransposeSchema(22),
             MakeConvTransposeSchema(11),
             MakeConvTransposeSchema(1),
         };
       }},
      {"DeformConv",
       [] {
         return std::vector<LightOpSchema>{
             MakeDeformConvSchema(22),
             MakeDeformConvSchema(19),
         };
       }},
      {"Dropout",
       [] {
         return std::vector<LightOpSchema>{
             MakeDropoutSchema(22), MakeDropoutSchema(13), MakeDropoutSchema(12),
             MakeDropoutSchema(10), MakeDropoutSchema(7),  MakeDropoutSchema(6),
             MakeDropoutSchema(1),
         };
       }},
      {"Flatten",
       [] {
         return std::vector<LightOpSchema>{
             MakeFlattenSchema(25), MakeFlattenSchema(24), MakeFlattenSchema(23),
             MakeFlattenSchema(21), MakeFlattenSchema(13), MakeFlattenSchema(11),
             MakeFlattenSchema(9),  MakeFlattenSchema(1),
         };
       }},
      {"GlobalAveragePool",
       [] {
         return std::vector<LightOpSchema>{
             MakeGlobalAveragePoolSchema(22),
             MakeGlobalAveragePoolSchema(1),
         };
       }},
      {"GlobalLpPool",
       [] {
         return std::vector<LightOpSchema>{
             MakeGlobalLpPoolSchema(22),
             MakeGlobalLpPoolSchema(2),
             MakeGlobalLpPoolSchema(1),
         };
       }},
      {"GlobalMaxPool",
       [] {
         return std::vector<LightOpSchema>{
             MakeGlobalMaxPoolSchema(22),
             MakeGlobalMaxPoolSchema(1),
         };
       }},
      {"GRU",
       [] {
         return std::vector<LightOpSchema>{
             MakeGRUSchema(22), MakeGRUSchema(14), MakeGRUSchema(7),
             MakeGRUSchema(3),  MakeGRUSchema(1),
         };
       }},
      {"GroupNormalization",
       [] {
         return std::vector<LightOpSchema>{
             MakeGroupNormalizationSchema(21),
             MakeGroupNormalizationSchema(18),
         };
       }},
      {"InstanceNormalization",
       [] {
         return std::vector<LightOpSchema>{
             MakeInstanceNormalizationSchema(22),
             MakeInstanceNormalizationSchema(6),
             MakeInstanceNormalizationSchema(1),
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
      {"LRN",
       [] {
         return std::vector<LightOpSchema>{
             MakeLRNSchema(13),
             MakeLRNSchema(1),
         };
       }},
      {"MeanVarianceNormalization",
       [] {
         return std::vector<LightOpSchema>{
             MakeMeanVarianceNormalizationSchema(13),
             MakeMeanVarianceNormalizationSchema(9),
         };
       }},
      {"LpNormalization",
       [] {
         return std::vector<LightOpSchema>{
             MakeLpNormalizationSchema(22),
             MakeLpNormalizationSchema(1),
         };
       }},
      {"MaxPool",
       [] {
         return std::vector<LightOpSchema>{
             MakeMaxPoolSchema(22), MakeMaxPoolSchema(12), MakeMaxPoolSchema(11),
             MakeMaxPoolSchema(10), MakeMaxPoolSchema(8),  MakeMaxPoolSchema(1),
         };
       }},
      {"MaxUnpool",
       [] {
         return std::vector<LightOpSchema>{
             MakeMaxUnpoolSchema(22),
             MakeMaxUnpoolSchema(11),
             MakeMaxUnpoolSchema(9),
         };
       }},
      {"MaxRoiPool",
       [] {
         return std::vector<LightOpSchema>{
             MakeMaxRoiPoolSchema(22),
             MakeMaxRoiPoolSchema(1),
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
      {"RMSNormalization",
       [] {
         return std::vector<LightOpSchema>{
             MakeRMSNormalizationSchema(23),
         };
       }},
      {"RotaryEmbedding",
       [] {
         return std::vector<LightOpSchema>{
             MakeRotaryEmbeddingSchema(23),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

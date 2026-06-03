// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_tensor_doc.h"

#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

namespace {

// Mirrors OpSchema::all_float_types_ir4() ordering used by the upstream
// AffineGrid schema: bfloat16, float16, float, double.
std::vector<TensorType> AffineGridFloatTypes() {
  return {
      TensorType::kBfloat16,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  };
}

// Mirrors OpSchema::all_non_string_tensor_types_ir13() used by upstream
// BitCast (opset 26): the full ONNX type set minus STRING. Order matches
// the upstream helper exactly.
std::vector<TensorType> BitCastTypesVer26() {
  return {
      TensorType::kUint8,          TensorType::kUint16,         TensorType::kUint32,
      TensorType::kUint64,         TensorType::kInt8,           TensorType::kInt16,
      TensorType::kInt32,          TensorType::kInt64,          TensorType::kBfloat16,
      TensorType::kFloat16,        TensorType::kFloat,          TensorType::kDouble,
      TensorType::kBool,           TensorType::kComplex64,      TensorType::kComplex128,
      TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2,
      TensorType::kFloat8e5m2fnuz, TensorType::kUint4,          TensorType::kInt4,
      TensorType::kFloat4e2m1,     TensorType::kFloat8e8m0,     TensorType::kUint2,
      TensorType::kInt2,
  };
}

// Mirrors OpSchema::all_non_complex_tensor_types_ir10() used by upstream
// CastLike v21 (uint8-first ordering matching ONNX IR version 10). This is
// distinct from Cast v21's CastTypesVer21() helper which preserves the
// float16-first historical ordering used by Cast; CastLike v21 uses its own
// helper because the OnnxOpSchemaParityTest enforces exact ordering match
// against onnx_lib.
std::vector<TensorType> CastLikeTypesVer21() {
  return {
      TensorType::kUint8,          TensorType::kUint16,     TensorType::kUint32,
      TensorType::kUint64,         TensorType::kInt8,       TensorType::kInt16,
      TensorType::kInt32,          TensorType::kInt64,      TensorType::kBfloat16,
      TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
      TensorType::kString,         TensorType::kBool,       TensorType::kFloat8e4m3fn,
      TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
      TensorType::kUint4,          TensorType::kInt4,
  };
}

std::vector<TensorType> TransposeTypesVer21() {
  std::vector<TensorType> types = ConcatTypesVer13();
  types.push_back(TensorType::kFloat8e4m3fn);
  types.push_back(TensorType::kFloat8e4m3fnuz);
  types.push_back(TensorType::kFloat8e5m2);
  types.push_back(TensorType::kFloat8e5m2fnuz);
  types.push_back(TensorType::kUint4);
  types.push_back(TensorType::kInt4);
  return types;
}

std::vector<TensorType> TransposeTypesVer23() {
  std::vector<TensorType> types = TransposeTypesVer21();
  types.push_back(TensorType::kFloat4e2m1);
  return types;
}

std::vector<TensorType> TransposeTypesVer24() {
  std::vector<TensorType> types = TransposeTypesVer23();
  types.push_back(TensorType::kFloat8e8m0);
  return types;
}

std::vector<TensorType> TransposeTypesVer25() {
  std::vector<TensorType> types = TransposeTypesVer24();
  types.push_back(TensorType::kUint2);
  types.push_back(TensorType::kInt2);
  return types;
}

// Mirrors OpSchema::all_tensor_types_ir4() ordering used by the upstream
// GridSample v22 schema (T1): adds bfloat16 to all_tensor_types() — order
// matches ConcatTypesVer13() exactly.
std::vector<TensorType> GridSampleInputTypesVer22() { return ConcatTypesVer13(); }

// Mirrors OpSchema::all_tensor_types_ir9() used by the upstream Shape v19
// schema: extends all_tensor_types_ir4 with the four float8 tensor types
// (E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ) in that order.
std::vector<TensorType> ShapeTypesVer19() {
  std::vector<TensorType> types = ConcatTypesVer13();
  types.push_back(TensorType::kFloat8e4m3fn);
  types.push_back(TensorType::kFloat8e4m3fnuz);
  types.push_back(TensorType::kFloat8e5m2);
  types.push_back(TensorType::kFloat8e5m2fnuz);
  return types;
}

// Mirrors OpSchema::all_float_types_ir4() ordering used by the upstream
// GridSample v22 schema (T2). Same set as AffineGridFloatTypes() above; kept
// as a separate helper so the GridSample schema reads naturally.
std::vector<TensorType> GridSampleGridTypesVer22() { return AffineGridFloatTypes(); }

} // namespace

LightOpSchema MakeAffineGridSchema(int since_version) {
  return LightOpSchema(
      "AffineGrid", kOnnxDomain, since_version, MakeAffineGridDoc(since_version),
      {
          {"theta",
           "input batch of affine matrices with shape (N, 2, 3) for 2D or (N, 3, 4) for 3D", "T1"},
          {"size", "the target output image size (N, C, H, W) for 2D or (N, C, D, H, W) for 3D",
           "T2"},
      },
      {
          {"grid",
           "output tensor of shape (N, H, W, 2) of 2D sample coordinates or (N, D, H, W, 3) "
           "of 3D sample coordinates.",
           "T1"},
      },
      {
          {"T1", AffineGridFloatTypes(),
           MakeAffineGridGridTypeConstraintDescription(since_version)},
          {"T2", {TensorType::kInt64}, MakeAffineGridSizeTypeConstraintDescription(since_version)},
      },
      {
          {"align_corners",
           "if align_corners=1, consider -1 and 1 to refer to the centers of the corner pixels. "
           "if align_corners=0, consider -1 and 1 to refer to the outer edge the corner pixels.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
      },
      /*has_function_implementation=*/true);
}

LightOpSchema MakeGridSampleSchema(int since_version, const std::vector<TensorType> &x_types,
                                   const std::vector<TensorType> &grid_types) {
  const bool is_v16 = since_version <= 16;
  const std::string mode_default = is_v16 ? "bilinear" : "linear";
  const std::string mode_doc =
      is_v16 ? "Three interpolation modes: bilinear (default), nearest and bicubic."
             : "Three interpolation modes: linear (default), nearest and cubic. "
               "The \"linear\" mode includes linear and N-linear interpolation modes depending "
               "on the number of spatial dimensions "
               "of the input tensor (i.e. linear for 1 spatial dimension, bilinear for 2 spatial "
               "dimensions, etc.). "
               "The \"cubic\" mode also includes N-cubic interpolation modes following the same "
               "rules. The \"nearest\" mode rounds "
               "to the nearest even index when the sampling point falls halfway between two "
               "indices.";
  const std::string padding_doc =
      "Support padding modes for outside grid values: `zeros`(default), `border`, "
      "`reflection`. "
      "zeros: use 0 for out-of-bound grid locations, "
      "border: use border values for out-of-bound grid locations, "
      "reflection: use values at locations reflected by the border for out-of-bound grid "
      "locations. "
      "If index 0 represents the margin pixel, the reflected value at index -1 will be the "
      "same as the value at index 1. "
      "For location far away from the border, it will keep being reflected until becoming "
      "in bound. "
      "If pixel location x = -3.5 reflects by border -1 and becomes x' = 1.5, then "
      "reflects by border 1 and becomes x'' = 0.5.";
  const std::string align_doc =
      is_v16 ? "If align_corners=1, the extrema (-1 and 1) are considered as referring to the "
               "center points of the input's corner pixels. "
               "If align_corners=0, they are instead considered as referring to the corner points "
               "of the input's corner pixels, making the sampling more resolution agnostic."
             : "If align_corners=1, the extrema (-1 and 1) are considered as referring to the "
               "center points of the input's corner pixels (voxels, etc.). "
               "If align_corners=0, they are instead considered as referring to the corner points "
               "of the input's corner pixels (voxels, etc.), "
               "making the sampling more resolution agnostic.";

  const std::string x_desc =
      is_v16 ? "4-D tensor of shape (N, C, H, W), "
               "where N is the batch size, C is the numbers of channels, "
               "H and W are the height and width of the input data."
             : "Input tensor of rank r+2 that has shape (N, C, D1, D2, ..., Dr), where N is the "
               "batch size, "
               "C is the number of channels, D1, D2, ..., Dr are the spatial dimensions.";
  const std::string grid_desc =
      is_v16 ? "Input offset, 4-D tensor of shape (N, H_out, W_out, 2), "
               "where H_out and W_out are the height and width of grid and output, "
               "Grid specifies the sampling pixel locations normalized by the input spatial "
               "dimensions. "
               "Therefore, it should have most values in the range of [-1, 1]. "
               "If grid has values outside the range of [-1, 1], the corresponding outputs will be "
               "handled as defined by padding_mode."
             : "Input offset of shape (N, D1_out, D2_out, ..., Dr_out, r), where D1_out, D2_out, "
               "..., "
               "Dr_out are the spatial dimensions of the grid and output, and r is the number of "
               "spatial dimensions. "
               "Grid specifies the sampling locations normalized by the input spatial dimensions. "
               "Therefore, it should have most values in the range of [-1, 1]. If the grid has "
               "values "
               "outside the range of [-1, 1], "
               "the corresponding outputs will be handled as defined by padding_mode. Following "
               "computer vision convention, "
               "the coordinates in the length-r location vector are listed from the innermost "
               "tensor "
               "dimension to the outermost, "
               "the opposite of regular tensor indexing.";
  const std::string y_desc =
      is_v16 ? "4-D tensor of shape (N, C, H_out, W_out) of sampled values. "
               "For integer input types, intermediate values are computed as floating point and "
               "cast to integer at the end."
             : "Output tensor of rank r+2 that has shape (N, C, D1_out, D2_out, ..., Dr_out) of "
               "the sampled values. "
               "For integer input types, intermediate values are computed as floating point and "
               "cast to integer at the end.";

  return LightOpSchema(
      "GridSample", kOnnxDomain, since_version, MakeGridSampleDoc(since_version),
      {
          {"X", x_desc, "T1"},
          {"grid", grid_desc, "T2"},
      },
      {
          {"Y", y_desc, "T1"},
      },
      {
          {"T1", x_types, MakeGridSampleInputTypeConstraintDescription(since_version)},
          {"T2", grid_types, MakeGridSampleGridTypeConstraintDescription(since_version)},
      },
      {
          {"mode", mode_doc, AttributeType::STRING, /*required=*/false, mode_default},
          {"padding_mode", padding_doc, AttributeType::STRING, /*required=*/false,
           std::string("zeros")},
          {"align_corners", align_doc, AttributeType::INT, /*required=*/false,
           static_cast<int64_t>(0)},
      });
}

LightOpSchema MakeCastSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "Cast", kOnnxDomain, since_version, MakeCastDoc(since_version),
      {
          {"input", "Input tensor to be cast.", "T1"},
      },
      {
          {"output",
           "Output tensor with the same shape as input with type specified by the 'to' argument",
           "T2"},
      },
      {
          {"T1", types, MakeCastInputTypeConstraintDescription(since_version)},
          {"T2", types, MakeCastOutputTypeConstraintDescription(since_version)},
      });
}

LightOpSchema MakeBitCastSchema() {
  // BitCast (opset 26): unary reinterpret-cast with a required ``to``
  // attribute. Inputs and outputs share the same type set
  // (all_non_string_tensor_types_ir13); the bit-width check happens at
  // shape inference / runtime, not in the schema metadata.
  const std::vector<TensorType> types = BitCastTypesVer26();
  std::vector<AttributeParam> attributes;
  attributes.push_back({"to",
                        "The data type to which the input tensor is bitwise reinterpreted. "
                        "Must be one of the non-string types from DataType enum in TensorProto. "
                        "The target type must have the same bit-width as the input type.",
                        AttributeType::INT, /*required=*/true});
  return LightOpSchema(
      "BitCast", kOnnxDomain, 26, MakeBitCastDoc(),
      {
          {"input", "Input tensor to be bitcast.", "T1"},
      },
      {
          {"output", "Output tensor with the same shape as the input.", "T2"},
      },
      {
          {"T1", types, "Constrain input types. Bitcasting from string is not supported."},
          {"T2", types, "Constrain output types. Bitcasting to string is not supported."},
      },
      std::move(attributes));
}

LightOpSchema MakeCastLikeSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "CastLike", kOnnxDomain, since_version, MakeCastLikeDoc(since_version),
      {
          {"input", "Input tensor to be cast.", "T1"},
          {"target_type",
           "The (first) input tensor will be cast to produce a tensor of the same type as this "
           "(second input) tensor.",
           "T2"},
      },
      {
          {"output",
           "Output tensor produced by casting the first input tensor to have the same type as "
           "the second input tensor.",
           "T2"},
      },
      {
          {"T1", types, MakeCastLikeInputTypeConstraintDescription(since_version)},
          {"T2", types, MakeCastLikeOutputTypeConstraintDescription(since_version)},
      },
      /*has_function_implementation=*/true);
}

LightOpSchema MakeConcatSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema("Concat", kOnnxDomain, since_version, MakeConcatDoc(since_version),
                       {
                           {"inputs", "List of tensors for concatenation", "T"},
                       },
                       {
                           {"concat_result", "Concatenated tensor", "T"},
                       },
                       {
                           {"T", types, MakeConcatTypeConstraintDescription(since_version)},
                       });
}

LightOpSchema MakeExpandSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "Expand", kOnnxDomain, since_version, MakeExpandDoc(since_version),
      {
          {"input", "Input tensor", "T"},
          {"shape",
           "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
           "tensor(int64)"},
      },
      {
          {"output", "Output tensor", "T"},
      },
      {
          {"T", types, MakeExpandTypeConstraintDescription(since_version)},
      });
}

LightOpSchema MakeReshapeSchema(int since_version, const std::vector<TensorType> &types) {
  std::vector<AttributeParam> attributes;
  if (since_version >= 14) {
    attributes.push_back(
        {"allowzero",
         "(Optional) By default, when any value in the 'shape' input is equal to zero the "
         "corresponding dimension value is copied from the input tensor dynamically. allowzero=1 "
         "indicates that if any value in the 'shape' input is set to zero, the zero value is "
         "honored, similar to NumPy.",
         AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)});
  }
  return LightOpSchema("Reshape", kOnnxDomain, since_version,
                       "Reshape the input tensor similar to numpy.reshape.",
                       {
                           {"data", "An input tensor.", "T"},
                           {"shape", "Specified shape for output.", "tensor(int64)"},
                       },
                       {
                           {"reshaped", "Reshaped data.", "T"},
                       },
                       {
                           {"T", types, "Constrain input and output types to all tensor types."},
                       },
                       std::move(attributes));
}

LightOpSchema MakeSliceSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "Slice", kOnnxDomain, since_version,
      "Produces a slice of the input tensor along multiple axes.",
      {
          {"data", "Tensor of data to extract slices from.", "T"},
          {"starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind"},
          {"ends", "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`",
           "Tind"},
          {"axes",
           "1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting "
           "dimensions from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior "
           "is undefined if an axis is repeated.",
           "Tind"},
          {"steps",
           "1-D tensor of slice step of corresponding axis in `axes`. Negative value means "
           "slicing backward. 'steps' cannot be 0. Defaults to 1s.",
           "Tind"},
      },
      {
          {"output", "Sliced data tensor.", "T"},
      },
      {
          {"T", types, "Constrain input and output types to all tensor types."},
          {"Tind", {TensorType::kInt32, TensorType::kInt64}, "Constrain indices to integer types"},
      });
}

LightOpSchema MakeSqueezeSchema(int since_version, const std::vector<TensorType> &types) {
  const bool axes_is_input = since_version >= 13;
  LightOpSchema schema(
      "Squeeze", kOnnxDomain, since_version, MakeSqueezeDoc(since_version),
      axes_is_input ? std::vector<FormalParameter>{
                          {"data", "Tensors with at least max(dims) dimensions.", "T"},
                          {"axes",
                           since_version >= 23
                               ? "1D tensor of integers indicating the dimensions to squeeze. "
                                 "Negative value means counting dimensions from the back. Accepted "
                                 "range is [-r, r-1] where r = rank(data)."
                               : "List of integers indicating the dimensions to squeeze. Negative "
                                 "value means counting dimensions from the back. Accepted range is "
                                 "[-r, r-1] where r = rank(data).",
                           "tensor(int64)"},
                      }
                    : std::vector<FormalParameter>{
                          {"data", "Tensors with at least max(dims) dimensions.", "T"},
                      },
      {
          {"squeezed", "Reshaped tensor with same data as input.", "T"},
      },
      {
          {"T", types, MakeSqueezeTypeConstraintDescription(since_version)},
      });
  if (axes_is_input) {
    return schema;
  }
  return LightOpSchema(
      "Squeeze", kOnnxDomain, since_version, MakeSqueezeDoc(since_version),
      {
          {"data", "Tensors with at least max(dims) dimensions.", "T"},
      },
      {
          {"squeezed", "Reshaped tensor with same data as input.", "T"},
      },
      {
          {"T", types, MakeSqueezeTypeConstraintDescription(since_version)},
      },
      {
          {"axes",
           since_version == 1
               ? "List of non-negative integers, indicate the dimensions to squeeze."
               : "List of integers indicating the dimensions to squeeze. Negative value means "
                 "counting dimensions from the back. Accepted range is [-r, r-1] where r = "
                 "rank(data).",
           AttributeType::INTS, /*required=*/false},
      });
}

LightOpSchema MakeUnsqueezeSchema(int since_version, const std::vector<TensorType> &types) {
  if (since_version >= 13) {
    return LightOpSchema(
        "Unsqueeze", kOnnxDomain, since_version, MakeUnsqueezeDoc(since_version),
        {
            {"data", "Original tensor", "T"},
            {"axes",
             since_version >= 23
                 ? "1D tensor of integers indicating the dimensions to be inserted. Negative value "
                   "means counting dimensions from the back. Accepted range is [-r, r-1] where r "
                   "= rank(expanded)."
                 : "List of integers indicating the dimensions to be inserted. Negative value "
                   "means "
                   "counting dimensions from the back. Accepted range is [-r, r-1] where r = "
                   "rank(expanded).",
             "tensor(int64)"},
        },
        {
            {"expanded", "Reshaped tensor with same data as input.", "T"},
        },
        {
            {"T", types, MakeUnsqueezeTypeConstraintDescription(since_version)},
        });
  }
  return LightOpSchema(
      "Unsqueeze", kOnnxDomain, since_version, MakeUnsqueezeDoc(since_version),
      {
          {"data", "Original tensor", "T"},
      },
      {
          {"expanded", "Reshaped tensor with same data as input.", "T"},
      },
      {
          {"T", types, MakeUnsqueezeTypeConstraintDescription(since_version)},
      },
      {
          {"axes",
           since_version == 1
               ? "List of non-negative integers, indicate the dimensions to be inserted"
               : "List of integers indicating the dimensions to be inserted. Negative value means "
                 "counting dimensions from the back. Accepted range is [-r, r-1] where r = "
                 "rank(expanded).",
           AttributeType::INTS, /*required=*/true},
      });
}

LightOpSchema MakeNonZeroSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema("NonZero", kOnnxDomain, since_version, MakeNonZeroDoc(since_version),
                       {
                           {"X", "input", "T"},
                       },
                       {
                           {"Y", "output", "tensor(int64)"},
                       },
                       {
                           {"T", types, MakeNonZeroTypeConstraintDescription(since_version)},
                       });
}

LightOpSchema MakeUniqueSchema(int since_version, const std::vector<TensorType> &types) {
  LightOpSchema schema(
      "Unique", kOnnxDomain, since_version, MakeUniqueDoc(since_version),
      {
          {"X", "A N-D input tensor that is to be processed.", "T"},
      },
      {
          {"Y",
           "A tensor of the same type as 'X' "
           "containing all the unique values or subtensors sliced along a provided 'axis' in "
           "'X', either sorted "
           "or maintained in the same order they occur in input 'X'",
           "T"},
          {"indices",
           "A 1-D INT64 tensor "
           "containing indices of 'Y' elements' first occurrence in 'X'. "
           "When 'axis' is provided, it contains indices to subtensors in input 'X' on the "
           "'axis'. "
           "When 'axis' is not provided, it contains indices to values in the flattened input "
           "tensor. ",
           "tensor(int64)"},
          {"inverse_indices",
           "A 1-D INT64 tensor "
           "containing, for elements of 'X', its corresponding indices in 'Y'. "
           "When 'axis' is provided, it contains indices to subtensors in output 'Y' on the "
           "'axis'. "
           "When 'axis' is not provided, it contains indices to values in output 'Y'. ",
           "tensor(int64)"},
          {"counts",
           "A 1-D INT64 tensor containing "
           "the count of each element "
           "of 'Y' in input 'X'",
           "tensor(int64)"},
      },
      {
          {"T", types, MakeUniqueTypeConstraintDescription(since_version)},
      },
      {
          {"sorted",
           "(Optional) Whether to sort the unique elements in ascending order before returning "
           "as output. "
           "Must be one of 0, or 1 (default).",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
          {"axis",
           "(Optional) The dimension to apply unique. If not specified, the unique elements of "
           "the "
           "flattened input are returned. Negative value means counting dimensions "
           "from the back. Accepted range is [-r, r-1] where r = rank(input).",
           AttributeType::INT, /*required=*/false},
      });
  schema.set_min_output(1).set_max_output(4);
  return schema;
}

LightOpSchema MakeShapeSchema(int since_version, const std::vector<TensorType> &types) {
  std::vector<AttributeParam> attributes;
  if (since_version >= 15) {
    attributes.push_back({"start",
                          "(Optional) Starting axis for slicing the shape. Default value is 0."
                          "Negative value means counting dimensions from the back.",
                          AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)});
    attributes.push_back(
        {"end",
         "(Optional) Ending axis for slicing the shape. "
         "Negative value means counting dimensions from the back. "
         "If omitted, sizes of all axes upto (including) the last one will be included.",
         AttributeType::INT, /*required=*/false});
  }
  return LightOpSchema("Shape", kOnnxDomain, since_version, MakeShapeDoc(since_version),
                       {
                           {"data", "An input tensor.", "T"},
                       },
                       {
                           {"shape", "Shape of the input tensor", "T1"},
                       },
                       {
                           {"T", types, MakeShapeTypeConstraintDescription(since_version)},
                           {"T1", {TensorType::kInt64}, "Constrain output to int64 tensor."},
                       },
                       std::move(attributes));
}

LightOpSchema MakeTileSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "Tile", kOnnxDomain, since_version, MakeTileDoc(since_version),
      {
          {"input", "Input tensor of any shape.", "T"},
          {"repeats",
           "1D int64 tensor of the same length as input's dimension number, "
           "includes numbers of repeated copies along input's dimensions.",
           "T1"},
      },
      {
          {"output",
           "Output tensor of the same dimensions and type as tensor input. "
           "output_dim[i] = input_dim[i] * repeats[i]",
           "T"},
      },
      {
          {"T", types, MakeTileTypeConstraintDescription(since_version)},
          {"T1", {TensorType::kInt64}, "Constrain repeat's type to int64 tensors."},
      });
}

namespace {

// Type-constraint helper for Upsample v1: {bool, int32, int64, float16, float, double}.
std::vector<TensorType> UpsampleTypesVer1() {
  return {
      TensorType::kBool,    TensorType::kInt32, TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble,
  };
}

} // namespace

LightOpSchema MakeUpsampleSchema(int since_version) {
  if (since_version == 1) {
    return LightOpSchema(
        "Upsample", kOnnxDomain, since_version, MakeUpsampleDoc(since_version),
        {
            {"X", "4-D tensor, [N,C,H,W]", "T"},
        },
        {
            {"Y", "4-D tensor after resizing, [N,C,H,W]", "T"},
        },
        {
            {"T", UpsampleTypesVer1(), MakeUpsampleTypeConstraintDescription(since_version)},
        },
        {
            {"mode", "Two interpolation modes: nearest(default), bilinear", AttributeType::STRING,
             /*required=*/false, std::string("nearest")},
            {"width_scale",
             "The scale along width dimension. It takes value greater than or equal to 1.",
             AttributeType::FLOAT, /*required=*/true},
            {"height_scale",
             "The scale along height dimension. It takes value greater than or equal to 1.",
             AttributeType::FLOAT, /*required=*/true},
        });
  }
  if (since_version == 7) {
    return LightOpSchema(
        "Upsample", kOnnxDomain, since_version, MakeUpsampleDoc(since_version),
        {
            {"X", "N-D tensor", "T"},
        },
        {
            {"Y", "N-D tensor after resizing", "T"},
        },
        {
            {"T", AllTensorTypes(), MakeUpsampleTypeConstraintDescription(since_version)},
        },
        {
            {"mode",
             "Two interpolation modes: nearest (default), and linear (including bilinear, "
             "trilinear, etc)",
             AttributeType::STRING, /*required=*/false, std::string("nearest")},
            {"scales",
             "The scale array along each dimension. It takes value greater than or equal to 1."
             " The number of elements of 'scales' should be the same as the rank of input 'X'.",
             AttributeType::FLOATS, /*required=*/true},
        });
  }
  // v9 and v10 (v10 deprecated): same signature.
  return LightOpSchema(
      "Upsample", kOnnxDomain, since_version, MakeUpsampleDoc(since_version),
      {
          {"X", "N-D tensor", "T"},
          {"scales",
           "The scale array along each dimension. It takes value greater than or equal to 1."
           " The number of elements of 'scales' should be the same as the rank of input 'X'.",
           "tensor(float)"},
      },
      {
          {"Y", "N-D tensor after resizing", "T"},
      },
      {
          {"T", AllTensorTypes(), MakeUpsampleTypeConstraintDescription(since_version)},
      },
      {
          {"mode",
           "Two interpolation modes: nearest (default), and linear (including bilinear, "
           "trilinear, etc)",
           AttributeType::STRING, /*required=*/false, std::string("nearest")},
      });
}

LightOpSchema MakeTransposeSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "Transpose", kOnnxDomain, since_version, MakeTransposeDoc(since_version),
      {
          {"data", "An input tensor.", "T"},
      },
      {
          {"transposed", "Transposed output.", "T"},
      },
      {
          {"T", types, MakeTransposeTypeConstraintDescription(since_version)},
      },
      {
          {"perm",
           since_version >= 21
               ? "A list of integers. By default, reverse the dimensions, otherwise permute the "
                 "axes according to the values given. Its length must be equal to the rank of "
                 "the input."
               : "A list of integers. By default, reverse the dimensions, otherwise permute the "
                 "axes according to the values given.",
           AttributeType::INTS, /*required=*/false},
      });
}

LightOpSchema MakeDepthToSpaceSchema(int since_version, const std::vector<TensorType> &types) {
  std::vector<AttributeParam> attributes;
  attributes.push_back({"blocksize", "Blocks of [blocksize, blocksize] are moved.",
                        AttributeType::INT, /*required=*/true});
  if (since_version >= 11) {
    attributes.push_back({"mode",
                          "DCR (default) for depth-column-row order re-arrangement. Use CRD for "
                          "column-row-depth order.",
                          AttributeType::STRING, /*required=*/false, std::string("DCR")});
  }
  return LightOpSchema(
      "DepthToSpace", kOnnxDomain, since_version, MakeDepthToSpaceDoc(since_version),
      {
          {"input",
           "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, "
           "H is the height and W is the width.",
           "T"},
      },
      {
          {"output",
           "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].", "T"},
      },
      {
          {"T", types, MakeDepthToSpaceTypeConstraintDescription(since_version)},
      },
      std::move(attributes));
}

LightOpSchema MakeSpaceToDepthSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "SpaceToDepth", kOnnxDomain, since_version, MakeSpaceToDepthDoc(since_version),
      {
          {"input",
           "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, "
           "H is the height and W is the width.",
           "T"},
      },
      {
          {"output", "Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].",
           "T"},
      },
      {
          {"T", types, MakeSpaceToDepthTypeConstraintDescription(since_version)},
      },
      {
          {"blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeType::INT,
           /*required=*/true},
      });
}

LightOpSchema MakeGatherSchema(int since_version, const std::vector<TensorType> &types) {
  const std::string axis_desc =
      since_version >= 11
          ? "Which axis to gather on. Negative value means counting dimensions from the back. "
            "Accepted range is [-r, r-1] where r = rank(data)."
          : "Which axis to gather on. Negative value means counting dimensions from the back. "
            "Accepted range is [-r, r-1]";
  const std::string indices_desc =
      since_version >= 11
          ? "Tensor of int32/int64 indices, of any rank q. All index values are expected to be "
            "within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds."
          : "Tensor of int32/int64 indices, of any rank q. All index values are expected to be "
            "within bounds. "
            "It is an error if any of the index values are out of bounds.";
  return LightOpSchema(
      "Gather", kOnnxDomain, since_version, MakeGatherDoc(since_version),
      {
          {"data", "Tensor of rank r >= 1.", "T"},
          {"indices", indices_desc, "Tind"},
      },
      {
          {"output", "Tensor of rank q + (r - 1).", "T"},
      },
      {
          {"T", types, "Constrain input and output types to any tensor type."},
          {"Tind", {TensorType::kInt32, TensorType::kInt64}, "Constrain indices to integer types"},
      },
      {
          {"axis", axis_desc, AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
      });
}

LightOpSchema MakeGatherElementsSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "GatherElements", kOnnxDomain, since_version, MakeGatherElementsDoc(since_version),
      {
          {"data", "Tensor of rank r >= 1.", "T"},
          {"indices",
           "Tensor of int32/int64 indices, with the same rank r as the input. All index values "
           "are expected to be "
           "within bounds [-s, s-1] along axis of size s. It is an error if any of the index "
           "values are out of bounds.",
           "Tind"},
      },
      {
          {"output", "Tensor of the same shape as indices.", "T"},
      },
      {
          {"T", types, "Constrain input and output types to any tensor type."},
          {"Tind", {TensorType::kInt32, TensorType::kInt64}, "Constrain indices to integer types"},
      },
      {
          {"axis",
           "Which axis to gather on. Negative value means counting dimensions from the back. "
           "Accepted range is [-r, r-1] where r = rank(data).",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
      });
}

LightOpSchema MakeGatherNDSchema(int since_version, const std::vector<TensorType> &types) {
  std::vector<AttributeParam> attributes;
  if (since_version >= 12) {
    attributes.push_back(
        {"batch_dims",
         "The number of batch dimensions. The gather of indexing starts from dimension of "
         "data[batch_dims:]",
         AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)});
  }
  return LightOpSchema(
      "GatherND", kOnnxDomain, since_version, MakeGatherNDDoc(since_version),
      {
          {"data", "Tensor of rank r >= 1.", "T"},
          {"indices",
           "Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] "
           "along axis of size s. It is an error if any of the index values are out of bounds.",
           "tensor(int64)"},
      },
      {
          {"output", "Tensor of rank q + r - indices_shape[-1] - 1.", "T"},
      },
      {
          {"T", types, "Constrain input and output types to any tensor type."},
      },
      std::move(attributes));
}

LightOpSchema MakeTensorScatterSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "TensorScatter", kOnnxDomain, since_version, MakeTensorScatterDoc(since_version),
      {
          {"past_cache",
           "Past state cache for key or value with shape `(batch_size, D1, D2, ..., "
           "max_sequence_length, ..., Dn)`.",
           "T"},
          {"update",
           "New update tensor with shape `(batch_size, D1, D2, ..., sequence_length, ..., Dn)`.",
           "T"},
          {"write_indices",
           "Write indices for the incoming update tensor in the cache. Shape is `(batch_size,)`. "
           "Assumed to be all zeros if not provided.",
           "tensor(int64)"},
      },
      {
          {"present_cache", "Updated cache. Same shape as `past_cache`.", "T"},
      },
      {
          {"T", types, "Constrain input and output types to any tensor type."},
      },
      {
          {"axis",
           "Sequence dimension of the `past_cache` and `update` tensors. It cannot be 0 (the "
           "batch dimension). Default is -2.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(-2)},
          {"mode",
           "Write mode of cache update. Supported modes include `linear` and `circular`. "
           "`linear` mode requires "
           "write_indices+sequence_length<=max_sequence_length. For `circular` mode, the "
           "updates happen in "
           "wrap-around fashion, ie, the update index is modulo `max_sequence_length`",
           AttributeType::STRING, /*required=*/false, std::string("linear")},
      });
}

LightOpSchema MakeTriluSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "Trilu", kOnnxDomain, since_version, MakeTriluDoc(since_version),
      {
          {"input", "Input tensor of rank 2 or higher.", "T"},
          {"k",
           "A 0-D tensor containing a single value corresponding to the number diagonals above "
           "or below the main diagonal to exclude or include. "
           "Default value is 0 if it's not specified.",
           "tensor(int64)"},
      },
      {
          {"output", "Output tensor of the same type and shape as the input tensor.", "T"},
      },
      {
          {"T", types, MakeTriluTypeConstraintDescription(since_version)},
      },
      {
          {"upper",
           "Boolean. Indicates whether upper or lower part of matrix is retained. Default is "
           "true.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
      });
}

LightOpSchema MakeReverseSequenceSchema(int since_version, const std::vector<TensorType> &types) {
  return LightOpSchema(
      "ReverseSequence", kOnnxDomain, since_version, MakeReverseSequenceDoc(since_version),
      {
          {"input", "Tensor of rank r >= 2.", "T"},
          {"sequence_lens",
           "Tensor specifying lengths of the sequences in a batch. It has shape `[batch_size]`.",
           "tensor(int64)"},
      },
      {
          {"Y", "Tensor with same shape of input.", "T"},
      },
      {
          {"T", types, MakeReverseSequenceTypeConstraintDescription(since_version)},
      },
      {
          {"time_axis",
           "(Optional) Specify which axis is time axis. Must be one of 0 (default), or 1.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
          {"batch_axis",
           "(Optional) Specify which axis is batch axis. Must be one of 1 (default), or 0.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
      });
}

LightOpSchema MakeCompressSchema(int since_version, const std::vector<TensorType> &types) {
  const std::string axis_desc =
      since_version >= 11
          ? "(Optional) Axis along which to take slices. If not specified, input is flattened "
            "before elements being selected. Negative value means counting dimensions from the "
            "back. Accepted range is [-r, r-1] where r = rank(input)."
          : "(Optional) Axis along which to take slices. If not specified, "
            "input is flattened before elements being selected.";
  // The condition description differs between v9 (contains a typo "alone") and v11 ("along").
  const std::string condition_desc =
      since_version >= 11
          ? "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. "
            "Its length can be less than the input length along the axis "
            "or the flattened input size if axis is not specified. "
            "In such cases data slices or elements exceeding the condition length are discarded."
          : "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. "
            "Its length can be less than the input length alone the axis "
            "or the flattened input size if axis is not specified. "
            "In such cases data slices or elements exceeding the condition length are discarded.";
  return LightOpSchema(
      "Compress", kOnnxDomain, since_version, MakeCompressDoc(since_version),
      {
          {"input", "Tensor of rank r >= 1.", "T"},
          {"condition", condition_desc, "T1"},
      },
      {
          {"output",
           "Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.", "T"},
      },
      {
          {"T", types, MakeCompressTypeConstraintDescription(since_version)},
          {"T1", {TensorType::kBool}, "Constrain to boolean tensors."},
      },
      {
          {"axis", axis_desc, AttributeType::INT, /*required=*/false},
      });
}

LightOpSchema MakeSplitSchema(int since_version, const std::vector<TensorType> &types) {
  const std::string outputs_desc = "One or more outputs forming list of tensors after splitting";
  // Build input parameters: opset 13+ takes ``split`` as an optional input;
  // earlier opsets carry it as an attribute (or no split at all in v1).
  std::vector<FormalParameter> inputs;
  if (since_version >= 13) {
    inputs = {
        {"input", "The tensor to split", "T"},
        {"split",
         "Optional length of each output. Values should be >= 0."
         "Sum of the values must be equal to the dim value at 'axis' specified.",
         "tensor(int64)"},
    };
  } else if (since_version == 1) {
    inputs = {
        {"input", "The tensor to split", "T"},
        {"split", "Optional list of output lengths (see also arg 'split')", "T"},
    };
  } else {
    inputs = {
        {"input", "The tensor to split", "T"},
    };
  }

  const std::string output_name = since_version == 1 ? "outputs..." : "outputs";

  // Build attributes per opset version.
  std::vector<AttributeParam> attributes;
  if (since_version == 1) {
    attributes.push_back({"axis", "Which axis to split on", AttributeType::INT,
                          /*required=*/false});
    attributes.push_back({"split", "length of each output", AttributeType::INTS,
                          /*required=*/false});
  } else if (since_version == 2 || since_version == 11) {
    const std::string axis_desc =
        since_version == 2 ? "Which axis to split on. "
                           : "Which axis to split on. "
                             "A negative value means counting dimensions from the back. Accepted "
                             "range is [-rank, rank-1] where r = rank(input).";
    attributes.push_back(
        {"axis", axis_desc, AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)});
    attributes.push_back({"split", "length of each output. Values should be >= 0.",
                          AttributeType::INTS,
                          /*required=*/false});
  } else if (since_version == 13) {
    attributes.push_back({"axis",
                          "Which axis to split on. "
                          "A negative value means counting dimensions from the back. Accepted "
                          "range is [-rank, rank-1] where r = rank(input).",
                          AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)});
  } else {
    // since_version >= 18
    attributes.push_back({"axis",
                          "Which axis to split on. "
                          "A negative value means counting dimensions from the back. Accepted "
                          "range is [-rank, rank-1] where r = rank(input).",
                          AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)});
    attributes.push_back({"num_outputs",
                          "Number of outputs to split parts of the tensor into. "
                          "If the tensor is not evenly splittable the last chunk will be smaller.",
                          AttributeType::INT, /*required=*/false});
  }

  LightOpSchema schema("Split", kOnnxDomain, since_version, MakeSplitDoc(since_version),
                       std::move(inputs),
                       {
                           {output_name, outputs_desc, "T"},
                       },
                       {
                           {"T", types, MakeSplitTypeConstraintDescription(since_version)},
                       },
                       std::move(attributes));
  // ``outputs`` is variadic: at least one output, no upper bound.
  schema.set_min_output(1).set_max_output(std::numeric_limits<int>::max());
  return schema;
}

std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory(const std::string &op_type,
                                                                bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"AffineGrid", [] { return std::vector<LightOpSchema>{MakeAffineGridSchema(20)}; }},
      {"BitCast", [] { return std::vector<LightOpSchema>{MakeBitCastSchema()}; }},
      {"Cast",
       [] {
         return std::vector<LightOpSchema>{
             MakeCastSchema(1, CastTypesVer1And6()), MakeCastSchema(6, CastTypesVer1And6()),
             MakeCastSchema(9, CastTypesVer9()),     MakeCastSchema(13, CastTypesVer13()),
             MakeCastSchema(19, CastTypesVer19()),   MakeCastSchema(21, CastTypesVer21()),
             MakeCastSchema(23, CastTypesVer23()),   MakeCastSchema(24, CastTypesVer24()),
             MakeCastSchema(25, CastTypesVer25()),
         };
       }},
      {"CastLike",
       [] {
         return std::vector<LightOpSchema>{
             MakeCastLikeSchema(15, CastTypesVer13()),     MakeCastLikeSchema(19, CastTypesVer19()),
             MakeCastLikeSchema(21, CastLikeTypesVer21()), MakeCastLikeSchema(23, CastTypesVer23()),
             MakeCastLikeSchema(24, CastTypesVer24()),     MakeCastLikeSchema(25, CastTypesVer25()),
         };
       }},
      {"Compress",
       [] {
         return std::vector<LightOpSchema>{
             MakeCompressSchema(11, AllTensorTypes()),
             MakeCompressSchema(9, AllTensorTypes()),
         };
       }},
      {"Concat",
       [] {
         return std::vector<LightOpSchema>{
             MakeConcatSchema(13, ConcatTypesVer13()),
             MakeConcatSchema(11, ConcatTypesVer4And11()),
             MakeConcatSchema(4, ConcatTypesVer4And11()),
             MakeConcatSchema(1, ConcatTypesVer1()),
         };
       }},
      {"DepthToSpace",
       [] {
         return std::vector<LightOpSchema>{
             MakeDepthToSpaceSchema(13, ConcatTypesVer13()),
             MakeDepthToSpaceSchema(11, AllTensorTypes()),
             MakeDepthToSpaceSchema(1, AllTensorTypes()),
         };
       }},
      {"SpaceToDepth",
       [] {
         return std::vector<LightOpSchema>{
             MakeSpaceToDepthSchema(13, ConcatTypesVer13()),
             MakeSpaceToDepthSchema(1, AllTensorTypes()),
         };
       }},
      {"Expand",
       [] {
         return std::vector<LightOpSchema>{
             MakeExpandSchema(13, ConcatTypesVer13()),
             MakeExpandSchema(8, AllTensorTypes()),
         };
       }},
      {"Reshape",
       [] {
         return std::vector<LightOpSchema>{
             MakeReshapeSchema(25, TransposeTypesVer25()),
             MakeReshapeSchema(13, ConcatTypesVer13()),
         };
       }},
      {"Shape",
       [] {
         return std::vector<LightOpSchema>{
             MakeShapeSchema(25, TransposeTypesVer25()), MakeShapeSchema(24, TransposeTypesVer24()),
             MakeShapeSchema(23, TransposeTypesVer23()), MakeShapeSchema(21, TransposeTypesVer21()),
             MakeShapeSchema(19, ShapeTypesVer19()),     MakeShapeSchema(15, ConcatTypesVer13()),
             MakeShapeSchema(13, ConcatTypesVer13()),    MakeShapeSchema(1, AllTensorTypes()),
         };
       }},
      {"Slice",
       [] {
         return std::vector<LightOpSchema>{
             MakeSliceSchema(13, ConcatTypesVer13()),
         };
       }},
      {"Split",
       [] {
         return std::vector<LightOpSchema>{
             MakeSplitSchema(18, ConcatTypesVer13()), MakeSplitSchema(13, ConcatTypesVer13()),
             MakeSplitSchema(11, AllTensorTypes()),   MakeSplitSchema(2, AllTensorTypes()),
             MakeSplitSchema(1, FloatTypes()),
         };
       }},
      {"GridSample",
       [] {
         return std::vector<LightOpSchema>{
             MakeGridSampleSchema(22, GridSampleInputTypesVer22(), GridSampleGridTypesVer22()),
             MakeGridSampleSchema(20, AllTensorTypes(),
                                  {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble}),
             MakeGridSampleSchema(16, AllTensorTypes(),
                                  {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble}),
         };
       }},
      {"Gather",
       [] {
         return std::vector<LightOpSchema>{
             MakeGatherSchema(13, ConcatTypesVer13()),
             MakeGatherSchema(11, AllTensorTypes()),
             MakeGatherSchema(1, AllTensorTypes()),
         };
       }},
      {"GatherElements",
       [] {
         return std::vector<LightOpSchema>{
             MakeGatherElementsSchema(13, ConcatTypesVer13()),
             MakeGatherElementsSchema(11, AllTensorTypes()),
         };
       }},
      {"GatherND",
       [] {
         return std::vector<LightOpSchema>{
             MakeGatherNDSchema(13, ConcatTypesVer13()),
             MakeGatherNDSchema(12, AllTensorTypes()),
             MakeGatherNDSchema(11, AllTensorTypes()),
         };
       }},
      {"TensorScatter",
       [] {
         return std::vector<LightOpSchema>{
             MakeTensorScatterSchema(24, TransposeTypesVer24()),
         };
       }},
      {"Squeeze",
       [] {
         return std::vector<LightOpSchema>{
             MakeSqueezeSchema(25, TransposeTypesVer25()),
             MakeSqueezeSchema(24, TransposeTypesVer24()),
             MakeSqueezeSchema(23, TransposeTypesVer23()),
             MakeSqueezeSchema(21, TransposeTypesVer21()),
             MakeSqueezeSchema(13, ConcatTypesVer13()),
             MakeSqueezeSchema(11, AllTensorTypes()),
             MakeSqueezeSchema(1, AllTensorTypes()),
         };
       }},
      {"Tile",
       [] {
         return std::vector<LightOpSchema>{
             MakeTileSchema(13, ConcatTypesVer13()),
             MakeTileSchema(6, AllTensorTypes()),
         };
       }},
      {"Upsample",
       [] {
         return std::vector<LightOpSchema>{
             MakeUpsampleSchema(10),
             MakeUpsampleSchema(9),
             MakeUpsampleSchema(7),
             MakeUpsampleSchema(1),
         };
       }},
      {"NonZero",
       [] {
         return std::vector<LightOpSchema>{
             MakeNonZeroSchema(13, ConcatTypesVer13()),
             MakeNonZeroSchema(9, AllTensorTypes()),
         };
       }},
      {"Transpose",
       [] {
         return std::vector<LightOpSchema>{
             MakeTransposeSchema(25, TransposeTypesVer25()),
             MakeTransposeSchema(24, TransposeTypesVer24()),
             MakeTransposeSchema(23, TransposeTypesVer23()),
             MakeTransposeSchema(21, TransposeTypesVer21()),
             MakeTransposeSchema(13, ConcatTypesVer13()),
             MakeTransposeSchema(1, AllTensorTypes()),
         };
       }},
      {"Trilu",
       [] {
         return std::vector<LightOpSchema>{
             MakeTriluSchema(14, ConcatTypesVer13()),
         };
       }},
      {"ReverseSequence",
       [] {
         return std::vector<LightOpSchema>{
             MakeReverseSequenceSchema(10, AllTensorTypes()),
         };
       }},
      {"Unsqueeze",
       [] {
         return std::vector<LightOpSchema>{
             MakeUnsqueezeSchema(25, TransposeTypesVer25()),
             MakeUnsqueezeSchema(24, TransposeTypesVer24()),
             MakeUnsqueezeSchema(23, TransposeTypesVer23()),
             MakeUnsqueezeSchema(21, TransposeTypesVer21()),
             MakeUnsqueezeSchema(13, ConcatTypesVer13()),
             MakeUnsqueezeSchema(11, AllTensorTypes()),
             MakeUnsqueezeSchema(1, AllTensorTypes()),
         };
       }},
      {"Unique",
       [] {
         return std::vector<LightOpSchema>{
             MakeUniqueSchema(11, AllTensorTypes()),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

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

// Helpers for Identity type constraints: append all tensor sequence types
// (since opset 14) and all optional types (since opset 16) to a base
// tensor-type set. The ordering matches the upstream schema where the base
// tensor types are listed first, then sequence types, then optional types.
std::vector<TensorType> AppendSequenceTypes(std::vector<TensorType> types) {
  const std::vector<TensorType> seq = AllTensorSequenceTypes();
  types.insert(types.end(), seq.begin(), seq.end());
  return types;
}

std::vector<TensorType> AppendSequenceAndOptionalTypes(std::vector<TensorType> types) {
  const std::vector<TensorType> seq = AllTensorSequenceTypes();
  const std::vector<TensorType> opt = AllOptionalTypes();
  types.insert(types.end(), seq.begin(), seq.end());
  types.insert(types.end(), opt.begin(), opt.end());
  return types;
}

// Identity type sets per opset version (V/T type constraint):
//   v1  : OpSchema::all_tensor_types()                                       -> AllTensorTypes()
//   v13 : OpSchema::all_tensor_types_ir4()                                   -> ConcatTypesVer13()
//   v14 : all_tensor_types_ir4 + sequence
//   v16 : all_tensor_types_ir4 + sequence + optional
//   v19 : all_tensor_types_ir9 + sequence + optional
//   v21 : all_tensor_types_ir10 + sequence + optional
//   v23 : all_tensor_types_ir11 + sequence + optional
//   v24 : all_tensor_types_ir12 + sequence + optional
//   v25 : all_tensor_types_ir13 + sequence + optional
std::vector<TensorType> IdentityTypesVer14() { return AppendSequenceTypes(ConcatTypesVer13()); }
std::vector<TensorType> IdentityTypesVer16() {
  return AppendSequenceAndOptionalTypes(ConcatTypesVer13());
}
std::vector<TensorType> IdentityTypesVer19() {
  return AppendSequenceAndOptionalTypes(ShapeTypesVer19());
}
std::vector<TensorType> IdentityTypesVer21() {
  return AppendSequenceAndOptionalTypes(TransposeTypesVer21());
}
std::vector<TensorType> IdentityTypesVer23() {
  return AppendSequenceAndOptionalTypes(TransposeTypesVer23());
}
std::vector<TensorType> IdentityTypesVer24() {
  return AppendSequenceAndOptionalTypes(TransposeTypesVer24());
}
std::vector<TensorType> IdentityTypesVer25() {
  return AppendSequenceAndOptionalTypes(TransposeTypesVer25());
}

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

LightOpSchema MakePadSchema(int since_version, const std::vector<TensorType> &types) {
  const std::string mode_description = MakePadModeDescription(since_version);
  const std::string type_constraint_description = MakePadTypeConstraintDescription(since_version);
  const std::string doc = MakePadDoc(since_version);

  // Opsets 1 and 2: only one input (``data``); pads/value live as attributes.
  if (since_version <= 2) {
    const bool is_v1 = since_version == 1;
    const std::string pads_attr_name = is_v1 ? "paddings" : "pads";
    const std::string pads_attr_description =
        is_v1 ? "List of integers indicate the padding element count at the beginning and end of "
                "each axis, for 2D it is the number of pixel. `paddings` rank should be double of "
                "the input's rank. `paddings` format should be as follow [x1_begin, "
                "x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the "
                "beginning of axis `i` and xi_end, the number of pixels added at the end of axis "
                "`i`."
              : "List of integers indicating the number of padding elements to add or remove (if "
                "negative) at the beginning and end of each axis. For 2D it is the number of "
                "pixels. `pads` rank should be double of the input's rank. `pads` format should be "
                "as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of "
                "pixels added at the beginning of axis `i` and xi_end, the number of pixels added "
                "at the end of axis `i`.";
    const std::string value_description =
        is_v1 ? "One float, indicates the value to be filled, default is 0"
              : "One float, indicates the value to be filled.";
    return LightOpSchema(
        "Pad", kOnnxDomain, since_version, doc,
        {
            {"data", "Input tensor.", "T"},
        },
        {
            {"output", "Tensor after padding.", "T"},
        },
        {
            {"T", types, type_constraint_description},
        },
        {
            {"value", value_description, AttributeType::FLOAT, /*required=*/false, 0.0},
            {"mode", mode_description, AttributeType::STRING, /*required=*/false,
             std::string("constant")},
            {pads_attr_name, pads_attr_description, AttributeType::INTS, /*required=*/true,
             std::monostate{}},
        });
  }

  // Opset 11 and 13: ``pads`` and ``constant_value`` become inputs (no ``axes``).
  if (since_version <= 13) {
    const bool is_v11 = since_version == 11;
    const std::string constant_value_description =
        is_v11
            ? "(Optional) A scalar value to be used if the mode chosen is `constant` (by default "
              "it is 0)."
            : "(Optional) A scalar value to be used if the mode chosen is `constant` (by default "
              "it is 0, empty string or False).";
    return LightOpSchema(
        "Pad", kOnnxDomain, since_version, doc,
        {
            {"data", "Input tensor.", "T"},
            {"pads",
             "Tensor of integers indicating the number of padding elements to add or remove (if "
             "negative) at the beginning and end of each axis. For 2D input tensor, it is the "
             "number of pixels. `pads` should be a 1D tensor of shape [2 * input_rank]. `pads` "
             "format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], where xi_begin is "
             "the number of pad values added at the beginning of axis `i` and xi_end, the number "
             "of pad values added at the end of axis `i`.",
             "tensor(int64)"},
            {"constant_value", constant_value_description, "T"},
        },
        {
            {"output", "Tensor after padding.", "T"},
        },
        {
            {"T", types, type_constraint_description},
        },
        {
            {"mode", mode_description, AttributeType::STRING, /*required=*/false,
             std::string("constant")},
        });
  }

  // Opset 18 onwards: adds the optional ``axes`` input (and updates ``pads``
  // documentation to talk about ``num_axes``); opset 19+ further allows the
  // ``wrap`` mode (handled by ``mode_description``).
  return LightOpSchema(
      "Pad", kOnnxDomain, since_version, doc,
      {
          {"data", "Input tensor.", "T"},
          {"pads",
           "Tensor of integers indicating the number of padding elements to add or remove (if "
           "negative) at the beginning and end of each axis. For 2D input tensor, it is the "
           "number of pixels. `pads` should be a 1D tensor of shape [2 * num_axes] where "
           "`num_axes` refers to the number of elements in the `axes` input or the input rank if "
           "`axes` are not provided explicitly. `pads` format should be: [x1_begin, x2_begin, "
           "..., x1_end, x2_end,...], where xi_begin is the number of pad values added at the "
           "beginning of axis `axes[i]` and xi_end, the number of pad values added at the end of "
           "axis `axes[i]`.",
           "tensor(int64)"},
          {"constant_value",
           "(Optional) A scalar value to be used if the mode chosen is `constant` (by default it "
           "is 0, empty string or False).",
           "T"},
          {"axes",
           "1-D tensor of axes that `pads` apply to. Negative value means counting dimensions "
           "from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is "
           "undefined if an axis is repeated. If not provided, all axes are assumed (`[0, 1, "
           "..., input_rank-1]`).",
           "Tind"},
      },
      {
          {"output", "Tensor after padding.", "T"},
      },
      {
          {"T", types, type_constraint_description},
          {"Tind", {TensorType::kInt32, TensorType::kInt64}, "Constrain indices to integer types"},
      },
      {
          {"mode", mode_description, AttributeType::STRING, /*required=*/false,
           std::string("constant")},
      });
}

LightOpSchema MakeIdentitySchema(int since_version, const std::vector<TensorType> &types) {
  const std::string type_param = since_version >= 14 ? "V" : "T";
  return LightOpSchema(
      "Identity", kOnnxDomain, since_version, MakeIdentityDoc(since_version),
      {
          {"input", "Input tensor", type_param},
      },
      {
          {"output", "Tensor to copy input into.", type_param},
      },
      {
          {type_param, types, MakeIdentityTypeConstraintDescription(since_version)},
      });
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

LightOpSchema MakeOneHotSchema(int since_version, const std::vector<TensorType> &indices_types,
                               const std::vector<TensorType> &depth_types,
                               const std::vector<TensorType> &values_types) {
  const std::string indices_doc =
      since_version >= 11
          ? std::string(
                "Input tensor containing indices. Any entries in the 'indices' input tensor with "
                "values outside the range [-depth, depth-1] will result in one-hot representation "
                "with all 'off_value' values in the output tensor."
                "In case 'indices' is of non-integer type, the values will be casted to int64 "
                "before use.")
          : std::string(
                "Input tensor containing indices. The values must be non-negative integers. "
                "Any entries in the 'indices' input tensor with values outside the range "
                "[0, depth) will result in one-hot representation with all 'off_value' values in "
                "the output tensor."
                "In case 'indices' is of non-integer type, the values will be casted to int64 "
                "before use.");
  const std::string depth_doc =
      since_version >= 11
          ? std::string(
                "Scalar or Rank 1 tensor containing exactly one element, specifying the number "
                "of classes in one-hot tensor. This is also the size of the one-hot dimension "
                "(specified by 'axis' attribute) added on in the output tensor. The values in "
                "the 'indices' input tensor are expected to be in the range [-depth, depth-1]. "
                "In case 'depth' is of non-integer type, it will be casted to int64 before use.")
          : std::string(
                "Scalar or rank 1 tensor containing exactly one element, specifying the number "
                "of classes in one-hot tensor. This is also the size of the one-hot dimension "
                "(specified by 'axis' attribute) added on in the output tensor. The values in "
                "the 'indices' input tensor are expected to be in the range [0, depth). "
                "In case 'depth' is of non-integer type, it will be casted to int64 before use.");
  const std::string axis_doc =
      since_version >= 11
          ? std::string(
                "(Optional) Axis along which one-hot representation in added. Default: axis=-1. "
                "axis=-1 means that the additional dimension will be inserted as the "
                "innermost/last dimension in the output tensor. Negative value means counting "
                "dimensions from the back. Accepted range is [-r-1, r] where r = rank(indices).")
          : std::string(
                "(Optional) Axis along which one-hot representation in added. Default: axis=-1. "
                "axis=-1 means that the additional dimension will be inserted as the "
                "innermost/last dimension in the output tensor.");
  return LightOpSchema(
      "OneHot", kOnnxDomain, since_version, MakeOneHotDoc(since_version),
      {
          {"indices", indices_doc, "T1"},
          {"depth", depth_doc, "T2"},
          {"values",
           "Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], "
           "where 'on_value' is the value used for filling locations specified in 'indices' input "
           "tensor, and 'off_value' is the value used for filling locations other than those "
           "specified in 'indices' input tensor. ",
           "T3"},
      },
      {
          {"output",
           "Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = "
           "rank(indices) + 1. The data type for the elements of the output tensor is the same "
           "as the type of input 'values' is used.",
           "T3"},
      },
      {
          {"T1", indices_types, MakeOneHotIndicesTypeConstraintDescription(since_version)},
          {"T2", depth_types, MakeOneHotDepthTypeConstraintDescription(since_version)},
          {"T3", values_types, MakeOneHotValuesTypeConstraintDescription(since_version)},
      },
      {
          {"axis", axis_doc, AttributeType::INT, /*required=*/false,
           static_cast<int64_t>(-1)},
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

namespace {

// Type-constraint helper for Resize ``T2`` (input ``roi``):
// {float16, float, double}. Same for all opset versions starting at v11.
std::vector<TensorType> ResizeRoiTypes() {
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

} // namespace

LightOpSchema MakeResizeSchema(int since_version) {
  // Common parameter description strings shared across multiple opset versions.
  // Each multi-line string here is the concatenation that appears in
  // ``onnx_lib/defs/tensor/old.cc`` and ``defs.cc`` (one space between
  // adjacent pieces). The OnnxOpSchemaParityTest enforces an exact match,
  // so do not modify the wording without updating onnx_lib accordingly.
  if (since_version == 10) {
    return LightOpSchema(
        "Resize", kOnnxDomain, since_version, MakeResizeDoc(since_version),
        {
            {"X", "N-D tensor", "T"},
            {"scales",
             "The scale array along each dimension. It takes value greater than 0. If it's less "
             "than 1, it's sampling down, otherwise, it's upsampling. The number of elements of "
             "'scales' should be the same as the rank of input 'X'.",
             "tensor(float)"},
        },
        {
            {"Y", "N-D tensor after resizing", "T"},
        },
        {
            {"T", AllTensorTypes(), MakeResizeT1TypeConstraintDescription(since_version)},
        },
        {
            {"mode",
             "Two interpolation modes: nearest (default), and linear (including bilinear, "
             "trilinear, etc)",
             AttributeType::STRING, /*required=*/false, std::string("nearest")},
        });
  }
  if (since_version == 11) {
    return LightOpSchema(
        "Resize", kOnnxDomain, since_version, MakeResizeDoc(since_version),
        {
            {"X", "N-D tensor", "T1"},
            {"roi",
             "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of "
             "X. The RoIs' coordinates are normalized in the coordinate system of the input "
             "image. It only takes effect when coordinate_transformation_mode is "
             "\"tf_crop_and_resize\"",
             "T2"},
            {"scales",
             "The scale array along each dimension. It takes value greater than 0. If it's less "
             "than 1, it's sampling down, otherwise, it's upsampling. The number of elements of "
             "'scales' should be the same as the rank of input 'X'. If 'size' is needed, the "
             "user must set 'scales' to an empty tensor.",
             "tensor(float)"},
            {"sizes",
             "The size of the output tensor. The number of elements of 'sizes' should be the "
             "same as the rank of input 'X'. May only be set if 'scales' is set to an empty "
             "tensor.",
             "tensor(int64)"},
        },
        {
            {"Y", "N-D tensor after resizing", "T1"},
        },
        {
            {"T1", AllTensorTypes(), MakeResizeT1TypeConstraintDescription(since_version)},
            {"T2", ResizeRoiTypes(), MakeResizeT2TypeConstraintDescription(since_version)},
        },
        {
            {"mode",
             "Three interpolation modes: nearest (default), linear and cubic. "
             "The \"linear\" mode includes linear interpolation for 1D tensor and N-linear "
             "interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). "
             "The \"cubic\" mode includes cubic interpolation for 1D tensor and N-cubic "
             "interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).",
             AttributeType::STRING, /*required=*/false, std::string("nearest")},
            {"cubic_coeff_a",
             "The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in "
             "some cases of TensorFlow) and -0.75 (in PyTorch). Check out Equation (4) in "
             "https://ieeexplore.ieee.org/document/1163711 for the details. "
             "This attribute is valid only if \"mode\" is \"cubic\".",
             AttributeType::FLOAT, /*required=*/false, static_cast<double>(-0.75)},
            {"exclude_outside",
             "If set to 1, the weight of sampling locations outside the tensor will be set to 0"
             " and the weight will be renormalized so that their sum is 1.0. The default value "
             "is 0.",
             AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
            {"coordinate_transformation_mode",
             "Describes how to transform coordinates between the resized and original tensor "
             "(e.g. half_pixel, asymmetric, align_corners, pytorch_half_pixel, "
             "tf_crop_and_resize, tf_half_pixel_for_nn).",
             AttributeType::STRING, /*required=*/false, std::string("half_pixel")},
            {"nearest_mode",
             "Four modes: round_prefer_floor (default, as known as round half down), "
             "round_prefer_ceil (as known as round half up), floor, ceil. Only used by nearest "
             "interpolation. It indicates how to get \"nearest\" pixel in input tensor from "
             "x_original, so this attribute is valid only if \"mode\" is \"nearest\".",
             AttributeType::STRING, /*required=*/false, std::string("round_prefer_floor")},
            {"extrapolation_value",
             "When coordinate_transformation_mode is \"tf_crop_and_resize\" and x_original is "
             "outside the range [0, length_original - 1], this value is used as the "
             "corresponding output value. Default is 0.0f.",
             AttributeType::FLOAT, /*required=*/false, static_cast<double>(0)},
        });
  }
  if (since_version == 13) {
    return LightOpSchema(
        "Resize", kOnnxDomain, since_version, MakeResizeDoc(since_version),
        {
            {"X", "N-D tensor", "T1"},
            {"roi",
             "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of "
             "X. The RoIs' coordinates are normalized in the coordinate system of the input "
             "image. It only takes effect when coordinate_transformation_mode is "
             "\"tf_crop_and_resize\"",
             "T2"},
            {"scales",
             "The scale array along each dimension. It takes value greater than 0. If it's less "
             "than 1, it's sampling down, otherwise, it's upsampling. The number of elements of "
             "'scales' should be the same as the rank of input 'X'. One of 'scales' and 'sizes' "
             "MUST be specified and it is an error if both are specified. If 'sizes' is needed, "
             "the user can use an empty string as the name of 'scales' in this operator's input "
             "list.",
             "tensor(float)"},
            {"sizes",
             "The size of the output tensor. The number of elements of 'sizes' should be the "
             "same as the rank of input 'X'. Only one of 'scales' and 'sizes' can be specified.",
             "tensor(int64)"},
        },
        {
            {"Y", "N-D tensor after resizing", "T1"},
        },
        {
            {"T1", ConcatTypesVer13(), MakeResizeT1TypeConstraintDescription(since_version)},
            {"T2", ResizeRoiTypes(), MakeResizeT2TypeConstraintDescription(since_version)},
        });
  }
  // v18 and v19 share the same set of formal parameters; only the doc body and
  // a couple of attribute defaults differ (the parity test does not check
  // attributes, only inputs/outputs/type constraints).
  return LightOpSchema(
      "Resize", kOnnxDomain, since_version, MakeResizeDoc(since_version),
      {
          {"X", "N-D tensor", "T1"},
          {"roi",
           "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X "
           "or the length of axes, if provided. The RoIs' coordinates are normalized in the "
           "coordinate system of the input image. It only takes effect when "
           "coordinate_transformation_mode is \"tf_crop_and_resize\"",
           "T2"},
          {"scales",
           "The scale array along each dimension. It takes value greater than 0. If it's less "
           "than 1, it's sampling down, otherwise, it's upsampling. The number of elements of "
           "'scales' should be the same as the rank of input 'X' or the length of 'axes', if "
           "provided. One of 'scales' and 'sizes' MUST be specified and it is an error if both "
           "are specified. If 'sizes' is needed, the user can use an empty string as the name of "
           "'scales' in this operator's input list.",
           "tensor(float)"},
          {"sizes",
           "Target size of the output tensor. Its interpretation depends on the "
           "'keep_aspect_ratio_policy' value.The number of elements of 'sizes' should be the "
           "same as the rank of input 'X', or the length of 'axes', if provided. Only one of "
           "'scales' and 'sizes' can be specified. ",
           "tensor(int64)"},
      },
      {
          {"Y", "N-D tensor after resizing", "T1"},
      },
      {
          {"T1", ConcatTypesVer13(), MakeResizeT1TypeConstraintDescription(since_version)},
          {"T2", ResizeRoiTypes(), MakeResizeT2TypeConstraintDescription(since_version)},
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

LightOpSchema MakeScatterElementsSchema(int since_version, const std::vector<TensorType> &types) {
  std::vector<AttributeParam> attributes;
  attributes.push_back(
      {"axis",
       "Which axis to scatter on. Negative value means counting dimensions from the back. "
       "Accepted range is [-r, r-1] where r = rank(data).",
       AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)});
  if (since_version >= 16) {
    std::string reduction_desc;
    if (since_version >= 18) {
      reduction_desc = "Type of reduction to apply: none (default), add, mul, max, min. "
                       "'none': no reduction applied. "
                       "'add':  reduction using the addition operation. "
                       "'mul': reduction using the multiplication operation."
                       "'max': reduction using the maximum operation."
                       "'min': reduction using the minimum operation.";
    } else {
      reduction_desc = "Type of reduction to apply: none (default), add, mul. "
                       "'none': no reduction applied. "
                       "'add':  reduction using the addition operation. "
                       "'mul': reduction using the multiplication operation.";
    }
    attributes.push_back({"reduction", reduction_desc, AttributeType::STRING,
                          /*required=*/false, std::string("none")});
  }
  return LightOpSchema(
      "ScatterElements", kOnnxDomain, since_version, MakeScatterElementsDoc(since_version),
      {
          {"data", "Tensor of rank r >= 1.", "T"},
          {"indices",
           "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values "
           "are expected to be "
           "within bounds [-s, s-1] along axis of size s. It is an error if any of the index "
           "values are out of bounds.",
           "Tind"},
          {"updates", "Tensor of rank r >=1 (same rank and shape as indices)", "T"},
      },
      {
          {"output", "Tensor of rank r >= 1 (same rank as input).", "T"},
      },
      {
          {"T", types, "Input and output types can be of any tensor type."},
          {"Tind", {TensorType::kInt32, TensorType::kInt64}, "Constrain indices to integer types"},
      },
      std::move(attributes));
}

LightOpSchema MakeScatterNDSchema(int since_version, const std::vector<TensorType> &types) {
  std::vector<AttributeParam> attributes;
  if (since_version >= 16) {
    std::string reduction_desc;
    if (since_version >= 18) {
      reduction_desc = "Type of reduction to apply: none (default), add, mul, max, min. "
                       "'none': no reduction applied. "
                       "'add':  reduction using the addition operation. "
                       "'mul':  reduction using the addition operation. "
                       "'max': reduction using the maximum operation."
                       "'min': reduction using the minimum operation.";
    } else {
      reduction_desc = "Type of reduction to apply: none (default), add, mul. "
                       "'none': no reduction applied. "
                       "'add':  reduction using the addition operation. "
                       "'mul': reduction using the multiplication operation.";
    }
    attributes.push_back({"reduction", reduction_desc, AttributeType::STRING,
                          /*required=*/false, std::string("none")});
  }
  return LightOpSchema("ScatterND", kOnnxDomain, since_version, MakeScatterNDDoc(since_version),
                       {
                           {"data", "Tensor of rank r >= 1.", "T"},
                           {"indices", "Tensor of rank q >= 1.", "tensor(int64)"},
                           {"updates", "Tensor of rank q + r - indices_shape[-1] - 1.", "T"},
                       },
                       {
                           {"output", "Tensor of rank r >= 1.", "T"},
                       },
                       {
                           {"T", types, "Constrain input and output types to any tensor type."},
                       },
                       std::move(attributes));
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
      {"Identity",
       [] {
         return std::vector<LightOpSchema>{
             MakeIdentitySchema(25, IdentityTypesVer25()),
             MakeIdentitySchema(24, IdentityTypesVer24()),
             MakeIdentitySchema(23, IdentityTypesVer23()),
             MakeIdentitySchema(21, IdentityTypesVer21()),
             MakeIdentitySchema(19, IdentityTypesVer19()),
             MakeIdentitySchema(16, IdentityTypesVer16()),
             MakeIdentitySchema(14, IdentityTypesVer14()),
             MakeIdentitySchema(13, ConcatTypesVer13()),
             MakeIdentitySchema(1, AllTensorTypes()),
         };
       }},
      {"Slice",
       [] {
         return std::vector<LightOpSchema>{
             MakeSliceSchema(13, ConcatTypesVer13()),
         };
       }},
      {"Pad",
       [] {
         return std::vector<LightOpSchema>{
             MakePadSchema(25, TransposeTypesVer25()), MakePadSchema(24, TransposeTypesVer24()),
             MakePadSchema(23, TransposeTypesVer23()), MakePadSchema(21, TransposeTypesVer21()),
             MakePadSchema(19, ConcatTypesVer13()),    MakePadSchema(18, ConcatTypesVer13()),
             MakePadSchema(13, ConcatTypesVer13()),    MakePadSchema(11, AllNumericTypes()),
             MakePadSchema(2, FloatTypes()),           MakePadSchema(1, FloatTypes()),
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
      {"ScatterElements",
       [] {
         return std::vector<LightOpSchema>{
             MakeScatterElementsSchema(18, ConcatTypesVer13()),
             MakeScatterElementsSchema(16, ConcatTypesVer13()),
             MakeScatterElementsSchema(13, ConcatTypesVer13()),
             MakeScatterElementsSchema(11, AllTensorTypes()),
         };
       }},
      {"ScatterND",
       [] {
         return std::vector<LightOpSchema>{
             MakeScatterNDSchema(18, ConcatTypesVer13()),
             MakeScatterNDSchema(16, ConcatTypesVer13()),
             MakeScatterNDSchema(13, ConcatTypesVer13()),
             MakeScatterNDSchema(11, AllTensorTypes()),
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
      {"Resize",
       [] {
         return std::vector<LightOpSchema>{
             MakeResizeSchema(19), MakeResizeSchema(18), MakeResizeSchema(13),
             MakeResizeSchema(11), MakeResizeSchema(10),
         };
       }},
      {"NonZero",
       [] {
         return std::vector<LightOpSchema>{
             MakeNonZeroSchema(13, ConcatTypesVer13()),
             MakeNonZeroSchema(9, AllTensorTypes()),
         };
       }},
      {"OneHot",
       [] {
         return std::vector<LightOpSchema>{
             MakeOneHotSchema(11, AllNumericTypes(), AllNumericTypes(), AllTensorTypes()),
             MakeOneHotSchema(9, AllNumericTypes(), AllNumericTypes(), AllTensorTypes()),
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

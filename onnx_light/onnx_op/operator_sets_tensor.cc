// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_tensor_doc.h"

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

std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory(bool init_doc,
                                                                const std::string &op_type) {
  std::vector<LightOpSchema> schemas{
      MakeAffineGridSchema(20),
      MakeCastSchema(1, CastTypesVer1And6()),
      MakeCastSchema(6, CastTypesVer1And6()),
      MakeCastSchema(9, CastTypesVer9()),
      MakeCastSchema(13, CastTypesVer13()),
      MakeCastSchema(19, CastTypesVer19()),
      MakeCastSchema(21, CastTypesVer21()),
      MakeCastSchema(23, CastTypesVer23()),
      MakeCastSchema(24, CastTypesVer24()),
      MakeCastSchema(25, CastTypesVer25()),
      MakeConcatSchema(13, ConcatTypesVer13()),
      MakeConcatSchema(11, ConcatTypesVer4And11()),
      MakeConcatSchema(4, ConcatTypesVer4And11()),
      MakeConcatSchema(1, ConcatTypesVer1()),
  };
  return FilterSchemasByOpType(init_doc ? schemas : StripDocs(schemas), op_type);
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

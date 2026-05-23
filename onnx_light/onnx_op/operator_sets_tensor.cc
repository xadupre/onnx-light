// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_tensor_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

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

std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      MakeCastSchema(1, CastTypesVer1And6()),       MakeCastSchema(6, CastTypesVer1And6()),
      MakeCastSchema(9, CastTypesVer9()),           MakeCastSchema(13, CastTypesVer13()),
      MakeCastSchema(19, CastTypesVer19()),         MakeCastSchema(21, CastTypesVer21()),
      MakeCastSchema(23, CastTypesVer23()),         MakeCastSchema(24, CastTypesVer24()),
      MakeCastSchema(25, CastTypesVer25()),         MakeConcatSchema(13, ConcatTypesVer13()),
      MakeConcatSchema(11, ConcatTypesVer4And11()), MakeConcatSchema(4, ConcatTypesVer4And11()),
      MakeConcatSchema(1, ConcatTypesVer1()),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

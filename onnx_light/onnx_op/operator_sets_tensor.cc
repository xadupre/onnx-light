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

std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory() {
  return std::vector<LightOpSchema>{
      MakeCastSchema(1, CastTypesVer1And6()), MakeCastSchema(6, CastTypesVer1And6()),
      MakeCastSchema(9, CastTypesVer9()),     MakeCastSchema(13, CastTypesVer13()),
      MakeCastSchema(19, CastTypesVer19()),   MakeCastSchema(21, CastTypesVer21()),
      MakeCastSchema(23, CastTypesVer23()),   MakeCastSchema(24, CastTypesVer24()),
      MakeCastSchema(25, CastTypesVer25()),
  };
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

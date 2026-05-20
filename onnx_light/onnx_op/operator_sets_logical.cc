// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_logical_utils.h"

#include <initializer_list>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {
namespace {

constexpr const char *kOnnxDomain = "ai.onnx";
using TensorType = onnx_op::math::TensorType;

std::vector<std::string>
TypesToStrings(const std::initializer_list<onnx_op::math::TensorType> types) {
  std::vector<std::string> allowed_types;
  allowed_types.reserve(types.size());
  for (const onnx_op::math::TensorType tensor_type : types) {
    allowed_types.emplace_back(onnx_op::math::ToTypeString(tensor_type));
  }
  return allowed_types;
}

std::vector<FormalParameter> BuildBinaryLogicalInputs(int since_version) {
  if (since_version == 1) {
    return {
        {"A", "Left input tensor for the logical operator.", "T"},
        {"B", "Right input tensor for the logical operator.", "T"},
    };
  }

  return {
      {"A", "First input operand for the logical operator.", "T"},
      {"B", "Second input operand for the logical operator.", "T"},
  };
}

std::vector<TypeConstraintParam> BuildGreaterOrLessTypeConstraints(int since_version) {
  if (since_version == 1 || since_version == 7) {
    return {
        {"T",
         TypesToStrings({
             TensorType::kFloat16,
             TensorType::kFloat,
             TensorType::kDouble,
         }),
         "Constrain input to float tensors."},
        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
    };
  }

  if (since_version == 9) {
    return {
        {"T",
         TypesToStrings({
             TensorType::kUint8,
             TensorType::kUint16,
             TensorType::kUint32,
             TensorType::kUint64,
             TensorType::kInt8,
             TensorType::kInt16,
             TensorType::kInt32,
             TensorType::kInt64,
             TensorType::kFloat16,
             TensorType::kFloat,
             TensorType::kDouble,
         }),
         "Constrain input types to all numeric tensors."},
        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
    };
  }

  return {
      {"T",
       TypesToStrings({
           TensorType::kUint8,
           TensorType::kUint16,
           TensorType::kUint32,
           TensorType::kUint64,
           TensorType::kInt8,
           TensorType::kInt16,
           TensorType::kInt32,
           TensorType::kInt64,
           TensorType::kFloat16,
           TensorType::kFloat,
           TensorType::kDouble,
           TensorType::kBfloat16,
       }),
       "Constrain input types to all numeric tensors."},
      {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
  };
}

std::vector<TypeConstraintParam> BuildEqualTypeConstraints(int since_version) {
  if (since_version == 1 || since_version == 7) {
    return {
        {"T",
         {
             "tensor(bool)",
             "tensor(int32)",
             "tensor(int64)",
         },
         "Constrain input to integral tensors."},
        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
    };
  }

  if (since_version == 11) {
    return {
        {"T",
         {
             "tensor(bool)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)",
         },
         "Constrain input types to all numeric tensors."},
        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
    };
  }

  if (since_version == 13) {
    return {
        {"T",
         {
             "tensor(bool)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)",
         },
         "Constrain input types to all numeric tensors."},
        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
    };
  }

  return {
      {"T",
       {
           "tensor(bool)",
           "tensor(uint8)",
           "tensor(uint16)",
           "tensor(uint32)",
           "tensor(uint64)",
           "tensor(int8)",
           "tensor(int16)",
           "tensor(int32)",
           "tensor(int64)",
           "tensor(float16)",
           "tensor(float)",
           "tensor(double)",
           "tensor(bfloat16)",
           "tensor(string)",
       },
       "Constrain input types to all (non-complex) tensors."},
      {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
  };
}

LightOpSchema BuildBinaryLogicalSchema(const char *name, int since_version,
                                       const std::vector<TypeConstraintParam> &type_constraints) {
  return LightOpSchema(name, kOnnxDomain, since_version,
                       detail::BuildLogicalOperatorDoc(name, since_version),
                       BuildBinaryLogicalInputs(since_version),
                       {
                           {"C", "Result tensor.", "T1"},
                       },
                       type_constraints);
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(15);
  schemas.push_back(LightOpSchema(
      "And", kOnnxDomain, 1, detail::BuildLogicalOperatorDoc("and", 1), BuildBinaryLogicalInputs(1),
      {
          {"C", "Result tensor.", "T1"},
      },
      {
          {"T", {"tensor(bool)"}, "Constrain input to boolean tensor."},
          {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
      }));
  schemas.push_back(LightOpSchema(
      "And", kOnnxDomain, 7, detail::BuildLogicalOperatorDoc("and", 7), BuildBinaryLogicalInputs(7),
      {
          {"C", "Result tensor.", "T1"},
      },
      {
          {"T", {"tensor(bool)"}, "Constrain input to boolean tensor."},
          {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
      }));
  schemas.push_back(BuildBinaryLogicalSchema("Greater", 1, BuildGreaterOrLessTypeConstraints(1)));
  schemas.push_back(BuildBinaryLogicalSchema("Greater", 7, BuildGreaterOrLessTypeConstraints(7)));
  schemas.push_back(BuildBinaryLogicalSchema("Greater", 9, BuildGreaterOrLessTypeConstraints(9)));
  schemas.push_back(BuildBinaryLogicalSchema("Greater", 13, BuildGreaterOrLessTypeConstraints(13)));
  schemas.push_back(BuildBinaryLogicalSchema("Less", 1, BuildGreaterOrLessTypeConstraints(1)));
  schemas.push_back(BuildBinaryLogicalSchema("Less", 7, BuildGreaterOrLessTypeConstraints(7)));
  schemas.push_back(BuildBinaryLogicalSchema("Less", 9, BuildGreaterOrLessTypeConstraints(9)));
  schemas.push_back(BuildBinaryLogicalSchema("Less", 13, BuildGreaterOrLessTypeConstraints(13)));
  schemas.push_back(BuildBinaryLogicalSchema("Equal", 1, BuildEqualTypeConstraints(1)));
  schemas.push_back(BuildBinaryLogicalSchema("Equal", 7, BuildEqualTypeConstraints(7)));
  schemas.push_back(BuildBinaryLogicalSchema("Equal", 11, BuildEqualTypeConstraints(11)));
  schemas.push_back(BuildBinaryLogicalSchema("Equal", 13, BuildEqualTypeConstraints(13)));
  schemas.push_back(BuildBinaryLogicalSchema("Equal", 19, BuildEqualTypeConstraints(19)));
  return schemas;
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

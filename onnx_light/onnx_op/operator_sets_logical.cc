// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_logical_utils.h"

#include <initializer_list>
#include <utility>
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

std::vector<TypeConstraintParam> BuildBooleanBinaryTypeConstraints() {
  return {
      {"T", {"tensor(bool)"}, "Constrain input to boolean tensor."},
      {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
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

LightOpSchema BuildBinaryLogicalSchema(const char *op_type, const char *op_name, int since_version,
                                       const std::vector<TypeConstraintParam> &type_constraints) {
  return LightOpSchema(op_type, kOnnxDomain, since_version,
                       detail::MakeBinaryLogicalOperatorDoc(op_name, since_version),
                       BuildBinaryLogicalInputs(since_version),
                       {
                           {"C", "Result tensor.", "T1"},
                       },
                       type_constraints);
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory() {
  const std::vector<int> binary_versions{1, 7};
  const std::vector<std::pair<const char *, const char *>> binary_ops{
      {"And", "and"},
      {"Or", "or"},
      {"Xor", "xor"},
  };
  const std::vector<int> greater_or_less_versions{1, 7, 9, 13};
  const std::vector<int> equal_versions{1, 7, 11, 13, 19};

  std::vector<LightOpSchema> schemas;
  schemas.reserve(20);

  for (const auto &[op_type, op_name] : binary_ops) {
    for (const int version : binary_versions) {
      schemas.push_back(
          BuildBinaryLogicalSchema(op_type, op_name, version, BuildBooleanBinaryTypeConstraints()));
    }
  }

  schemas.push_back(
      LightOpSchema("Not", kOnnxDomain, 1, detail::MakeNotLogicalOperatorDoc(),
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", {"tensor(bool)"}, "Constrain input/output to boolean tensors."},
                    }));

  for (const int version : greater_or_less_versions) {
    schemas.push_back(BuildBinaryLogicalSchema("Greater", "Greater", version,
                                               BuildGreaterOrLessTypeConstraints(version)));
  }
  for (const int version : greater_or_less_versions) {
    schemas.push_back(BuildBinaryLogicalSchema("Less", "Less", version,
                                               BuildGreaterOrLessTypeConstraints(version)));
  }
  for (const int version : equal_versions) {
    schemas.push_back(
        BuildBinaryLogicalSchema("Equal", "Equal", version, BuildEqualTypeConstraints(version)));
  }

  return schemas;
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

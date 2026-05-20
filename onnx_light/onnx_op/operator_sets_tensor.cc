// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

std::vector<std::string> CastTypesV1V6() {
  return {"tensor(float16)", "tensor(float)",  "tensor(double)", "tensor(int8)",
          "tensor(int16)",   "tensor(int32)",  "tensor(int64)",  "tensor(uint8)",
          "tensor(uint16)",  "tensor(uint32)", "tensor(uint64)", "tensor(bool)"};
}

std::vector<std::string> CastTypesV9() {
  return {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)",   "tensor(int16)",
          "tensor(int32)",   "tensor(int64)", "tensor(uint8)",  "tensor(uint16)", "tensor(uint32)",
          "tensor(uint64)",  "tensor(bool)",  "tensor(string)"};
}

std::vector<std::string> CastTypesV13() {
  return {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)",    "tensor(int16)",
          "tensor(int32)",   "tensor(int64)", "tensor(uint8)",  "tensor(uint16)",  "tensor(uint32)",
          "tensor(uint64)",  "tensor(bool)",  "tensor(string)", "tensor(bfloat16)"};
}

std::vector<std::string> CastTypesV19() {
  return {"tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(int8)",           "tensor(int16)",      "tensor(int32)",
          "tensor(int64)",          "tensor(uint8)",      "tensor(uint16)",
          "tensor(uint32)",         "tensor(uint64)",     "tensor(bool)",
          "tensor(string)",         "tensor(bfloat16)",   "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)"};
}

std::vector<std::string> CastTypesV21() {
  return {"tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(int8)",           "tensor(int16)",      "tensor(int32)",
          "tensor(int64)",          "tensor(uint8)",      "tensor(uint16)",
          "tensor(uint32)",         "tensor(uint64)",     "tensor(bool)",
          "tensor(string)",         "tensor(bfloat16)",   "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)"};
}

std::vector<std::string> CastTypesV23() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)"};
}

std::vector<std::string> CastTypesV24() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
          "tensor(float8e8m0)"};
}

std::vector<std::string> CastTypesV25() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
          "tensor(float8e8m0)",     "tensor(uint2)",      "tensor(int2)"};
}

void AddCastSchema(std::vector<LightOpSchema> &schemas, int version,
                   const std::vector<std::string> &types, const char *input_constraint_description,
                   const char *output_constraint_description) {
  schemas.push_back(LightOpSchema("Cast", kOnnxDomain, version,
                                  "Casts the elements of an input tensor to a specified data type.",
                                  {
                                      {"input", "Input tensor to be cast.", "T1"},
                                  },
                                  {
                                      {"output",
                                       "Output tensor with the same shape as input with type "
                                       "specified by the 'to' argument",
                                       "T2"},
                                  },
                                  {
                                      {"T1", types, input_constraint_description},
                                      {"T2", types, output_constraint_description},
                                  }));
}

std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(9);
  const char *legacy_input_constraint =
      "Constrain input types. Casting from strings and complex are not supported.";
  const char *legacy_output_constraint =
      "Constrain output types. Casting to strings and complex are not supported.";
  const char *input_constraint = "Constrain input types. Casting from complex is not supported.";
  const char *output_constraint = "Constrain output types. Casting to complex is not supported.";
  AddCastSchema(schemas, 1, CastTypesV1V6(), legacy_input_constraint, legacy_output_constraint);
  AddCastSchema(schemas, 6, CastTypesV1V6(), legacy_input_constraint, legacy_output_constraint);
  AddCastSchema(schemas, 9, CastTypesV9(), input_constraint, output_constraint);
  AddCastSchema(schemas, 13, CastTypesV13(), input_constraint, output_constraint);
  AddCastSchema(schemas, 19, CastTypesV19(), input_constraint, output_constraint);
  AddCastSchema(schemas, 21, CastTypesV21(), input_constraint, output_constraint);
  AddCastSchema(schemas, 23, CastTypesV23(), input_constraint, output_constraint);
  AddCastSchema(schemas, 24, CastTypesV24(), input_constraint, output_constraint);
  AddCastSchema(schemas, 25, CastTypesV25(), input_constraint, output_constraint);
  return schemas;
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

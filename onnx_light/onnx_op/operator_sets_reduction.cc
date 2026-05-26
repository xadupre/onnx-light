// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_reduction.h"
#include "onnx_op/operator_sets_reduction_doc.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace reduction {

namespace {

// Empty-set values used in the per-operator documentation strings.
constexpr const char *kEmptyZero = "0";
constexpr const char *kEmptyOne = "1";
constexpr const char *kEmptyUndefined = "undefined";
constexpr const char *kEmptyMin = "minus infinity (if supported by the datatype) or the minimum "
                                  "value of the data type otherwise";
constexpr const char *kEmptyMax = "plus infinity (if supported by the datatype) or the maximum "
                                  "value of the data type otherwise";
constexpr const char *kEmptyMinusInf =
    "minus infinity (if supported by the datatype) or undefined otherwise";

// Numeric type sets for ReduceMax/ReduceMin (which gained 8-bit and boolean support
// in opset 12 and 20 respectively).
std::vector<TensorType> NumericTypesForMaxMinIr4WithInt8() {
  std::vector<TensorType> types = NumericTypesForMathReductionIr4();
  types.push_back(TensorType::kUint8);
  types.push_back(TensorType::kInt8);
  return types;
}

std::vector<TensorType> NumericTypesForMaxMinWithInt8() {
  std::vector<TensorType> types = NumericTypesForMathReduction();
  types.push_back(TensorType::kUint8);
  types.push_back(TensorType::kInt8);
  return types;
}

std::vector<TensorType> NumericTypesForMaxMinIr4WithInt8AndBool() {
  std::vector<TensorType> types = NumericTypesForMaxMinIr4WithInt8();
  types.push_back(TensorType::kBool);
  return types;
}

// Common attribute definitions for Reduce* operators.
constexpr const char *kKeepdimsDesc =
    "Keep the reduced dimension or not, default 1 means keep reduced dimension.";
constexpr const char *kAxesAttrDesc =
    "A list of integers, along which to reduce. The default is to reduce over all the "
    "dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data).";
constexpr const char *kNoopWithEmptyAxesDesc =
    "Defines behavior if 'axes' is empty. Default behavior with 'false' is to reduce all "
    "axes. When axes is empty and this attribute is set to true, input tensor will not be "
    "reduced, and the output tensor would be equivalent to input tensor.";

AttributeParam MakeKeepdimsAttr() {
  return AttributeParam{"keepdims", kKeepdimsDesc, AttributeType::INT, false, int64_t{1}};
}

AttributeParam MakeAxesAttr() {
  return AttributeParam{"axes", kAxesAttrDesc, AttributeType::INTS, false, std::monostate{}};
}

AttributeParam MakeNoopWithEmptyAxesAttr() {
  return AttributeParam{"noop_with_empty_axes", kNoopWithEmptyAxesDesc, AttributeType::INT, false,
                        int64_t{0}};
}

// Builds a Reduce* schema entry whose axes are an attribute (used for opsets <= 11
// or 13 for most ops and opsets <= 13 for ReduceMax/ReduceMin).
LightOpSchema MakeReduceAttrSchema(const std::string &op_type, int since_version,
                                   const std::string &doc,
                                   const std::vector<TensorType> &allowed_types,
                                   const std::string &type_constraint_desc) {
  return LightOpSchema(op_type, kOnnxDomain, since_version, doc,
                       {
                           {"data", "An input tensor.", "T"},
                       },
                       {
                           {"reduced", "Reduced output tensor.", "T"},
                       },
                       {
                           {"T", allowed_types, type_constraint_desc},
                       },
                       {
                           MakeAxesAttr(),
                           MakeKeepdimsAttr(),
                       });
}

// Builds a Reduce* schema entry whose axes are an optional second input tensor
// (used for opsets >= 18 for most ops, >= 13 for ReduceSum).
LightOpSchema MakeReduceAxesInputSchema(const std::string &op_type, int since_version,
                                        const std::string &doc,
                                        const std::vector<TensorType> &allowed_types,
                                        const std::string &type_constraint_desc) {
  return LightOpSchema(
      op_type, kOnnxDomain, since_version, doc,
      {
          {"data", "An input tensor.", "T"},
          {"axes",
           "Optional input list of integers, along which to reduce. "
           "The default is to reduce over all the dimensions of the input tensor if "
           "noop_with_empty_axes is false, else act as an Identity op when "
           "noop_with_empty_axes is true. Accepted range is [-r, r-1] where r = rank(data).",
           "tensor(int64)"},
      },
      {
          {"reduced", "Reduced output tensor.", "T"},
      },
      {
          {"T", allowed_types, type_constraint_desc},
      },
      {
          MakeKeepdimsAttr(),
          MakeNoopWithEmptyAxesAttr(),
      });
}

// Builds an ArgMax/ArgMin schema entry. All versions share the same I/O shape
// (single input "data", single int64 output "reduced"); only docs differ.
LightOpSchema MakeArgReduceSchema(const std::string &op_type, int since_version,
                                  const std::string &doc,
                                  const std::vector<TensorType> &allowed_types) {
  return LightOpSchema(
      op_type, kOnnxDomain, since_version, doc,
      {
          {"data", "An input tensor.", "T"},
      },
      {
          {"reduced", "Reduced output tensor with integer data type.", "tensor(int64)"},
      },
      {
          {"T", allowed_types, "Constrain input and output types to all numeric tensors."},
      });
}

constexpr const char *kHighPrecisionDesc =
    "Constrain input and output types to high-precision numeric tensors.";
constexpr const char *kHighPrecisionPlus8BitDesc =
    "Constrain input and output types to high-precision and 8 bit numeric tensors.";
constexpr const char *kHighPrecisionPlus8BitAndBoolDesc =
    "Constrain input and output types to high-precision, 8 bit numeric tensors and Boolean.";

} // namespace

std::vector<LightOpSchema> BuildReduceSumSchemas() {
  return std::vector<LightOpSchema>{
      // ReduceSum opset 13: axes moved from attribute to optional second input; added bfloat16.
      MakeReduceAxesInputSchema("ReduceSum", 13, MakeReduceSumDoc(13),
                                NumericTypesForMathReductionIr4(), kHighPrecisionDesc),
      // ReduceSum opset 11: axes remain an attribute; adds support for negative axes.
      MakeReduceAttrSchema("ReduceSum", 11, MakeReduceSumDoc(11), NumericTypesForMathReduction(),
                           kHighPrecisionDesc),
      // ReduceSum opset 1: axes are provided as an attribute.
      MakeReduceAttrSchema("ReduceSum", 1, MakeReduceSumDoc(1), NumericTypesForMathReduction(),
                           kHighPrecisionDesc),
  };
}

// Builds the version history for simple reduce ops that share the pattern
// {v18 with axes input, v13/v11/v1 with axes attribute}.
// Used by ReduceMean, ReduceProd, ReduceSumSquare, ReduceLogSum, ReduceLogSumExp,
// ReduceL1 and ReduceL2.
std::vector<LightOpSchema> BuildSimpleReduceSchemas(const std::string &op_type,
                                                    const std::string &name,
                                                    const std::string &empty_value) {
  return std::vector<LightOpSchema>{
      // opset 18: axes moved to optional second input; noop_with_empty_axes attribute added.
      MakeReduceAxesInputSchema(op_type, 18, MakeReduceOpDoc(name, empty_value, 18),
                                NumericTypesForMathReductionIr4(), kHighPrecisionDesc),
      // opset 13: axes remain an attribute; bfloat16 added.
      MakeReduceAttrSchema(op_type, 13, MakeReduceOpDoc(name, empty_value, 13),
                           NumericTypesForMathReductionIr4(), kHighPrecisionDesc),
      // opset 11: axes attribute supports negative values.
      MakeReduceAttrSchema(op_type, 11, MakeReduceOpDoc(name, empty_value, 11),
                           NumericTypesForMathReduction(), kHighPrecisionDesc),
      // opset 1: axes attribute, no negative axis support.
      MakeReduceAttrSchema(op_type, 1, MakeReduceOpDoc(name, empty_value, 1),
                           NumericTypesForMathReduction(), kHighPrecisionDesc),
  };
}

// Builds the version history for ReduceMax / ReduceMin, which additionally
// support 8-bit numeric tensors from opset 12 and boolean tensors from opset 20.
std::vector<LightOpSchema> BuildReduceMaxMinSchemas(const std::string &op_type,
                                                    const std::string &name,
                                                    const std::string &empty_value) {
  return std::vector<LightOpSchema>{
      // opset 20: adds support for boolean tensors.
      MakeReduceAxesInputSchema(op_type, 20, MakeReduceOpDoc(name, empty_value, 20),
                                NumericTypesForMaxMinIr4WithInt8AndBool(),
                                kHighPrecisionPlus8BitAndBoolDesc),
      // opset 18: axes moved to optional second input; noop_with_empty_axes attribute added.
      MakeReduceAxesInputSchema(op_type, 18, MakeReduceOpDoc(name, empty_value, 18),
                                NumericTypesForMaxMinIr4WithInt8(), kHighPrecisionPlus8BitDesc),
      // opset 13: axes remain an attribute; bfloat16 added.
      MakeReduceAttrSchema(op_type, 13, MakeReduceOpDoc(name, empty_value, 13),
                           NumericTypesForMaxMinIr4WithInt8(), kHighPrecisionPlus8BitDesc),
      // opset 12: 8-bit numeric tensors added.
      MakeReduceAttrSchema(op_type, 12, MakeReduceOpDoc(name, empty_value, 12),
                           NumericTypesForMaxMinWithInt8(), kHighPrecisionPlus8BitDesc),
      // opset 11: axes attribute supports negative values.
      MakeReduceAttrSchema(op_type, 11, MakeReduceOpDoc(name, empty_value, 11),
                           NumericTypesForMathReduction(), kHighPrecisionDesc),
      // opset 1: axes attribute, no negative axis support.
      MakeReduceAttrSchema(op_type, 1, MakeReduceOpDoc(name, empty_value, 1),
                           NumericTypesForMathReduction(), kHighPrecisionDesc),
  };
}

// Builds the version history for ArgMax / ArgMin.
std::vector<LightOpSchema> BuildArgReduceSchemas(const std::string &op_type,
                                                 const std::string &name) {
  return std::vector<LightOpSchema>{
      // opset 13: bfloat16 added to type constraint.
      MakeArgReduceSchema(op_type, 13, MakeArgReduceDoc(name, 13), AllNumericTypesIr4()),
      // opset 12: select_last_index attribute introduced; doc requires non-empty input.
      MakeArgReduceSchema(op_type, 12, MakeArgReduceDoc(name, 12), AllNumericTypes()),
      // opset 11: axis attribute supports negative values; input must not be empty.
      MakeArgReduceSchema(op_type, 11, MakeArgReduceDoc(name, 11), AllNumericTypes()),
      // opset 1: axis attribute, no negative axis support.
      MakeArgReduceSchema(op_type, 1, MakeArgReduceDoc(name, 1), AllNumericTypes()),
  };
}

std::vector<LightOpSchema> GetAllOnnxOpReductionSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas;

  auto append = [&schemas](std::vector<LightOpSchema> &&v) {
    for (auto &s : v) {
      schemas.push_back(std::move(s));
    }
  };

  append(BuildArgReduceSchemas("ArgMax", "max"));
  append(BuildArgReduceSchemas("ArgMin", "min"));
  append(BuildSimpleReduceSchemas("ReduceL1", "L1 norm", kEmptyZero));
  append(BuildSimpleReduceSchemas("ReduceL2", "L2 norm", kEmptyZero));
  append(BuildSimpleReduceSchemas("ReduceLogSum", "log sum", kEmptyMinusInf));
  append(BuildSimpleReduceSchemas("ReduceLogSumExp", "log sum exponent", kEmptyMinusInf));
  append(BuildReduceMaxMinSchemas("ReduceMax", "max", kEmptyMin));
  append(BuildSimpleReduceSchemas("ReduceMean", "mean", kEmptyUndefined));
  append(BuildReduceMaxMinSchemas("ReduceMin", "min", kEmptyMax));
  append(BuildSimpleReduceSchemas("ReduceProd", "product", kEmptyOne));
  append(BuildReduceSumSchemas());
  append(BuildSimpleReduceSchemas("ReduceSumSquare", "sum square", kEmptyZero));

  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace reduction
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

constexpr const char *kOnnxDomain = "ai.onnx";

struct FormalParameter {
  std::string name;
  std::string description;
  std::string type;
};

enum class TensorType : uint8_t {
  kBool,
  kString,
  kUint8,
  kUint16,
  kUint32,
  kUint64,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kFloat16,
  kFloat,
  kDouble,
  kBfloat16,
  kFloat8e4m3fn,
  kFloat8e4m3fnuz,
  kFloat8e5m2,
  kFloat8e5m2fnuz,
  kFloat8e8m0,
  kFloat4e2m1,
  kUint4,
  kInt4,
  kUint2,
  kInt2,
  kComplex64,
  kComplex128,
  kSeqBool,
  kSeqString,
  kSeqUint8,
  kSeqUint16,
  kSeqUint32,
  kSeqUint64,
  kSeqInt8,
  kSeqInt16,
  kSeqInt32,
  kSeqInt64,
  kSeqFloat16,
  kSeqFloat,
  kSeqDouble,
  kSeqComplex64,
  kSeqComplex128,
  kSeqMapStringFloat,
  kSeqMapInt64Float,
};

struct TypeConstraintParam {
  std::string type_param_str;
  std::vector<TensorType> allowed_type_strs;
  std::string description;
};

const char *ToTypeString(TensorType type);

class SchemaError final : public std::runtime_error {
public:
  explicit SchemaError(const std::string &message) : std::runtime_error(message) {}
};

class LightOpSchema {
public:
  LightOpSchema(std::string name, std::string domain, int since_version, std::string doc,
                std::vector<FormalParameter> inputs, std::vector<FormalParameter> outputs,
                std::vector<TypeConstraintParam> type_constraints,
                bool has_function_implementation = false)
      : name_(std::move(name)), domain_(std::move(domain)), since_version_(since_version),
        doc_(std::move(doc)), inputs_(std::move(inputs)), outputs_(std::move(outputs)),
        type_constraints_(std::move(type_constraints)),
        has_function_implementation_(has_function_implementation) {}

  const std::string &name() const { return name_; }
  const std::string &domain() const { return domain_; }
  int since_version() const { return since_version_; }
  const std::string &doc() const { return doc_; }
  const std::vector<FormalParameter> &inputs() const { return inputs_; }
  const std::vector<FormalParameter> &outputs() const { return outputs_; }
  const std::vector<TypeConstraintParam> &type_constraints() const { return type_constraints_; }
  bool has_function_implementation() const { return has_function_implementation_; }

private:
  std::string name_;
  std::string domain_;
  int since_version_;
  std::string doc_;
  std::vector<FormalParameter> inputs_;
  std::vector<FormalParameter> outputs_;
  std::vector<TypeConstraintParam> type_constraints_;
  bool has_function_implementation_;
};

std::vector<TensorType> FloatTypes();
std::vector<TensorType> NumericTypesForMathReduction();
std::vector<TensorType> NumericTypesForMathReductionIr4();
std::vector<TensorType> AllNumericTypes();
std::vector<TensorType> AllNumericTypesIr4();
std::vector<TensorType> CastTypesVer1And6();
std::vector<TensorType> CastTypesVer9();
std::vector<TensorType> CastTypesVer13();
std::vector<TensorType> CastTypesVer19();
std::vector<TensorType> CastTypesVer21();
std::vector<TensorType> CastTypesVer23();
std::vector<TensorType> CastTypesVer24();
std::vector<TensorType> CastTypesVer25();
std::vector<TensorType> EqualTypesV1V7();
std::vector<TensorType> EqualTypesV11();
std::vector<TensorType> EqualTypesV13();
std::vector<TensorType> EqualTypesV19();

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

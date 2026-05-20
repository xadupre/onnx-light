// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ostream>
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
};

const char *ToTypeString(TensorType type);

struct TypeConstraintParam {
  std::string type_param_str;
  struct AllowedType {
    std::string type_str;

    AllowedType(TensorType type) : type_str(ToTypeString(type)) {}
    AllowedType(std::string value) : type_str(std::move(value)) {}
    AllowedType(const char *value) : type_str(value) {}

    bool operator==(const AllowedType &other) const = default;
    bool operator==(TensorType type) const { return type_str == ToTypeString(type); }
    friend bool operator==(TensorType type, const AllowedType &allowed_type) {
      return allowed_type == type;
    }
    friend void PrintTo(const AllowedType &allowed_type, std::ostream *os) {
      *os << allowed_type.type_str;
    }
  };

  TypeConstraintParam() = default;
  TypeConstraintParam(std::string type_param_str, std::vector<AllowedType> allowed_type_strs,
                      std::string description)
      : type_param_str(std::move(type_param_str)), allowed_type_strs(std::move(allowed_type_strs)),
        description(std::move(description)) {}
  TypeConstraintParam(std::string type_param_str,
                      std::initializer_list<AllowedType> allowed_type_strs, std::string description)
      : type_param_str(std::move(type_param_str)), allowed_type_strs(allowed_type_strs),
        description(std::move(description)) {}

  std::vector<AllowedType> allowed_type_strs;
  std::string description;
};

const std::string &ToTypeString(const TypeConstraintParam::AllowedType &type);

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
  LightOpSchema(std::string name, std::string domain, int since_version, std::string doc,
                std::initializer_list<FormalParameter> inputs,
                std::initializer_list<FormalParameter> outputs,
                std::initializer_list<TypeConstraintParam> type_constraints,
                bool has_function_implementation = false)
      : name_(std::move(name)), domain_(std::move(domain)), since_version_(since_version),
        doc_(std::move(doc)), inputs_(inputs), outputs_(outputs),
        type_constraints_(type_constraints),
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

std::vector<TypeConstraintParam::AllowedType> FloatTypes();
std::vector<TypeConstraintParam::AllowedType> NumericTypesForMathReduction();
std::vector<TypeConstraintParam::AllowedType> NumericTypesForMathReductionIr4();
std::vector<TypeConstraintParam::AllowedType> AllNumericTypes();
std::vector<TypeConstraintParam::AllowedType> AllNumericTypesIr4();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer1And6();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer9();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer13();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer19();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer21();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer23();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer24();
std::vector<TypeConstraintParam::AllowedType> CastTypesVer25();
std::vector<TypeConstraintParam::AllowedType> EqualTypesV1V7();
std::vector<TypeConstraintParam::AllowedType> EqualTypesV11();
std::vector<TypeConstraintParam::AllowedType> EqualTypesV13();
std::vector<TypeConstraintParam::AllowedType> EqualTypesV19();

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

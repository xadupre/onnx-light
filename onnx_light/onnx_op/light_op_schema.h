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
  kComplex64,
  kComplex128,
};

struct TypeConstraintParam {
  std::string type_param_str;
  std::vector<TensorType> allowed_type_strs;
  std::string description;
};

inline const char *ToTypeString(TensorType type) {
  switch (type) {
  case TensorType::kBool:
    return "tensor(bool)";
  case TensorType::kUint8:
    return "tensor(uint8)";
  case TensorType::kUint16:
    return "tensor(uint16)";
  case TensorType::kUint32:
    return "tensor(uint32)";
  case TensorType::kUint64:
    return "tensor(uint64)";
  case TensorType::kInt8:
    return "tensor(int8)";
  case TensorType::kInt16:
    return "tensor(int16)";
  case TensorType::kInt32:
    return "tensor(int32)";
  case TensorType::kInt64:
    return "tensor(int64)";
  case TensorType::kFloat16:
    return "tensor(float16)";
  case TensorType::kFloat:
    return "tensor(float)";
  case TensorType::kDouble:
    return "tensor(double)";
  case TensorType::kBfloat16:
    return "tensor(bfloat16)";
  case TensorType::kComplex64:
    return "tensor(complex64)";
  case TensorType::kComplex128:
    return "tensor(complex128)";
  }
  throw std::logic_error("Unknown TensorType.");
}

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

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

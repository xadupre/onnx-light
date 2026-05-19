// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

struct FormalParameter {
  std::string name;
  std::string description;
  std::string type;
};

struct TypeConstraintParam {
  std::string type_param_str;
  std::vector<std::string> allowed_type_strs;
  std::string description;
};

class SchemaError final : public std::runtime_error {
public:
  explicit SchemaError(const std::string &message) : std::runtime_error(message) {}
};

class MathOpSchema {
public:
  MathOpSchema(std::string name, std::string domain, int since_version, std::string doc,
               std::vector<FormalParameter> inputs, std::vector<FormalParameter> outputs,
               std::vector<TypeConstraintParam> type_constraints)
      : name_(std::move(name)), domain_(std::move(domain)), since_version_(since_version),
        doc_(std::move(doc)), inputs_(std::move(inputs)), outputs_(std::move(outputs)),
        type_constraints_(std::move(type_constraints)) {}

  const std::string &name() const { return name_; }
  const std::string &domain() const { return domain_; }
  int since_version() const { return since_version_; }
  const std::string &doc() const { return doc_; }
  const std::vector<FormalParameter> &inputs() const { return inputs_; }
  const std::vector<FormalParameter> &outputs() const { return outputs_; }
  const std::vector<TypeConstraintParam> &type_constraints() const { return type_constraints_; }
  bool has_type_and_shape_inference_function() const { return false; }
  bool has_data_propagation_function() const { return false; }

private:
  std::string name_;
  std::string domain_;
  int since_version_;
  std::string doc_;
  std::vector<FormalParameter> inputs_;
  std::vector<FormalParameter> outputs_;
  std::vector<TypeConstraintParam> type_constraints_;
};

const std::string &OnnxOpMathDomain();

void RegisterOnnxOpMathOperatorSetSchema(int target_version = 0, bool fail_duplicate_schema = true);

void DeregisterOnnxOpMathOperatorSetSchema();

const MathOpSchema *
GetOnnxOpMathSchema(const std::string &op_type,
                    int max_inclusive_version = std::numeric_limits<int>::max());

std::vector<MathOpSchema> GetAllOnnxOpMathSchemasWithHistory();

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

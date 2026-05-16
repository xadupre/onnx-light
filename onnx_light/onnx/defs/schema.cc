// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/schema.h"
#include "onnx/checker.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets_ml.h"
#include "onnx/defs/operator_sets_preview.h"
#include "onnx/defs/operator_sets_training.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE {

int OpSchemaRegistry::loaded_schema_version = -1;

void RegisterSchema(const OpSchema &schema, int opset_version_to_load, bool fail_duplicate_schema,
                    bool fail_with_exception) {
  RegisterSchema(OpSchema(schema), opset_version_to_load, fail_duplicate_schema,
                 fail_with_exception);
}

void RegisterSchema(OpSchema &&schema, int opset_version_to_load, bool fail_duplicate_schema,
                    bool fail_with_exception) {
  if (fail_with_exception) {
    OpSchemaRegistry::OpSchemaRegisterOnce::OpSchemaRegisterImpl(
        std::move(schema), opset_version_to_load, fail_duplicate_schema);
  } else {
    OpSchemaRegistry::OpSchemaRegisterOnce::OpSchemaRegisterNoExcept(
        std::move(schema), opset_version_to_load, fail_duplicate_schema);
  }
}

void DeregisterSchema(const std::string &op_type, int version, const std::string &domain) {
  auto &registry = OpSchemaRegistry::map();
  auto it_name = registry.find(op_type);
  if (it_name == registry.end()) {
    return;
  }
  auto it_domain = it_name->second.find(domain);
  if (it_domain == it_name->second.end()) {
    return;
  }
  it_domain->second.erase(version);
  if (it_domain->second.empty()) {
    it_name->second.erase(it_domain);
  }
  if (it_name->second.empty()) {
    registry.erase(it_name);
  }
}

const std::string &OpSchema::FormalParameter::GetName() const { return name_; }

const DataTypeSet &OpSchema::FormalParameter::GetTypes() const { return type_set_; }

DataTypeSet &OpSchema::FormalParameter::MutableTypes() { return type_set_; }

const std::string &OpSchema::FormalParameter::GetTypeStr() const { return type_str_; }

const std::string &OpSchema::FormalParameter::GetDescription() const { return description_; }

OpSchema::FormalParameterOption OpSchema::FormalParameter::GetOption() const {
  return param_option_;
}

bool OpSchema::FormalParameter::GetIsHomogeneous() const { return is_homogeneous_; }

int OpSchema::FormalParameter::GetMinArity() const { return min_arity_; }

OpSchema::DifferentiationCategory OpSchema::FormalParameter::GetDifferentiationCategory() const {
  return differentiation_category_;
}

OpSchemaRegistry *OpSchemaRegistry::Instance() {
  static OpSchemaRegistry instance;
  return &instance;
}

OpSchema &OpSchema::SinceVersion(OperatorSetVersion v) {
  since_version_ = v;
  return *this;
}

OpSchema &OpSchema::Deprecate() {
  deprecated_ = true;
  return *this;
}

OpSchema &OpSchema::NumInputs(std::unordered_set<int> allowed_input_nums) {
  num_inputs_allowed_ = [allowed_input_nums = std::move(allowed_input_nums)](int n) -> bool {
    return allowed_input_nums.count(n) > 0;
  };
  return *this;
}

OpSchema &OpSchema::NumOutputs(std::unordered_set<int> allowed_output_nums) {
  num_outputs_allowed_ = [allowed_output_nums = std::move(allowed_output_nums)](int n) -> bool {
    return allowed_output_nums.count(n) > 0;
  };
  return *this;
}

OpSchema &OpSchema::TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction) {
  tensor_inference_function_ = std::move(inferenceFunction);
  return *this;
}

OpSchema &
OpSchema::PartialDataPropagationFunction(DataPropagationFunction dataPropagationFunction) {
  data_propagation_function_ = std::move(dataPropagationFunction);
  return *this;
}

OpSchema &OpSchema::SetSupportLevel(SupportType support) {
  support_ = support;
  return *this;
}

OpSchema &OpSchema::SetName(std::string name) {
  name_ = std::move(name);
  return *this;
}

OpSchema &OpSchema::SetName(const char *name) { return SetName(std::string(name)); }

OpSchema &OpSchema::SetLocation(std::string file, int line) {
  file_ = std::move(file);
  line_ = line;
  return *this;
}

OpSchema &OpSchema::SetLocation(const char *file, int line) {
  return SetLocation(std::string(file), line);
}

OpSchema &OpSchema::SetDomain(std::string domain) {
  domain_ = std::move(domain);
  return *this;
}

OpSchema &OpSchema::SetDomain(const char *domain) { return SetDomain(std::string(domain)); }

OpSchema &OpSchema::SetDoc(const char *doc) { return SetDoc(std::string(doc)); }

OpSchema &OpSchema::SetDoc(const std::string &doc) {
  doc_ = doc;
  return *this;
}

OpSchema &OpSchema::Attr(Attribute attr) {
  auto name = attr.name;
  attributes_.emplace(std::move(name), std::move(attr));
  return *this;
}

OpSchema &OpSchema::Attr(std::string name, std::string description,
                         AttributeProto::AttributeType type, bool required) {
  Attr(Attribute{std::move(name), std::move(description), type, required});
  return *this;
}

OpSchema &OpSchema::Attr(const char *name, const char *description,
                         AttributeProto::AttributeType type, bool required) {
  return Attr(std::string(name), std::string(description), type, required);
}

#define ATTR_SETTER_WITH_SINGLE_VALUE(type, field, attrtype)                                       \
  OpSchema &OpSchema::Attr(std::string name, std::string description,                              \
                           AttributeProto::AttributeType attr_type, const type &default_value) {   \
    if (attrtype != attr_type) {                                                                   \
      fail_schema("Attribute specification type mismatch.");                                       \
    }                                                                                              \
    AttributeProto a;                                                                              \
    a.set_name(name);                                                                              \
    a.set_##field(default_value);                                                                  \
    a.set_type(attr_type);                                                                         \
    Attr(Attribute(std::move(name), std::move(description), std::move(a)));                        \
    return *this;                                                                                  \
  }

#define ATTR_SETTER_WITH_LIST_VALUE(type, field, attrtype)                                         \
  OpSchema &OpSchema::Attr(std::string name, std::string description,                              \
                           AttributeProto::AttributeType attr_type,                                \
                           const std::vector<type> &default_value) {                               \
    if (attrtype != attr_type) {                                                                   \
      fail_schema("Attribute specification type mismatch.");                                       \
    }                                                                                              \
    AttributeProto a;                                                                              \
    a.set_name(name);                                                                              \
    a.set_type(attr_type);                                                                         \
    for (const auto &v : default_value) {                                                          \
      a.ref_##field().emplace_back(type(v));                                                       \
    }                                                                                              \
    Attr(Attribute(std::move(name), std::move(description), std::move(a)));                        \
    return *this;                                                                                  \
  }

#define ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(type, field, attrtype)                                \
  OpSchema &OpSchema::Attr(std::string name, std::string description,                              \
                           AttributeProto::AttributeType attr_type, const type &default_value) {   \
    if (attrtype != attr_type) {                                                                   \
      fail_schema("Attribute specification type mismatch.");                                       \
    }                                                                                              \
    AttributeProto a;                                                                              \
    a.set_name(name);                                                                              \
    a.ref_##field().CopyFrom(default_value);                                                       \
    a.set_type(attr_type);                                                                         \
    Attr(Attribute(std::move(name), std::move(description), std::move(a)));                        \
    return *this;                                                                                  \
  }

#define ATTR_SETTER_WITH_LIST_COMPLEXVALUE(type, field, attrtype)                                  \
  OpSchema &OpSchema::Attr(std::string name, std::string description,                              \
                           AttributeProto::AttributeType attr_type,                                \
                           const std::vector<type> &default_value) {                               \
    if (attrtype != attr_type) {                                                                   \
      fail_schema("Attribute specification type mismatch.");                                       \
    }                                                                                              \
    AttributeProto a;                                                                              \
    a.set_name(name);                                                                              \
    a.set_type(attr_type);                                                                         \
    for (const auto &v : default_value) {                                                          \
      a.add_##field()->CopyFrom(v);                                                                \
    }                                                                                              \
    Attr(Attribute(std::move(name), std::move(description), std::move(a)));                        \
    return *this;                                                                                  \
  }

ATTR_SETTER_WITH_SINGLE_VALUE(int64_t, i, AttributeProto::INT)
ATTR_SETTER_WITH_SINGLE_VALUE(float, f, AttributeProto::FLOAT)
ATTR_SETTER_WITH_SINGLE_VALUE(std::string, s, AttributeProto::STRING)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(TensorProto, t, AttributeProto::TENSOR)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(GraphProto, g, AttributeProto::GRAPH)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(TypeProto, tp, AttributeProto::TYPE_PROTO)
ATTR_SETTER_WITH_LIST_VALUE(int64_t, ints, AttributeProto::INTS)
ATTR_SETTER_WITH_LIST_VALUE(float, floats, AttributeProto::FLOATS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(TensorProto, tensors, AttributeProto::TENSORS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(GraphProto, graphs, AttributeProto::GRAPHS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(TypeProto, type_protos, AttributeProto::TYPE_PROTOS)

OpSchema &OpSchema::Attr(std::string name, std::string description,
                         AttributeProto::AttributeType attr_type,
                         const std::vector<std::string> &default_value) {
  if (AttributeProto::STRINGS != attr_type) {
    fail_schema("Attribute specification type mismatch.");
  }
  AttributeProto a;
  a.set_name(name);
  a.set_type(attr_type);
  for (const auto &v : default_value) {
    a.add_strings(utils::String(v));
  }
  Attr(Attribute(std::move(name), std::move(description), std::move(a)));
  return *this;
}

OpSchema &OpSchema::AllowUncheckedAttributes() {
  allows_unchecked_attributes_ = true;
  return *this;
}

OpSchema &OpSchema::Input(int n, FormalParameter formal_parameter) {
  if (inputs_.size() <= static_cast<size_t>(n)) {
    inputs_.resize(n + 1);
  }
  inputs_[n] = std::move(formal_parameter);
  return *this;
}

OpSchema &OpSchema::Input(int n, std::string name, const std::string &description,
                          std::string type_str, OpSchema::FormalParameterOption param_option,
                          bool is_homogeneous, int min_arity,
                          DifferentiationCategory differentiation_category) {
  return Input(n, FormalParameter(std::move(name), description, std::move(type_str), param_option,
                                  is_homogeneous, min_arity, differentiation_category));
}

OpSchema &OpSchema::Output(int n, FormalParameter formal_parameter) {
  if (outputs_.size() <= static_cast<size_t>(n)) {
    outputs_.resize(n + 1);
  }
  outputs_[n] = std::move(formal_parameter);
  return *this;
}

OpSchema &OpSchema::Output(int n, std::string name, const std::string &description,
                           std::string type_str, OpSchema::FormalParameterOption param_option,
                           bool is_homogeneous, int min_arity,
                           DifferentiationCategory differentiation_category) {
  return Output(n, FormalParameter(std::move(name), description, std::move(type_str), param_option,
                                   is_homogeneous, min_arity, differentiation_category));
}

OpSchema &OpSchema::TypeConstraint(std::string type_str, std::vector<std::string> constraints,
                                   std::string description) {
  if (type_constraints_.find(type_str) != type_constraints_.end()) {
    fail_schema("Duplicate type constraint name");
  }

  DataTypeSet d;
  for (const auto &t : constraints) {
    d.insert(Utils::DataTypeUtils::ToType(t));
  }
  type_constraints_.emplace(type_str, std::make_pair(d, description));
  type_constraint_params_.emplace_back(std::move(type_str), std::move(constraints),
                                       std::move(description));
  return *this;
}

OpSchema &OpSchema::TypeConstraint(const char *type_str,
                                   std::initializer_list<const char *> constraints,
                                   const char *description) {
  std::vector<std::string> constraints_vector;
  constraints_vector.reserve(constraints.size());
  for (const auto *const constraint : constraints) {
    constraints_vector.emplace_back(constraint);
  }

  return TypeConstraint(std::string(type_str), constraints_vector, std::string(description));
}

void OpSchema::ParseAndSetTypes(std::vector<OpSchema::FormalParameter> *formal_parameters) {
  for (auto &formal_parameter : *formal_parameters) {
    const auto &type = formal_parameter.GetTypeStr();
    DataTypeSet allowed_types;
    if (auto it = type_constraints_.find(type); it != type_constraints_.end()) {
      allowed_types = it->second.first;
    } else {
      allowed_types.emplace(Utils::DataTypeUtils::ToType(type));
    }

    formal_parameter.MutableTypes() = allowed_types;
  }
}

OpSchema &OpSchema::SetContextDependentFunctionBodyBuilder(
    ContextDependentFunctionBodyBuilder functionBuilder, int opset_version) {
  if (opset_version == OpSchema::kUninitializedSinceVersion &&
      since_version_ != OpSchema::kUninitializedSinceVersion) {
    opset_version_to_function_builder_[since_version_] = std::move(functionBuilder);
  } else {
    opset_version_to_function_builder_[opset_version] = std::move(functionBuilder);
  }
  return *this;
}

OpSchema &OpSchema::FunctionBody(const std::vector<NodeProto> &func_nodes, int opset_version) {
  if (opset_version == OpSchema::kUninitializedSinceVersion &&
      since_version_ != OpSchema::kUninitializedSinceVersion) {
    opset_version = since_version_;
  }
  auto function_proto = std::make_shared<FunctionProto>();
  for (const auto &node : func_nodes) {
    function_proto->add_node()->CopyFrom(node);
  }
  if (function_proto->ref_opset_import().empty()) {
    auto *schema_opset = function_proto->add_opset_import();
    schema_opset->set_domain(domain_);
    schema_opset->set_version(opset_version);
  }
  opset_version_to_function_body_.emplace(opset_version, std::move(function_proto));
  return *this;
}

const FunctionProto *OpSchema::GetFunction(int requested_opset_version, bool /*validate*/) const {
  if (opset_version_to_function_body_.empty()) {
    return nullptr;
  }
  if (requested_opset_version == OpSchema::kUninitializedSinceVersion) {
    return opset_version_to_function_body_.rbegin()->second.get();
  }
  auto it = opset_version_to_function_body_.upper_bound(requested_opset_version);
  if (it != opset_version_to_function_body_.begin()) {
    --it;
    return it->second.get();
  }
  return nullptr;
}

void OpSchema::BuildContextDependentFunction(const FunctionBodyBuildContext &ctx,
                                             FunctionProto &function_proto,
                                             int opset_version) const {
  if (opset_version == kUninitializedSinceVersion) {
    if (!opset_version_to_function_builder_.empty()) {
      opset_version = opset_version_to_function_builder_.rbegin()->first;
    }
  }
  auto it = opset_version_to_function_builder_.find(opset_version);
  if (it == opset_version_to_function_builder_.end()) {
    return;
  }
  it->second(ctx, *this, function_proto);
}

std::vector<int> OpSchema::function_opset_versions() const {
  std::vector<int> versions;
  versions.reserve(opset_version_to_function_body_.size());
  for (const auto &[ver, _] : opset_version_to_function_body_) {
    versions.push_back(ver);
  }
  return versions;
}

std::vector<int> OpSchema::context_dependent_function_opset_versions() const {
  std::vector<int> versions;
  versions.reserve(opset_version_to_function_builder_.size());
  for (const auto &[ver, _] : opset_version_to_function_builder_) {
    versions.push_back(ver);
  }
  return versions;
}

std::string OpSchema::VerifyFailPrefix(std::string_view node_name) const {
  std::string str = "Node";
  if (!node_name.empty()) {
    str = str + "(" + std::string(node_name) + ")";
  }
  str = str + " with schema(" + domain() + "::" + Name() + ":" + std::to_string(since_version()) +
        ")";
  return str;
}

void OpSchema::VerifyInputNum(int input_num, std::string_view node_name) const {
  if (input_num < min_input_ || input_num > max_input_) {
    fail_schema(VerifyFailPrefix(node_name), " has input size ", input_num,
                " not in range [min=", min_input_, ", max=", max_input_, "].");
  }
  if (!num_inputs_allowed_(input_num)) {
    fail_schema(VerifyFailPrefix(node_name), " has input size ", input_num,
                " not in allowed input sizes.");
  }
}

void OpSchema::VerifyOutputNum(int output_num, std::string_view node_name) const {
  if (output_num < min_output_ || output_num > max_output_) {
    fail_schema(VerifyFailPrefix(node_name), " has output size ", output_num,
                " not in range [min=", min_output_, ", max=", max_output_, "].");
  }
  if (!num_outputs_allowed_(output_num)) {
    fail_schema(VerifyFailPrefix(node_name), " has output size ", output_num,
                " not in allowed output sizes.");
  }
}

void OpSchema::CheckInputOutputType(struct InferenceContext &ctx) const {
  VerifyInputNum(static_cast<int>(ctx.getNumInputs()));
  VerifyOutputNum(static_cast<int>(ctx.getNumOutputs()));
}

void OpSchema::Verify(const NodeProto &node) const {
  VerifyInputNum(static_cast<int>(node.ref_input().size()), node.ref_name().as_string());
  VerifyOutputNum(static_cast<int>(node.ref_output().size()), node.ref_name().as_string());
}

OpSchema &OpSchema::FillUsing(const std::function<void(OpSchema &)> &populator) {
  if (populator) {
    populator(*this);
  }
  return *this;
}

void OpSchema::BuildFunction(FunctionProto &function_body) const {
  function_body.set_name(name_);
  function_body.set_doc_string(doc_);
  function_body.set_domain(domain_);
  for (const auto &i : inputs_) {
    *function_body.add_input() = i.GetName();
  }
  for (const auto &o : outputs_) {
    *function_body.add_output() = o.GetName();
  }
  for (const auto &a : attributes_) {
    *function_body.add_attribute() = a.first;
  }
  if (function_body.ref_opset_import().empty()) {
    auto *schema_opset = function_body.add_opset_import();
    schema_opset->set_domain(domain_);
    schema_opset->set_version(since_version_);
  }
}

void OpSchema::Finalize() {
  ParseAndSetTypes(&inputs_);
  ParseAndSetTypes(&outputs_);

  min_input_ = 0;
  max_input_ = 0;
  for (const auto &input : inputs_) {
    if (input.GetOption() == Variadic) {
      min_input_ += input.GetMinArity();
      max_input_ = std::numeric_limits<int>::max();
    } else if (input.GetOption() == Optional) {
      max_input_ += 1;
    } else {
      min_input_ += 1;
      max_input_ += 1;
    }
  }

  min_output_ = 0;
  max_output_ = 0;
  for (const auto &output : outputs_) {
    if (output.GetOption() == Variadic) {
      min_output_ += output.GetMinArity();
      max_output_ = std::numeric_limits<int>::max();
    } else if (output.GetOption() == Optional) {
      max_output_ += 1;
    } else {
      min_output_ += 1;
      max_output_ += 1;
    }
  }

  for (auto &kv : opset_version_to_function_body_) {
    BuildFunction(*kv.second);
  }
}

const std::vector<std::string> &OpSchema::all_numeric_types() {
  static const std::vector<std::string> numeric_types = {
      "tensor(uint8)",   "tensor(uint16)", "tensor(uint32)",     "tensor(uint64)",
      "tensor(int8)",    "tensor(int16)",  "tensor(int32)",      "tensor(int64)",
      "tensor(float16)", "tensor(float)",  "tensor(double)",     "tensor(bfloat16)",
      "tensor(uint4)",   "tensor(int4)",   "tensor(float4e2m1)", "tensor(float8e8m0)",
  };
  return numeric_types;
}

const std::vector<std::string> &OpSchema::all_tensor_types() {
  static const std::vector<std::string> tensor_types = {
      "tensor(float)",      "tensor(uint8)",          "tensor(int8)",
      "tensor(uint16)",     "tensor(int16)",          "tensor(int32)",
      "tensor(int64)",      "tensor(string)",         "tensor(bool)",
      "tensor(float16)",    "tensor(double)",         "tensor(uint32)",
      "tensor(uint64)",     "tensor(complex64)",      "tensor(complex128)",
      "tensor(bfloat16)",   "tensor(float8e4m3fn)",   "tensor(float8e4m3fnuz)",
      "tensor(float8e5m2)", "tensor(float8e5m2fnuz)", "tensor(uint4)",
      "tensor(int4)",       "tensor(float4e2m1)",     "tensor(float8e8m0)",
  };
  return tensor_types;
}

// ---------------------------------------------------------------------------
// IR-version-specific type lists (adapted from onnx/defs/schema.cc)
// ---------------------------------------------------------------------------

const std::vector<std::string> &OpSchema::all_tensor_types_ir4() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",    "tensor(uint16)",  "tensor(uint32)",    "tensor(uint64)",
      "tensor(int8)",     "tensor(int16)",   "tensor(int32)",     "tensor(int64)",
      "tensor(bfloat16)", "tensor(float16)", "tensor(float)",     "tensor(double)",
      "tensor(string)",   "tensor(bool)",    "tensor(complex64)", "tensor(complex128)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_types_ir9() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",      "tensor(uint16)",        "tensor(uint32)",
      "tensor(uint64)",     "tensor(int8)",          "tensor(int16)",
      "tensor(int32)",      "tensor(int64)",         "tensor(bfloat16)",
      "tensor(float16)",    "tensor(float)",         "tensor(double)",
      "tensor(string)",     "tensor(bool)",          "tensor(complex64)",
      "tensor(complex128)", "tensor(float8e4m3fn)",  "tensor(float8e4m3fnuz)",
      "tensor(float8e5m2)", "tensor(float8e5m2fnuz)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_types_ir10() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",      "tensor(uint16)",         "tensor(uint32)",
      "tensor(uint64)",     "tensor(int8)",           "tensor(int16)",
      "tensor(int32)",      "tensor(int64)",          "tensor(bfloat16)",
      "tensor(float16)",    "tensor(float)",          "tensor(double)",
      "tensor(string)",     "tensor(bool)",           "tensor(complex64)",
      "tensor(complex128)", "tensor(float8e4m3fn)",   "tensor(float8e4m3fnuz)",
      "tensor(float8e5m2)", "tensor(float8e5m2fnuz)", "tensor(uint4)",
      "tensor(int4)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_types_ir11() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",      "tensor(uint16)",         "tensor(uint32)",
      "tensor(uint64)",     "tensor(int8)",           "tensor(int16)",
      "tensor(int32)",      "tensor(int64)",          "tensor(bfloat16)",
      "tensor(float16)",    "tensor(float)",          "tensor(double)",
      "tensor(string)",     "tensor(bool)",           "tensor(complex64)",
      "tensor(complex128)", "tensor(float8e4m3fn)",   "tensor(float8e4m3fnuz)",
      "tensor(float8e5m2)", "tensor(float8e5m2fnuz)", "tensor(uint4)",
      "tensor(int4)",       "tensor(float4e2m1)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_types_ir12() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",      "tensor(uint16)",         "tensor(uint32)",
      "tensor(uint64)",     "tensor(int8)",           "tensor(int16)",
      "tensor(int32)",      "tensor(int64)",          "tensor(bfloat16)",
      "tensor(float16)",    "tensor(float)",          "tensor(double)",
      "tensor(string)",     "tensor(bool)",           "tensor(complex64)",
      "tensor(complex128)", "tensor(float8e4m3fn)",   "tensor(float8e4m3fnuz)",
      "tensor(float8e5m2)", "tensor(float8e5m2fnuz)", "tensor(uint4)",
      "tensor(int4)",       "tensor(float4e2m1)",     "tensor(float8e8m0)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_types_ir13() {
  static const std::vector<std::string> v = {"tensor(uint8)",        "tensor(uint16)",
                                             "tensor(uint32)",       "tensor(uint64)",
                                             "tensor(int8)",         "tensor(int16)",
                                             "tensor(int32)",        "tensor(int64)",
                                             "tensor(bfloat16)",     "tensor(float16)",
                                             "tensor(float)",        "tensor(double)",
                                             "tensor(string)",       "tensor(bool)",
                                             "tensor(complex64)",    "tensor(complex128)",
                                             "tensor(float8e4m3fn)", "tensor(float8e4m3fnuz)",
                                             "tensor(float8e5m2)",   "tensor(float8e5m2fnuz)",
                                             "tensor(uint4)",        "tensor(int4)",
                                             "tensor(float4e2m1)",   "tensor(float8e8m0)",
                                             "tensor(uint2)",        "tensor(int2)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_non_complex_tensor_types_ir10() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
      "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
      "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
      "tensor(float16)",        "tensor(float)",      "tensor(double)",
      "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
      "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
      "tensor(uint4)",          "tensor(int4)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_non_complex_tensor_types_ir11() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
      "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
      "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
      "tensor(float16)",        "tensor(float)",      "tensor(double)",
      "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
      "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
      "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_non_complex_tensor_types_ir12() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
      "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
      "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
      "tensor(float16)",        "tensor(float)",      "tensor(double)",
      "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
      "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
      "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
      "tensor(float8e8m0)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_non_complex_tensor_types_ir13() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
      "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
      "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
      "tensor(float16)",        "tensor(float)",      "tensor(double)",
      "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
      "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
      "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
      "tensor(float8e8m0)",     "tensor(uint2)",      "tensor(int2)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_non_string_tensor_types_ir13() {
  static const std::vector<std::string> v = {"tensor(uint8)",
                                             "tensor(uint16)",
                                             "tensor(uint32)",
                                             "tensor(uint64)",
                                             "tensor(int8)",
                                             "tensor(int16)",
                                             "tensor(int32)",
                                             "tensor(int64)",
                                             "tensor(bfloat16)",
                                             "tensor(float16)",
                                             "tensor(float)",
                                             "tensor(double)",
                                             "tensor(bool)",
                                             "tensor(complex64)",
                                             "tensor(complex128)",
                                             "tensor(float8e4m3fn)",
                                             "tensor(float8e4m3fnuz)",
                                             "tensor(float8e5m2)",
                                             "tensor(float8e5m2fnuz)",
                                             "tensor(uint4)",
                                             "tensor(int4)",
                                             "tensor(float4e2m1)",
                                             "tensor(float8e8m0)",
                                             "tensor(uint2)",
                                             "tensor(int2)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_float_types_ir4() {
  static const std::vector<std::string> v = {"tensor(bfloat16)", "tensor(float16)", "tensor(float)",
                                             "tensor(double)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_float_types_ir9() {
  static const std::vector<std::string> v = {"tensor(bfloat16)",     "tensor(float16)",
                                             "tensor(float)",        "tensor(double)",
                                             "tensor(float8e4m3fn)", "tensor(float8e4m3fnuz)",
                                             "tensor(float8e5m2)",   "tensor(float8e5m2fnuz)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_float_types_plus_Xint8_ir4() {
  static const std::vector<std::string> v = {"tensor(bfloat16)", "tensor(float16)",
                                             "tensor(float)",    "tensor(double)",
                                             "tensor(int8)",     "tensor(uint8)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_non_complex_numeric_types_plus_bool_ir4() {
  static const std::vector<std::string> v = {
      "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",   "tensor(int8)",
      "tensor(int16)", "tensor(int32)",  "tensor(int64)",  "tensor(bfloat16)", "tensor(float16)",
      "tensor(float)", "tensor(double)", "tensor(bool)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_numeric_types_ir4() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",   "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
      "tensor(int8)",    "tensor(int16)",  "tensor(int32)",  "tensor(int64)",
      "tensor(float16)", "tensor(float)",  "tensor(double)", "tensor(bfloat16)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_numeric_types_ir9() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",         "tensor(uint16)",         "tensor(uint32)",
      "tensor(uint64)",        "tensor(int8)",           "tensor(int16)",
      "tensor(int32)",         "tensor(int64)",          "tensor(float16)",
      "tensor(float)",         "tensor(double)",         "tensor(bfloat16)",
      "tensor(float8e4m3fn)",  "tensor(float8e4m3fnuz)", "tensor(float8e5m2)",
      "tensor(float8e5m2fnuz)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_numeric_types_ir10() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",          "tensor(uint16)",         "tensor(uint32)",
      "tensor(uint64)",         "tensor(int8)",           "tensor(int16)",
      "tensor(int32)",          "tensor(int64)",          "tensor(float16)",
      "tensor(float)",          "tensor(double)",         "tensor(bfloat16)",
      "tensor(float8e4m3fn)",   "tensor(float8e4m3fnuz)", "tensor(float8e5m2)",
      "tensor(float8e5m2fnuz)", "tensor(uint4)",          "tensor(int4)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_numeric_types_ir11() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",          "tensor(uint16)",         "tensor(uint32)",
      "tensor(uint64)",         "tensor(int8)",           "tensor(int16)",
      "tensor(int32)",          "tensor(int64)",          "tensor(float16)",
      "tensor(float)",          "tensor(double)",         "tensor(bfloat16)",
      "tensor(float8e4m3fn)",   "tensor(float8e4m3fnuz)", "tensor(float8e5m2)",
      "tensor(float8e5m2fnuz)", "tensor(uint4)",          "tensor(int4)",
      "tensor(float4e2m1)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_numeric_types_ir12() {
  static const std::vector<std::string> v = {
      "tensor(uint8)",          "tensor(uint16)",         "tensor(uint32)",
      "tensor(uint64)",         "tensor(int8)",           "tensor(int16)",
      "tensor(int32)",          "tensor(int64)",          "tensor(float16)",
      "tensor(float)",          "tensor(double)",         "tensor(bfloat16)",
      "tensor(float8e4m3fn)",   "tensor(float8e4m3fnuz)", "tensor(float8e5m2)",
      "tensor(float8e5m2fnuz)", "tensor(uint4)",          "tensor(int4)",
      "tensor(float4e2m1)",     "tensor(float8e8m0)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_numeric_types_ir13() {
  static const std::vector<std::string> v = {"tensor(uint8)",        "tensor(uint16)",
                                             "tensor(uint32)",       "tensor(uint64)",
                                             "tensor(int8)",         "tensor(int16)",
                                             "tensor(int32)",        "tensor(int64)",
                                             "tensor(float16)",      "tensor(float)",
                                             "tensor(double)",       "tensor(bfloat16)",
                                             "tensor(float8e4m3fn)", "tensor(float8e4m3fnuz)",
                                             "tensor(float8e5m2)",   "tensor(float8e5m2fnuz)",
                                             "tensor(uint4)",        "tensor(int4)",
                                             "tensor(float4e2m1)",   "tensor(float8e8m0)",
                                             "tensor(uint2)",        "tensor(int2)"};
  return v;
}

const std::vector<std::string> &OpSchema::numeric_types_for_math_reduction() {
  static const std::vector<std::string> v = {"tensor(uint32)", "tensor(uint64)",  "tensor(int32)",
                                             "tensor(int64)",  "tensor(float16)", "tensor(float)",
                                             "tensor(double)"};
  return v;
}

const std::vector<std::string> &OpSchema::numeric_types_for_math_reduction_ir4() {
  static const std::vector<std::string> v = {"tensor(uint32)", "tensor(uint64)",  "tensor(int32)",
                                             "tensor(int64)",  "tensor(float16)", "tensor(float)",
                                             "tensor(double)", "tensor(bfloat16)"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_sequence_types() {
  static const std::vector<std::string> v = {
      "seq(tensor(uint8))",  "seq(tensor(uint16))",    "seq(tensor(uint32))",
      "seq(tensor(uint64))", "seq(tensor(int8))",      "seq(tensor(int16))",
      "seq(tensor(int32))",  "seq(tensor(int64))",     "seq(tensor(float16))",
      "seq(tensor(float))",  "seq(tensor(double))",    "seq(tensor(string))",
      "seq(tensor(bool))",   "seq(tensor(complex64))", "seq(tensor(complex128))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_sequence_types_ir4() {
  static const std::vector<std::string> v = {
      "seq(tensor(uint8))",     "seq(tensor(uint16))", "seq(tensor(uint32))",
      "seq(tensor(uint64))",    "seq(tensor(int8))",   "seq(tensor(int16))",
      "seq(tensor(int32))",     "seq(tensor(int64))",  "seq(tensor(bfloat16))",
      "seq(tensor(float16))",   "seq(tensor(float))",  "seq(tensor(double))",
      "seq(tensor(string))",    "seq(tensor(bool))",   "seq(tensor(complex64))",
      "seq(tensor(complex128))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_sequence_types_ir9() {
  static const std::vector<std::string> v = {
      "seq(tensor(uint8))",      "seq(tensor(uint16))",        "seq(tensor(uint32))",
      "seq(tensor(uint64))",     "seq(tensor(int8))",          "seq(tensor(int16))",
      "seq(tensor(int32))",      "seq(tensor(int64))",         "seq(tensor(bfloat16))",
      "seq(tensor(float16))",    "seq(tensor(float))",         "seq(tensor(double))",
      "seq(tensor(string))",     "seq(tensor(bool))",          "seq(tensor(complex64))",
      "seq(tensor(complex128))", "seq(tensor(float8e4m3fn))",  "seq(tensor(float8e4m3fnuz))",
      "seq(tensor(float8e5m2))", "seq(tensor(float8e5m2fnuz))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_sequence_types_ir10() {
  static const std::vector<std::string> v = {
      "seq(tensor(uint8))",      "seq(tensor(uint16))",         "seq(tensor(uint32))",
      "seq(tensor(uint64))",     "seq(tensor(int8))",           "seq(tensor(int16))",
      "seq(tensor(int32))",      "seq(tensor(int64))",          "seq(tensor(bfloat16))",
      "seq(tensor(float16))",    "seq(tensor(float))",          "seq(tensor(double))",
      "seq(tensor(string))",     "seq(tensor(bool))",           "seq(tensor(complex64))",
      "seq(tensor(complex128))", "seq(tensor(float8e4m3fn))",   "seq(tensor(float8e4m3fnuz))",
      "seq(tensor(float8e5m2))", "seq(tensor(float8e5m2fnuz))", "seq(tensor(uint4))",
      "seq(tensor(int4))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_sequence_types_ir11() {
  static const std::vector<std::string> v = {
      "seq(tensor(uint8))",      "seq(tensor(uint16))",         "seq(tensor(uint32))",
      "seq(tensor(uint64))",     "seq(tensor(int8))",           "seq(tensor(int16))",
      "seq(tensor(int32))",      "seq(tensor(int64))",          "seq(tensor(bfloat16))",
      "seq(tensor(float16))",    "seq(tensor(float))",          "seq(tensor(double))",
      "seq(tensor(string))",     "seq(tensor(bool))",           "seq(tensor(complex64))",
      "seq(tensor(complex128))", "seq(tensor(float8e4m3fn))",   "seq(tensor(float8e4m3fnuz))",
      "seq(tensor(float8e5m2))", "seq(tensor(float8e5m2fnuz))", "seq(tensor(uint4))",
      "seq(tensor(int4))",       "seq(tensor(float4e2m1))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_sequence_types_ir12() {
  static const std::vector<std::string> v = {
      "seq(tensor(uint8))",      "seq(tensor(uint16))",         "seq(tensor(uint32))",
      "seq(tensor(uint64))",     "seq(tensor(int8))",           "seq(tensor(int16))",
      "seq(tensor(int32))",      "seq(tensor(int64))",          "seq(tensor(bfloat16))",
      "seq(tensor(float16))",    "seq(tensor(float))",          "seq(tensor(double))",
      "seq(tensor(string))",     "seq(tensor(bool))",           "seq(tensor(complex64))",
      "seq(tensor(complex128))", "seq(tensor(float8e4m3fn))",   "seq(tensor(float8e4m3fnuz))",
      "seq(tensor(float8e5m2))", "seq(tensor(float8e5m2fnuz))", "seq(tensor(uint4))",
      "seq(tensor(int4))",       "seq(tensor(float4e2m1))",     "seq(tensor(float8e8m0))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_tensor_sequence_types_ir13() {
  static const std::vector<std::string> v = {
      "seq(tensor(uint8))",        "seq(tensor(uint16))",
      "seq(tensor(uint32))",       "seq(tensor(uint64))",
      "seq(tensor(int8))",         "seq(tensor(int16))",
      "seq(tensor(int32))",        "seq(tensor(int64))",
      "seq(tensor(bfloat16))",     "seq(tensor(float16))",
      "seq(tensor(float))",        "seq(tensor(double))",
      "seq(tensor(string))",       "seq(tensor(bool))",
      "seq(tensor(complex64))",    "seq(tensor(complex128))",
      "seq(tensor(float8e4m3fn))", "seq(tensor(float8e4m3fnuz))",
      "seq(tensor(float8e5m2))",   "seq(tensor(float8e5m2fnuz))",
      "seq(tensor(uint4))",        "seq(tensor(int4))",
      "seq(tensor(float4e2m1))",   "seq(tensor(float8e8m0))",
      "seq(tensor(uint2))",        "seq(tensor(int2))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_optional_types() {
  static const std::vector<std::string> v = {
      "optional(seq(tensor(uint8)))",      "optional(seq(tensor(uint16)))",
      "optional(seq(tensor(uint32)))",     "optional(seq(tensor(uint64)))",
      "optional(seq(tensor(int8)))",       "optional(seq(tensor(int16)))",
      "optional(seq(tensor(int32)))",      "optional(seq(tensor(int64)))",
      "optional(seq(tensor(float16)))",    "optional(seq(tensor(float)))",
      "optional(seq(tensor(double)))",     "optional(seq(tensor(string)))",
      "optional(seq(tensor(bool)))",       "optional(seq(tensor(complex64)))",
      "optional(seq(tensor(complex128)))", "optional(tensor(uint8))",
      "optional(tensor(uint16))",          "optional(tensor(uint32))",
      "optional(tensor(uint64))",          "optional(tensor(int8))",
      "optional(tensor(int16))",           "optional(tensor(int32))",
      "optional(tensor(int64))",           "optional(tensor(float16))",
      "optional(tensor(float))",           "optional(tensor(double))",
      "optional(tensor(string))",          "optional(tensor(bool))",
      "optional(tensor(complex64))",       "optional(tensor(complex128))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_optional_types_ir4() {
  static const std::vector<std::string> v = {
      "optional(seq(tensor(uint8)))",     "optional(seq(tensor(uint16)))",
      "optional(seq(tensor(uint32)))",    "optional(seq(tensor(uint64)))",
      "optional(seq(tensor(int8)))",      "optional(seq(tensor(int16)))",
      "optional(seq(tensor(int32)))",     "optional(seq(tensor(int64)))",
      "optional(seq(tensor(bfloat16)))",  "optional(seq(tensor(float16)))",
      "optional(seq(tensor(float)))",     "optional(seq(tensor(double)))",
      "optional(seq(tensor(string)))",    "optional(seq(tensor(bool)))",
      "optional(seq(tensor(complex64)))", "optional(seq(tensor(complex128)))",
      "optional(tensor(uint8))",          "optional(tensor(uint16))",
      "optional(tensor(uint32))",         "optional(tensor(uint64))",
      "optional(tensor(int8))",           "optional(tensor(int16))",
      "optional(tensor(int32))",          "optional(tensor(int64))",
      "optional(tensor(bfloat16))",       "optional(tensor(float16))",
      "optional(tensor(float))",          "optional(tensor(double))",
      "optional(tensor(string))",         "optional(tensor(bool))",
      "optional(tensor(complex64))",      "optional(tensor(complex128))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_optional_types_ir9() {
  static const std::vector<std::string> v = {
      "optional(seq(tensor(uint8)))",     "optional(seq(tensor(uint16)))",
      "optional(seq(tensor(uint32)))",    "optional(seq(tensor(uint64)))",
      "optional(seq(tensor(int8)))",      "optional(seq(tensor(int16)))",
      "optional(seq(tensor(int32)))",     "optional(seq(tensor(int64)))",
      "optional(seq(tensor(bfloat16)))",  "optional(seq(tensor(float16)))",
      "optional(seq(tensor(float)))",     "optional(seq(tensor(double)))",
      "optional(seq(tensor(string)))",    "optional(seq(tensor(bool)))",
      "optional(seq(tensor(complex64)))", "optional(seq(tensor(complex128)))",
      "optional(tensor(uint8))",          "optional(tensor(uint16))",
      "optional(tensor(uint32))",         "optional(tensor(uint64))",
      "optional(tensor(int8))",           "optional(tensor(int16))",
      "optional(tensor(int32))",          "optional(tensor(int64))",
      "optional(tensor(bfloat16))",       "optional(tensor(float16))",
      "optional(tensor(float))",          "optional(tensor(double))",
      "optional(tensor(string))",         "optional(tensor(bool))",
      "optional(tensor(complex64))",      "optional(tensor(complex128))",
      "optional(tensor(float8e4m3fn))",   "optional(tensor(float8e4m3fnuz))",
      "optional(tensor(float8e5m2))",     "optional(tensor(float8e5m2fnuz))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_optional_types_ir10() {
  static const std::vector<std::string> v = {
      "optional(seq(tensor(uint8)))",     "optional(seq(tensor(uint16)))",
      "optional(seq(tensor(uint32)))",    "optional(seq(tensor(uint64)))",
      "optional(seq(tensor(int8)))",      "optional(seq(tensor(int16)))",
      "optional(seq(tensor(int32)))",     "optional(seq(tensor(int64)))",
      "optional(seq(tensor(bfloat16)))",  "optional(seq(tensor(float16)))",
      "optional(seq(tensor(float)))",     "optional(seq(tensor(double)))",
      "optional(seq(tensor(string)))",    "optional(seq(tensor(bool)))",
      "optional(seq(tensor(complex64)))", "optional(seq(tensor(complex128)))",
      "optional(tensor(uint8))",          "optional(tensor(uint16))",
      "optional(tensor(uint32))",         "optional(tensor(uint64))",
      "optional(tensor(int8))",           "optional(tensor(int16))",
      "optional(tensor(int32))",          "optional(tensor(int64))",
      "optional(tensor(bfloat16))",       "optional(tensor(float16))",
      "optional(tensor(float))",          "optional(tensor(double))",
      "optional(tensor(string))",         "optional(tensor(bool))",
      "optional(tensor(complex64))",      "optional(tensor(complex128))",
      "optional(tensor(float8e4m3fn))",   "optional(tensor(float8e4m3fnuz))",
      "optional(tensor(float8e5m2))",     "optional(tensor(float8e5m2fnuz))",
      "optional(tensor(uint4))",          "optional(tensor(int4))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_optional_types_ir11() {
  static const std::vector<std::string> v = {
      "optional(seq(tensor(uint8)))",     "optional(seq(tensor(uint16)))",
      "optional(seq(tensor(uint32)))",    "optional(seq(tensor(uint64)))",
      "optional(seq(tensor(int8)))",      "optional(seq(tensor(int16)))",
      "optional(seq(tensor(int32)))",     "optional(seq(tensor(int64)))",
      "optional(seq(tensor(bfloat16)))",  "optional(seq(tensor(float16)))",
      "optional(seq(tensor(float)))",     "optional(seq(tensor(double)))",
      "optional(seq(tensor(string)))",    "optional(seq(tensor(bool)))",
      "optional(seq(tensor(complex64)))", "optional(seq(tensor(complex128)))",
      "optional(tensor(uint8))",          "optional(tensor(uint16))",
      "optional(tensor(uint32))",         "optional(tensor(uint64))",
      "optional(tensor(int8))",           "optional(tensor(int16))",
      "optional(tensor(int32))",          "optional(tensor(int64))",
      "optional(tensor(bfloat16))",       "optional(tensor(float16))",
      "optional(tensor(float))",          "optional(tensor(double))",
      "optional(tensor(string))",         "optional(tensor(bool))",
      "optional(tensor(complex64))",      "optional(tensor(complex128))",
      "optional(tensor(float8e4m3fn))",   "optional(tensor(float8e4m3fnuz))",
      "optional(tensor(float8e5m2))",     "optional(tensor(float8e5m2fnuz))",
      "optional(tensor(uint4))",          "optional(tensor(int4))",
      "optional(tensor(float4e2m1))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_optional_types_ir12() {
  static const std::vector<std::string> v = {
      "optional(seq(tensor(uint8)))",     "optional(seq(tensor(uint16)))",
      "optional(seq(tensor(uint32)))",    "optional(seq(tensor(uint64)))",
      "optional(seq(tensor(int8)))",      "optional(seq(tensor(int16)))",
      "optional(seq(tensor(int32)))",     "optional(seq(tensor(int64)))",
      "optional(seq(tensor(bfloat16)))",  "optional(seq(tensor(float16)))",
      "optional(seq(tensor(float)))",     "optional(seq(tensor(double)))",
      "optional(seq(tensor(string)))",    "optional(seq(tensor(bool)))",
      "optional(seq(tensor(complex64)))", "optional(seq(tensor(complex128)))",
      "optional(tensor(uint8))",          "optional(tensor(uint16))",
      "optional(tensor(uint32))",         "optional(tensor(uint64))",
      "optional(tensor(int8))",           "optional(tensor(int16))",
      "optional(tensor(int32))",          "optional(tensor(int64))",
      "optional(tensor(bfloat16))",       "optional(tensor(float16))",
      "optional(tensor(float))",          "optional(tensor(double))",
      "optional(tensor(string))",         "optional(tensor(bool))",
      "optional(tensor(complex64))",      "optional(tensor(complex128))",
      "optional(tensor(float8e4m3fn))",   "optional(tensor(float8e4m3fnuz))",
      "optional(tensor(float8e5m2))",     "optional(tensor(float8e5m2fnuz))",
      "optional(tensor(uint4))",          "optional(tensor(int4))",
      "optional(tensor(float4e2m1))",     "optional(tensor(float8e8m0))"};
  return v;
}

const std::vector<std::string> &OpSchema::all_optional_types_ir13() {
  static const std::vector<std::string> v = {
      "optional(seq(tensor(uint8)))",     "optional(seq(tensor(uint16)))",
      "optional(seq(tensor(uint32)))",    "optional(seq(tensor(uint64)))",
      "optional(seq(tensor(int8)))",      "optional(seq(tensor(int16)))",
      "optional(seq(tensor(int32)))",     "optional(seq(tensor(int64)))",
      "optional(seq(tensor(bfloat16)))",  "optional(seq(tensor(float16)))",
      "optional(seq(tensor(float)))",     "optional(seq(tensor(double)))",
      "optional(seq(tensor(string)))",    "optional(seq(tensor(bool)))",
      "optional(seq(tensor(complex64)))", "optional(seq(tensor(complex128)))",
      "optional(tensor(uint8))",          "optional(tensor(uint16))",
      "optional(tensor(uint32))",         "optional(tensor(uint64))",
      "optional(tensor(int8))",           "optional(tensor(int16))",
      "optional(tensor(int32))",          "optional(tensor(int64))",
      "optional(tensor(bfloat16))",       "optional(tensor(float16))",
      "optional(tensor(float))",          "optional(tensor(double))",
      "optional(tensor(string))",         "optional(tensor(bool))",
      "optional(tensor(complex64))",      "optional(tensor(complex128))",
      "optional(tensor(float8e4m3fn))",   "optional(tensor(float8e4m3fnuz))",
      "optional(tensor(float8e5m2))",     "optional(tensor(float8e5m2fnuz))",
      "optional(tensor(uint4))",          "optional(tensor(int4))",
      "optional(tensor(float4e2m1))",     "optional(tensor(float8e8m0))",
      "optional(tensor(uint2))",          "optional(tensor(int2))"};
  return v;
}

OpName_Domain_Version_Schema_Map &OpSchemaRegistry::map() {
  static OpName_Domain_Version_Schema_Map schema_map;
  return schema_map;
}

std::mutex &OpSchemaRegistry::Mutex() {
  static std::mutex mutex;
  return mutex;
}

const OpSchema *OpSchemaRegistry::Schema(const std::string &key, const int maxInclusiveVersion,
                                         const std::string &domain) {
  return Instance()->GetSchema(key, maxInclusiveVersion, domain);
}

const OpSchema *OpSchemaRegistry::Schema(const std::string &key, const std::string &domain) {
  const auto &schema_map = map();
  auto it_name = schema_map.find(key);
  if (it_name == schema_map.end()) {
    return nullptr;
  }
  auto it_domain = it_name->second.find(domain);
  if (it_domain == it_name->second.end()) {
    return nullptr;
  }
  if (it_domain->second.empty()) {
    return nullptr;
  }
  // Return the latest non-deprecated version.
  for (auto it = it_domain->second.rbegin(); it != it_domain->second.rend(); ++it) {
    if (!it->second.deprecated()) {
      return &it->second;
    }
  }
  return nullptr;
}

std::vector<OpSchema> OpSchemaRegistry::get_all_schemas() {
  bool register_schemas = false;
  {
    std::lock_guard<std::mutex> guard(Mutex());
    register_schemas = map().empty();
  }
  if (register_schemas) {
    RegisterAllOnnxOperatorSchemas();
  }

  std::vector<OpSchema> result;
  std::lock_guard<std::mutex> guard(Mutex());
  const auto &schema_map = map();
  for (const auto &[op_name, domain_map] : schema_map) {
    (void)op_name;
    for (const auto &[domain, ver_map] : domain_map) {
      (void)domain;
      if (!ver_map.empty()) {
        result.push_back(ver_map.rbegin()->second);
      }
    }
  }
  return result;
}

std::vector<OpSchema> OpSchemaRegistry::get_all_schemas_with_history() {
  bool register_schemas = false;
  {
    std::lock_guard<std::mutex> guard(Mutex());
    register_schemas = map().empty();
  }
  if (register_schemas) {
    RegisterAllOnnxOperatorSchemas();
  }

  std::vector<OpSchema> result;
  std::lock_guard<std::mutex> guard(Mutex());
  const auto &schema_map = map();
  for (const auto &[op_name, domain_map] : schema_map) {
    (void)op_name;
    for (const auto &[domain, ver_map] : domain_map) {
      (void)domain;
      for (const auto &[ver, schema] : ver_map) {
        (void)ver;
        result.push_back(schema);
      }
    }
  }
  return result;
}

const OpSchema *OpSchemaRegistry::GetSchema(const std::string &key, const int maxInclusiveVersion,
                                            const std::string &domain) const {
  std::lock_guard<std::mutex> guard(Mutex());
  const auto &schema_map = map();
  EXT_ENFORCE(schema_map.size() > 0, "No schema is registered.");
  auto it_name = schema_map.find(key);
  if (it_name == schema_map.end()) {
    return nullptr;
  }
  auto it_domain = it_name->second.find(domain);
  if (it_domain == it_name->second.end()) {
    return nullptr;
  }
  auto it = it_domain->second.upper_bound(maxInclusiveVersion);
  while (it != it_domain->second.begin()) {
    --it;
    if (!it->second.deprecated()) {
      return &it->second;
    }
  }
  return nullptr;
}

OpSchemaRegistry::DomainToVersionRange::DomainToVersionRange() {
  map_[ONNX_DOMAIN] = std::make_pair(1, 27);
  map_[AI_ONNX_ML_DOMAIN] = std::make_pair(1, 5);
  map_[AI_ONNX_TRAINING_DOMAIN] = std::make_pair(1, 1);
  map_[AI_ONNX_PREVIEW_DOMAIN] = std::make_pair(1, 1);
  map_[AI_ONNX_PREVIEW_TRAINING_DOMAIN] = std::make_pair(1, 1);

  last_release_version_map_[ONNX_DOMAIN] = map_[ONNX_DOMAIN].second;
  last_release_version_map_[AI_ONNX_ML_DOMAIN] = map_[AI_ONNX_ML_DOMAIN].second;
  last_release_version_map_[AI_ONNX_TRAINING_DOMAIN] = map_[AI_ONNX_TRAINING_DOMAIN].second;
  last_release_version_map_[AI_ONNX_PREVIEW_DOMAIN] = map_[AI_ONNX_PREVIEW_DOMAIN].second;
  last_release_version_map_[AI_ONNX_PREVIEW_TRAINING_DOMAIN] =
      map_[AI_ONNX_PREVIEW_TRAINING_DOMAIN].second;
}

void OpSchemaRegistry::DomainToVersionRange::AddDomainToVersion(const std::string &domain,
                                                                int min_version, int max_version,
                                                                int last_release_version) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (map_.count(domain) != 0 || last_release_version_map_.count(domain) != 0) {
    fail_schema("Trying to add a domain that already exists: ", domain);
  }

  map_[domain] = std::make_pair(min_version, max_version);
  if (last_release_version == -1) {
    last_release_version = max_version;
  }
  last_release_version_map_[domain] = last_release_version;
}

void OpSchemaRegistry::DomainToVersionRange::UpdateDomainToVersion(const std::string &domain,
                                                                   int min_version, int max_version,
                                                                   int last_release_version) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (map_.count(domain) == 0 || last_release_version_map_.count(domain) == 0) {
    fail_schema("Trying to update a domain that does not exist: ", domain);
  }

  map_[domain] = std::make_pair(min_version, max_version);
  if (last_release_version == -1) {
    last_release_version = max_version;
  }
  last_release_version_map_[domain] = last_release_version;
}

OpSchemaRegistry::DomainToVersionRange &OpSchemaRegistry::DomainToVersionRange::Instance() {
  static DomainToVersionRange instance;
  return instance;
}

void OpSchemaRegistry::OpSchemaRegisterOnce::OpSchemaRegisterNoExcept(OpSchema &&op_schema,
                                                                      int opset_version_to_load,
                                                                      bool fail_duplicate_schema) {
  ONNX_TRY {
    OpSchemaRegisterImpl(std::move(op_schema), opset_version_to_load, fail_duplicate_schema);
  }
  ONNX_CATCH(const std::exception &e) {
    ONNX_HANDLE_EXCEPTION([&]() { std::cerr << "Schema error: " << e.what() << '\n'; });
  }
}

void OpSchemaRegistry::OpSchemaRegisterOnce::OpSchemaRegisterImpl(OpSchema &&op_schema,
                                                                  int opset_version_to_load,
                                                                  bool fail_duplicate_schema) {
  op_schema.Finalize();

  if (op_schema.SinceVersion() == OpSchema::kUninitializedSinceVersion) {
    op_schema.SinceVersion(1);
  }
  const auto ver = op_schema.SinceVersion();

  auto &schema_map = OpSchemaRegistry::map();
  auto &schema_ver_map = schema_map[op_schema.Name()][op_schema.domain()];

  if (schema_ver_map.count(ver) != 0) {
    if (fail_duplicate_schema) {
      fail_schema("Trying to register schema with name ", op_schema.Name(), " and version ", ver,
                  " but it is already registered.");
    }
    return;
  }

  if (opset_version_to_load > 0) {
    if (ver > opset_version_to_load) {
      return;
    }
    for (auto it = schema_ver_map.rbegin(); it != schema_ver_map.rend(); ++it) {
      if (it->first <= opset_version_to_load && it->first > ver) {
        return;
      }
    }
  }

  schema_ver_map.emplace(ver, std::move(op_schema));
}

void OpSchemaRegistry::OpSchemaDeregisterAll(const std::string &domain) {
  auto &schema_map = map();
  for (auto &[op_type, domain_map] : schema_map) {
    (void)op_type;
    auto it = domain_map.find(domain);
    if (it != domain_map.end()) {
      it->second.clear();
      domain_map.erase(it);
    }
  }
}

bool IsOnnxStaticRegistrationDisabled() { return false; }

} // namespace ONNX_LIGHT_NAMESPACE

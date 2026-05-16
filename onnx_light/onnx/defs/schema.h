// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file schema.h
 * @brief Declares ONNX operator schema types and registration helpers.
 *
 * This header defines OpSchema and related utility types used to describe
 * ONNX operator signatures, constraints, and shape/type inference behavior.
 */

#pragma once

#include <cstring>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/common.h"
#include "onnx/common/constants.h"
#include "onnx/defs/data_type_utils.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Exposes node- and input-dependent information to function-body builders.
 */
struct FunctionBodyBuildContext {
  /**
   * Looks up an attribute by name and returns it.
   */
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  /**
   * Reports whether the node has a non-empty input at the specified index.
   */
  virtual bool hasInput(int inputIndex) const = 0;
  /**
   * Reports whether the node has a non-empty output at the specified index.
   */
  virtual bool hasOutput(int outputIndex) const = 0;
  /**
   * Returns the declared input type for the specified index, when available.
   */
  virtual const TypeProto *getInputType(int inputIndex) const = 0;
  virtual ~FunctionBodyBuildContext() = default;
};

/**
 * Default FunctionBodyBuildContext implementation backed by NodeProto data.
 */
struct FunctionBodyBuildContextImpl : public FunctionBodyBuildContext {
  explicit FunctionBodyBuildContextImpl(const NodeProto &node_proto,
                                        const std::vector<TypeProto> &input_types = {})
      : node_proto_(node_proto), input_types_(input_types) {
    for (const auto &attr : node_proto.ref_attribute()) {
      attributesByName_[attr.ref_name().as_string()] = &attr;
    }
  }

  const AttributeProto *getAttribute(const std::string &name) const override {
    auto iter = attributesByName_.find(name);
    return iter == attributesByName_.end() ? nullptr : iter->second;
  }

  bool hasInput(int inputIndex) const override {
    if (inputIndex < 0) {
      return false;
    }
    const auto idx = static_cast<size_t>(inputIndex);
    return idx < node_proto_.ref_input().size() && !node_proto_.ref_input()[idx].empty();
  }

  bool hasOutput(int outputIndex) const override {
    if (outputIndex < 0) {
      return false;
    }
    const auto idx = static_cast<size_t>(outputIndex);
    return idx < node_proto_.ref_output().size() && !node_proto_.ref_output()[idx].empty();
  }

  const TypeProto *getInputType(int inputIndex) const override {
    if (inputIndex < 0) {
      return nullptr;
    }
    const auto idx = static_cast<size_t>(inputIndex);
    if (idx >= input_types_.size()) {
      return nullptr;
    }
    const auto &type = input_types_[idx];
    if (!type.has_tensor_type() && !type.has_sparse_tensor_type() && !type.has_sequence_type() &&
        !type.has_optional_type() && !type.has_map_type()) {
      return nullptr;
    }
    return &type;
  }

  std::unordered_map<std::string, const AttributeProto *> attributesByName_;
  NodeProto node_proto_;
  std::vector<TypeProto> input_types_;
};

/// Function predicate used to decide whether a function body applies in a context.
using FunctionBodyQueryFunction = std::function<bool(FunctionBodyBuildContext &)>;
/// Alias for an ONNX operator-set version number.
using OperatorSetVersion = int;
/// Set of allowed ONNX tensor element types for a type parameter.
using DataTypeSet = std::unordered_set<DataType>;
/// Maps type parameter names to constrained type sets and descriptions.
using TypeConstraintMap = std::unordered_map<std::string, std::pair<DataTypeSet, std::string>>;

class OpSchema;
/// Builds a function body from a schema and call-site context.
using ContextDependentFunctionBodyBuilder =
    std::function<bool(const FunctionBodyBuildContext &, const OpSchema &, FunctionProto &)>;

/**
 * Represents schema validation and registration errors.
 */
class SchemaError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;

  explicit SchemaError(const std::string &message) : std::runtime_error(message) {}

  ONNX_API const char *what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  ONNX_API void AppendContext(const std::string &context) {
    expanded_message_ =
        ONNX_LIGHT_NAMESPACE::MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

private:
  std::string expanded_message_;
};

#define fail_schema(...)                                                                           \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::SchemaError(ONNX_LIGHT_NAMESPACE::MakeString(__VA_ARGS__)))

/**
 * Describes one ONNX operator schema (attributes, inputs, outputs, constraints).
 */
class OpSchema final {
public:
  static constexpr int kUninitializedSinceVersion = -1;

  /**
   * Specifies whether a formal parameter is single, optional, or variadic.
   */
  enum FormalParameterOption : uint8_t {
    Single = 0,
    Optional = 1,
    Variadic = 2,
  };

  /**
   * Describes differentiability of an input or output parameter.
   */
  enum DifferentiationCategory : uint8_t {
    Unknown = 0,
    Differentiable = 1,
    NonDifferentiable = 2,
  };

  /**
   * Describes whether an operator is deterministic.
   */
  enum class NodeDeterminism : uint8_t {
    Unknown = 0,
    NonDeterministic = 1,
    Deterministic = 2,
  };

  /**
   * Represents one named input/output parameter in an operator signature.
   * This class stores type constraints, arity, and differentiability metadata.
   */
  class FormalParameter final {
  public:
    FormalParameter() = default;

    explicit FormalParameter(std::string name, DataTypeSet allowed_type_set, std::string type_str,
                             std::string description, FormalParameterOption param_option = Single,
                             bool is_homogeneous = true, int min_arity = 1,
                             DifferentiationCategory differentiation_category = Unknown)
        : name_(std::move(name)), type_set_(std::move(allowed_type_set)),
          type_str_(std::move(type_str)), description_(std::move(description)),
          param_option_(param_option), is_homogeneous_(is_homogeneous), min_arity_(min_arity),
          differentiation_category_(differentiation_category) {}

    explicit FormalParameter(std::string name, std::string description, std::string type_str,
                             FormalParameterOption param_option = Single,
                             bool is_homogeneous = true, int min_arity = 1,
                             DifferentiationCategory differentiation_category = Unknown)
        : name_(std::move(name)), type_str_(std::move(type_str)),
          description_(std::move(description)), param_option_(param_option),
          is_homogeneous_(is_homogeneous), min_arity_(min_arity),
          differentiation_category_(differentiation_category) {}

    ONNX_API const std::string &GetName() const;
    ONNX_API const DataTypeSet &GetTypes() const;
    ONNX_API const std::string &GetTypeStr() const;
    ONNX_API const std::string &GetDescription() const;
    ONNX_API FormalParameterOption GetOption() const;
    ONNX_API bool GetIsHomogeneous() const;
    ONNX_API int GetMinArity() const;
    ONNX_API DifferentiationCategory GetDifferentiationCategory() const;

  private:
    friend class OpSchema;

    DataTypeSet &MutableTypes();

    std::string name_;
    DataTypeSet type_set_;
    std::string type_str_;
    std::string description_;
    FormalParameterOption param_option_{};
    bool is_homogeneous_{};
    int min_arity_{};
    DifferentiationCategory differentiation_category_{};
  };

  /**
   * Marks operator support level.
   */
  enum class SupportType : uint8_t {
    COMMON,
    EXPERIMENTAL,
  };

  /**
   * Represents one operator attribute declaration.
   * This struct stores declaration metadata and optional default values.
   */
  struct Attribute final {
    Attribute(std::string name_, std::string description_, AttributeProto::AttributeType type_,
              bool required_)
        : name(std::move(name_)), description(std::move(description_)), type(type_),
          required(required_) {}

    Attribute(std::string name_, std::string description_, AttributeProto default_value_)
        : name(std::move(name_)), description(std::move(description_)),
          type(default_value_.ref_type()), required(false),
          default_value(std::move(default_value_)) {}

    const std::string name;
    const std::string description;
    AttributeProto::AttributeType type;
    bool required;
    AttributeProto default_value;
  };

  /**
   * Represents one type-parameter declaration and allowed concrete types.
   * This struct captures schema-level type constraints for formal parameters.
   */
  struct TypeConstraintParam final {
    TypeConstraintParam(std::string type_param_str_, std::vector<std::string> allowed_type_strs_,
                        std::string description_)
        : type_param_str(std::move(type_param_str_)),
          allowed_type_strs(std::move(allowed_type_strs_)), description(std::move(description_)) {}

    std::string type_param_str;
    std::vector<std::string> allowed_type_strs;
    std::string description;
  };

  OpSchema() : OpSchema("unknown", "unknown", 0) {}
  OpSchema(std::string name, std::string file, int line)
      : name_(std::move(name)), file_(std::move(file)), line_(line), support_(SupportType::COMMON) {
  }

  /// Returns the source file where the schema declaration is defined.
  ONNX_API const std::string &file() const { return file_; }
  /// Returns the source line where the schema declaration is defined.
  ONNX_API int line() const { return line_; }
  /// Returns the schema support level.
  ONNX_API SupportType support_level() const { return support_; }
  /// Returns the schema documentation string, or null when empty.
  ONNX_API const char *doc() const { return doc_.empty() ? nullptr : doc_.c_str(); }
  /// Returns the schema domain.
  ONNX_API const std::string &domain() const { return domain_; }
  /// Returns the declared attributes keyed by attribute name.
  ONNX_API const std::unordered_map<std::string, Attribute> &attributes() const {
    return attributes_;
  }
  /// Returns the formal input parameters.
  ONNX_API const std::vector<FormalParameter> &inputs() const { return inputs_; }
  /// Returns the formal output parameters.
  ONNX_API const std::vector<FormalParameter> &outputs() const { return outputs_; }
  /// Returns the type-constraint declarations as authored in schema registration code.
  ONNX_API const std::vector<TypeConstraintParam> &typeConstraintParams() const {
    return type_constraint_params_;
  }
  /// Returns the normalized type-constraint map keyed by type parameter name.
  ONNX_API const TypeConstraintMap &typeConstraintMap() const { return type_constraints_; }
  /// Returns the operator name.
  ONNX_API const std::string &Name() const { return name_; }
  /// Returns the operator-set version in which this schema was introduced.
  ONNX_API OperatorSetVersion SinceVersion() const { return since_version_; }
  /// Returns SinceVersion() as an int (legacy alias of SinceVersion()).
  ONNX_API int since_version() const { return since_version_; }
  /// Returns whether the schema has been marked as deprecated.
  ONNX_API bool deprecated() const { return deprecated_; }
  /// Returns the minimum accepted number of inputs.
  ONNX_API int min_input() const { return min_input_; }
  /// Returns the maximum accepted number of inputs.
  ONNX_API int max_input() const { return max_input_; }
  /// Returns the minimum accepted number of outputs.
  ONNX_API int min_output() const { return min_output_; }
  /// Returns the maximum accepted number of outputs.
  ONNX_API int max_output() const { return max_output_; }
  /// Returns whether a type-and-shape inference function has been set.
  ONNX_API bool has_type_and_shape_inference_function() const {
    return static_cast<bool>(tensor_inference_function_);
  }
  /// Returns whether a partial data-propagation function has been set.
  ONNX_API bool has_data_propagation_function() const {
    return static_cast<bool>(data_propagation_function_);
  }
  /// Returns whether at least one static function body has been registered.
  ONNX_API bool HasFunction() const { return !opset_version_to_function_body_.empty(); }
  /// Returns whether at least one context-dependent function builder is registered.
  ONNX_API bool HasContextDependentFunction() const {
    return !opset_version_to_function_builder_.empty();
  }
  /// Returns whether a context-dependent function builder exists for an opset version.
  ONNX_API bool HasContextDependentFunctionWithOpsetVersion(int opset_version) const {
    return opset_version_to_function_builder_.count(opset_version) > 0;
  }
  /// Builds a function body from a context-dependent function builder.
  ONNX_API void BuildContextDependentFunction(const FunctionBodyBuildContext &ctx,
                                              FunctionProto &function_proto,
                                              int opset_version = kUninitializedSinceVersion) const;
  /// Returns the opset versions with stored static function bodies.
  ONNX_API std::vector<int> function_opset_versions() const;
  /// Returns the opset versions with context-dependent function builders.
  ONNX_API std::vector<int> context_dependent_function_opset_versions() const;
  /// Returns the configured type-and-shape inference function.
  ONNX_API const InferenceFunction &GetTypeAndShapeInferenceFunction() const {
    return tensor_inference_function_;
  }
  /// Returns the configured partial data-propagation function.
  ONNX_API const DataPropagationFunction &GetDataPropagationFunction() const {
    return data_propagation_function_;
  }

  /// Sets the schema since-version.
  ONNX_API OpSchema &SinceVersion(OperatorSetVersion n);
  /// Marks this schema as deprecated.
  ONNX_API OpSchema &Deprecate();
  /// Restricts valid input arities to the provided set.
  ONNX_API OpSchema &NumInputs(std::unordered_set<int> allowed_input_nums);
  /// Restricts valid output arities to the provided set.
  ONNX_API OpSchema &NumOutputs(std::unordered_set<int> allowed_output_nums);
  /// Sets the type-and-shape inference callback.
  ONNX_API OpSchema &TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction);
  /// Sets the partial data-propagation callback.
  ONNX_API OpSchema &
  PartialDataPropagationFunction(DataPropagationFunction dataPropagationFunction);
  /// Sets the schema support level.
  ONNX_API OpSchema &SetSupportLevel(SupportType supportType);
  /// Sets the operator name from a C string.
  ONNX_API OpSchema &SetName(const char *name);
  /// Sets the operator name from a string.
  ONNX_API OpSchema &SetName(std::string name);
  /// Sets declaration source location from a C-string file path.
  ONNX_API OpSchema &SetLocation(const char *file, int line);
  /// Sets declaration source location from a string file path.
  ONNX_API OpSchema &SetLocation(std::string file, int line);
  /// Sets the schema domain from a C string.
  ONNX_API OpSchema &SetDomain(const char *domain);
  /// Sets the schema domain from a string.
  ONNX_API OpSchema &SetDomain(std::string domain);
  /// Sets the operator documentation string from a C string.
  ONNX_API OpSchema &SetDoc(const char *doc);
  /// Sets the operator documentation string from a string.
  ONNX_API OpSchema &SetDoc(const std::string &doc);

  /// Adds a pre-built attribute declaration.
  ONNX_API OpSchema &Attr(Attribute attr);
  /// Adds a required-or-optional attribute declaration with explicit type.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType type, bool required = true);
  /// Adds a required-or-optional attribute declaration with explicit type (C-string overload).
  ONNX_API OpSchema &Attr(const char *name, const char *description,
                          AttributeProto::AttributeType type, bool required = true);

  /// Adds an integer attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type, const int64_t &default_value);
  /// Adds a float attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type, const float &default_value);
  /// Adds a string attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const std::string &default_value);
  /// Adds a tensor attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const TensorProto &default_value);
  /// Adds a graph attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type, const GraphProto &default_value);
  /// Adds a type-proto attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type, const TypeProto &default_value);

  /// Adds an integer-list attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const std::vector<int64_t> &default_value);
  /// Adds a float-list attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const std::vector<float> &default_value);
  /// Adds a string-list attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const std::vector<std::string> &default_value);
  /// Adds a tensor-list attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const std::vector<TensorProto> &default_value);
  /// Adds a graph-list attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const std::vector<GraphProto> &default_value);
  /// Adds a type-proto-list attribute declaration with a default value.
  ONNX_API OpSchema &Attr(std::string name, std::string description,
                          AttributeProto::AttributeType attr_type,
                          const std::vector<TypeProto> &default_value);

  /// Allows attributes that are not explicitly declared by this schema.
  ONNX_API OpSchema &AllowUncheckedAttributes();

  /// Declares a formal input parameter by index.
  ONNX_API OpSchema &Input(int n, FormalParameter formal_parameter);
  /// Declares a formal input parameter with field values.
  ONNX_API OpSchema &Input(int n, std::string name, const std::string &description,
                           std::string type_str, FormalParameterOption param_option = Single,
                           bool is_homogeneous = true, int min_arity = 1,
                           DifferentiationCategory differentiation_category = Unknown);

  /// Declares a formal output parameter by index.
  ONNX_API OpSchema &Output(int n, FormalParameter formal_parameter);
  /// Declares a formal output parameter with field values.
  ONNX_API OpSchema &Output(int n, std::string name, const std::string &description,
                            std::string type_str, FormalParameterOption param_option = Single,
                            bool is_homogeneous = true, int min_arity = 1,
                            DifferentiationCategory differentiation_category = Unknown);

  /// Adds a type-parameter constraint declaration.
  ONNX_API OpSchema &TypeConstraint(std::string type_str, std::vector<std::string> constraints,
                                    std::string description);
  /// Adds a type-parameter constraint declaration (C-string/initializer-list overload).
  ONNX_API OpSchema &TypeConstraint(const char *type_str,
                                    std::initializer_list<const char *> constraints,
                                    const char *description);

  /// Registers a context-dependent function body builder.
  ONNX_API OpSchema &
  SetContextDependentFunctionBodyBuilder(ContextDependentFunctionBodyBuilder functionBuilder,
                                         int opset_version = kUninitializedSinceVersion);

  /// Registers a static function body.
  ONNX_API OpSchema &FunctionBody(const std::vector<NodeProto> &func_nodes,
                                  int opset_version = kUninitializedSinceVersion);
  /// Returns a stored function body matching the requested opset version.
  ONNX_API const FunctionProto *
  GetFunction(int requested_opset_version = kUninitializedSinceVersion,
              bool validate = false) const;

  // Populates function_body with the schema's name, domain, doc_string, inputs,
  // outputs, attributes, and opset imports.  Called automatically from Finalize()
  // for every stored static function body.
  ONNX_API void BuildFunction(FunctionProto &function_body) const;

  /// Verifies that the inferred input/output types satisfy schema constraints.
  ONNX_API void CheckInputOutputType(struct InferenceContext &ctx) const;
  /// Verifies that a node matches this schema declaration.
  ONNX_API void Verify(const NodeProto &node) const;
  /// Applies a caller-provided populator function to this schema.
  ONNX_API OpSchema &FillUsing(const std::function<void(OpSchema &)> &populator);
  /// Finalizes internal metadata before registration.
  ONNX_API void Finalize();

  /// Sets whether node executions are deterministic.
  ONNX_API OpSchema &SetNodeDeterminism(NodeDeterminism ndi) {
    node_determinism_ = ndi;
    return *this;
  }
  /// Returns the declared node determinism behavior.
  ONNX_API NodeDeterminism GetNodeDeterminism() const { return node_determinism_; }

  /// Returns numeric tensor type constraints for current IR.
  ONNX_API static const std::vector<std::string> &all_numeric_types();
  /// Returns tensor type constraints for current IR.
  ONNX_API static const std::vector<std::string> &all_tensor_types();

  // IR-version–specific type lists (adapted from onnx/defs/schema.h).
  ONNX_API static const std::vector<std::string> &all_tensor_types_ir4();
  ONNX_API static const std::vector<std::string> &all_tensor_types_ir9();
  ONNX_API static const std::vector<std::string> &all_tensor_types_ir10();
  ONNX_API static const std::vector<std::string> &all_tensor_types_ir11();
  ONNX_API static const std::vector<std::string> &all_tensor_types_ir12();
  ONNX_API static const std::vector<std::string> &all_tensor_types_ir13();
  ONNX_API static const std::vector<std::string> &all_non_complex_tensor_types_ir10();
  ONNX_API static const std::vector<std::string> &all_non_complex_tensor_types_ir11();
  ONNX_API static const std::vector<std::string> &all_non_complex_tensor_types_ir12();
  ONNX_API static const std::vector<std::string> &all_non_complex_tensor_types_ir13();
  ONNX_API static const std::vector<std::string> &all_non_string_tensor_types_ir13();
  ONNX_API static const std::vector<std::string> &all_float_types_ir4();
  ONNX_API static const std::vector<std::string> &all_float_types_ir9();
  ONNX_API static const std::vector<std::string> &all_float_types_ir10() {
    return all_float_types_ir9();
  }
  ONNX_API static const std::vector<std::string> &all_float_types_plus_Xint8_ir4();
  ONNX_API static const std::vector<std::string> &all_non_complex_numeric_types_plus_bool_ir4();
  ONNX_API static const std::vector<std::string> &all_numeric_types_ir4();
  ONNX_API static const std::vector<std::string> &all_numeric_types_ir9();
  ONNX_API static const std::vector<std::string> &all_numeric_types_ir10();
  ONNX_API static const std::vector<std::string> &all_numeric_types_ir11();
  ONNX_API static const std::vector<std::string> &all_numeric_types_ir12();
  ONNX_API static const std::vector<std::string> &all_numeric_types_ir13();
  ONNX_API static const std::vector<std::string> &all_tensor_sequence_types();
  ONNX_API static const std::vector<std::string> &all_tensor_sequence_types_ir4();
  ONNX_API static const std::vector<std::string> &all_tensor_sequence_types_ir9();
  ONNX_API static const std::vector<std::string> &all_tensor_sequence_types_ir10();
  ONNX_API static const std::vector<std::string> &all_tensor_sequence_types_ir11();
  ONNX_API static const std::vector<std::string> &all_tensor_sequence_types_ir12();
  ONNX_API static const std::vector<std::string> &all_tensor_sequence_types_ir13();
  ONNX_API static const std::vector<std::string> &all_optional_types();
  ONNX_API static const std::vector<std::string> &all_optional_types_ir4();
  ONNX_API static const std::vector<std::string> &all_optional_types_ir9();
  ONNX_API static const std::vector<std::string> &all_optional_types_ir10();
  ONNX_API static const std::vector<std::string> &all_optional_types_ir11();
  ONNX_API static const std::vector<std::string> &all_optional_types_ir12();
  ONNX_API static const std::vector<std::string> &all_optional_types_ir13();
  ONNX_API static const std::vector<std::string> &numeric_types_for_math_reduction();
  ONNX_API static const std::vector<std::string> &numeric_types_for_math_reduction_ir4();

private:
  void ParseAndSetTypes(std::vector<OpSchema::FormalParameter> *formal_parameters);
  std::string VerifyFailPrefix(std::string_view node_name) const;
  void VerifyInputNum(int input_num, std::string_view node_name = "") const;
  void VerifyOutputNum(int output_num, std::string_view node_name = "") const;

  std::string name_;
  std::string file_;
  std::string doc_;
  std::string domain_ = ONNX_DOMAIN;
  std::unordered_map<std::string, Attribute> attributes_;
  bool allows_unchecked_attributes_ = false;
  std::vector<FormalParameter> inputs_;
  std::vector<FormalParameter> outputs_;
  std::vector<TypeConstraintParam> type_constraint_params_;
  TypeConstraintMap type_constraints_;
  int line_ = 0;
  SupportType support_;
  int min_input_ = 0;
  int max_input_ = 0;
  int min_output_ = 0;
  int max_output_ = 0;
  OperatorSetVersion since_version_ = kUninitializedSinceVersion;
  bool deprecated_{};
  NodeDeterminism node_determinism_ = NodeDeterminism::Unknown;
  std::function<bool(int)> num_inputs_allowed_ = [](int) { return true; };
  std::function<bool(int)> num_outputs_allowed_ = [](int) { return true; };
  InferenceFunction tensor_inference_function_;
  DataPropagationFunction data_propagation_function_;
  std::map<int, std::shared_ptr<FunctionProto>> opset_version_to_function_body_;
  std::map<int, ContextDependentFunctionBodyBuilder> opset_version_to_function_builder_;
};

/// Registry map indexed by operator name, domain, and opset version.
using OpName_Domain_Version_Schema_Map =
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::map<OperatorSetVersion, OpSchema>>>;

/**
 * Abstract interface used to query operator schemas by name/domain/version.
 * This interface allows schema lookups from custom registry backends.
 */
class ISchemaRegistry {
public:
  virtual ~ISchemaRegistry() = default;
  ONNX_API virtual const OpSchema *GetSchema(const std::string &key, const int maxInclusiveVersion,
                                             const std::string &domain = ONNX_DOMAIN) const = 0;
};

/**
 * Global registry for ONNX operator schemas and their version ranges.
 */
class OpSchemaRegistry final : public ISchemaRegistry {
public:
  /**
   * Maintains per-domain minimum/maximum version metadata.
   */
  class DomainToVersionRange final {
  public:
    DomainToVersionRange();

    /// Returns the domain-to-min/max-version mapping.
    ONNX_API const std::unordered_map<std::string, std::pair<int, int>> &Map() const {
      return map_;
    }

    /// Returns the last-release-version mapping keyed by domain.
    ONNX_API const std::unordered_map<std::string, int> &LastReleaseVersionMap() const {
      return last_release_version_map_;
    }

    /// Adds a new domain version range entry.
    ONNX_API void AddDomainToVersion(const std::string &domain, int min_version, int max_version,
                                     int last_release_version = -1);
    /// Updates an existing domain version range entry.
    ONNX_API void UpdateDomainToVersion(const std::string &domain, int min_version, int max_version,
                                        int last_release_version = -1);

    /// Returns the process-wide singleton instance.
    ONNX_API static DomainToVersionRange &Instance();

  private:
    std::unordered_map<std::string, std::pair<int, int>> map_;
    std::unordered_map<std::string, int> last_release_version_map_;
    std::mutex mutex_;
  };

  /**
   * Registers a schema once during static initialization.
   */
  class OpSchemaRegisterOnce final {
  public:
    OpSchemaRegisterOnce(OpSchema op_schema, int opset_version_to_load = 0,
                         bool fail_duplicate_schema = true) {
      OpSchemaRegisterNoExcept(std::move(op_schema), opset_version_to_load, fail_duplicate_schema);
    }

    ONNX_API static void OpSchemaRegisterNoExcept(OpSchema &&op_schema,
                                                  int opset_version_to_load = 0,
                                                  bool fail_duplicate_schema = true);
    ONNX_API static void OpSchemaRegisterImpl(OpSchema &&op_schema, int opset_version_to_load = 0,
                                              bool fail_duplicate_schema = true);
  };

  /// Returns the process-wide schema registry singleton.
  ONNX_API static OpSchemaRegistry *Instance();

  /// Looks up the most recent schema not newer than maxInclusiveVersion.
  ONNX_API const OpSchema *GetSchema(const std::string &key, const int maxInclusiveVersion,
                                     const std::string &domain = ONNX_DOMAIN) const override;

  /// Looks up the most recent schema not newer than maxInclusiveVersion.
  ONNX_API static const OpSchema *Schema(const std::string &key, const int maxInclusiveVersion,
                                         const std::string &domain = ONNX_DOMAIN);

  /// Looks up the latest schema for an operator in a domain.
  ONNX_API static const OpSchema *Schema(const std::string &key,
                                         const std::string &domain = ONNX_DOMAIN);

  /// Returns all currently loaded schemas (latest versions only).
  ONNX_API static std::vector<OpSchema> get_all_schemas();
  /// Returns all currently loaded schemas including historical versions.
  ONNX_API static std::vector<OpSchema> get_all_schemas_with_history();

  /// Returns the mutable global schema map.
  ONNX_API static OpName_Domain_Version_Schema_Map &map();

  static int loaded_schema_version;

  ONNX_API static void SetLoadedSchemaVersion(int target_version) {
    loaded_schema_version = target_version;
  }

  ONNX_API static int GetLoadedSchemaVersion() { return loaded_schema_version; }

  // Removes all schemas registered under the given domain.
  ONNX_API static void OpSchemaDeregisterAll(const std::string &domain = ONNX_DOMAIN);

private:
  ONNX_API static std::mutex &RegistrationMutex();
  ONNX_API static void register_schemas();
};

/// Returns true when static schema registration has been disabled at build time.
ONNX_API bool IsOnnxStaticRegistrationDisabled();

/// Registers one schema using a copy.
void RegisterSchema(const OpSchema &schema, int opset_version_to_load = 0,
                    bool fail_duplicate_schema = true, bool fail_with_exception = false);
/// Registers one schema using move semantics.
void RegisterSchema(OpSchema &&schema, int opset_version_to_load = 0,
                    bool fail_duplicate_schema = true, bool fail_with_exception = false);
/// Removes one schema entry from the global registry.
void DeregisterSchema(const std::string &op_type, int version,
                      const std::string &domain = ONNX_DOMAIN);

// Registers all built-in ONNX operator schemas across all opset versions.
// Duplicate registrations are silently ignored so this function is safe to call
// more than once.
ONNX_API void RegisterAllOnnxOperatorSchemas();

// Registers the latest opset schema up to opset_version_to_load.
// By default opset_version_to_load=0 means all versions are registered.
template <class T>
inline void RegisterOpSetSchema(int opset_version_to_load = 0, bool fail_duplicate_schema = true) {
  T::ForEachSchema([opset_version_to_load, fail_duplicate_schema](OpSchema &&schema) {
    RegisterSchema(std::move(schema), opset_version_to_load, fail_duplicate_schema);
  });
}

// Forward declaration for the non-specialized GetOpSchema function.
// Specializations are generated by the ONNX_OPERATOR_SET_SCHEMA_EX macro.
template <typename T> ONNX_API OpSchema GetOpSchema();

} // namespace ONNX_LIGHT_NAMESPACE

// Naming convention for operator schema classes.
#define ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name) name##_##domain##_ver##ver

// Naming convention for preview operator schema classes.
#define ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(ver, name)                                     \
  ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxPreview, ver, name)

// Stub for debug-build operator-set tracking (no runtime counting).
// The argument order passed from ONNX_OPERATOR_SET_SCHEMA_EX is
// (domain, ver, name, ...) which maps onto the macro parameters below as
// (name=domain, domain=ver, ver=name, ...). This matches the upstream ONNX
// convention and still produces a unique variable name per operator.
#define ONNX_OPERATOR_SET_SCHEMA_DEBUG_VARIABLE(name, domain, ver, dbg_included_in_static_opset)   \
  static size_t dbg_count_check_##name##_##domain##_ver##ver [[maybe_unused]] = 0

// Defines a GetOpSchema<> specialisation and a forward-declared class for each
// operator schema registered via ONNX_OPERATOR_SET_SCHEMA_EX.
#define ONNX_OPERATOR_SET_SCHEMA_EX(name, domain, domain_str, ver, dbg_included_in_static_opset,   \
                                    impl)                                                          \
  class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name);                                    \
  template <>                                                                                      \
  ONNX_API ONNX_LIGHT_NAMESPACE::OpSchema                                                          \
  ONNX_LIGHT_NAMESPACE::GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name)>() {    \
    return impl.SetName(#name)                                                                     \
        .SetDomain(domain_str)                                                                     \
        .SinceVersion(ver)                                                                         \
        .SetLocation(__FILE__, __LINE__);                                                          \
  }                                                                                                \
  ONNX_OPERATOR_SET_SCHEMA_DEBUG_VARIABLE(domain, ver, name, dbg_included_in_static_opset)

#define ONNX_OPERATOR_SET_SCHEMA(name, ver, impl)                                                  \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, Onnx, ONNX_LIGHT_NAMESPACE::ONNX_DOMAIN, ver, true, impl)

#define ONNX_ML_OPERATOR_SET_SCHEMA(name, ver, impl)                                               \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, OnnxML, ONNX_LIGHT_NAMESPACE::AI_ONNX_ML_DOMAIN, ver, true,    \
                              impl)

#define ONNX_TRAINING_OPERATOR_SET_SCHEMA(name, ver, impl)                                         \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, OnnxTraining, ONNX_LIGHT_NAMESPACE::AI_ONNX_TRAINING_DOMAIN,   \
                              ver, true, impl)

#define ONNX_PREVIEW_OPERATOR_SET_SCHEMA(name, ver, impl)                                          \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, OnnxPreview, ONNX_LIGHT_NAMESPACE::AI_ONNX_PREVIEW_DOMAIN,     \
                              ver, true, impl)

#define ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(name, ver, impl)                                 \
  ONNX_OPERATOR_SET_SCHEMA_EX(                                                                     \
      name, OnnxPreview, ONNX_LIGHT_NAMESPACE::AI_ONNX_PREVIEW_TRAINING_DOMAIN, ver, true, impl)

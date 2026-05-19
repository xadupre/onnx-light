// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file checker.h
 * @brief Declares ONNX model and graph validation entry points.
 *
 * This header exposes the checker context and lexical scope helpers used to
 * validate ONNX protobuf structures, as well as the public check_model()
 * overloads used by the C++ API.
 */

#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_lib/defs/schema.h"
#include "onnx_lib/onnx-data.pb.h"
#include "onnx_lib/string_utils.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace checker {
// std::string member means copy may throw when allocation fails
// NOLINTNEXTLINE(bugprone-exception-copy-constructor-throws)
class ValidationError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
  const char *what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }
  void AppendContext(const std::string &context) {
    expanded_message_ =
        ONNX_LIGHT_NAMESPACE::MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

private:
  std::string expanded_message_;
};

#define fail_check(...)                                                                            \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::checker::ValidationError(                                    \
      ONNX_LIGHT_NAMESPACE::MakeString(__VA_ARGS__)))

/**
 * Stores checker configuration shared across recursive validation calls.
 */
class CheckerContext final {
public:
  int get_ir_version() const { return ir_version_; }
  void set_ir_version(int v) { ir_version_ = v; }
  const std::unordered_map<std::string, int> &get_opset_imports() const { return opset_imports_; }
  void set_opset_imports(std::unordered_map<std::string, int> imps) {
    opset_imports_ = std::move(imps);
  }
  bool is_main_graph() const { return is_main_graph_; }
  void set_is_main_graph(bool is_main_graph) { is_main_graph_ = is_main_graph; }

  void set_schema_registry(const ISchemaRegistry *schema_registry) {
    schema_registry_ = schema_registry;
  }

  const ISchemaRegistry *get_schema_registry() const { return schema_registry_; }

  void set_model_dir(const std::string &model_dir) { model_dir_ = model_dir; }

  std::string get_model_dir() const { return model_dir_; }

  bool skip_opset_compatibility_check() const { return skip_opset_compatibility_check_; }

  void set_skip_opset_compatibility_check(bool value) { skip_opset_compatibility_check_ = value; }

  bool check_custom_domain() const { return check_custom_domain_; }

  void set_check_custom_domain(bool value) { check_custom_domain_ = value; }

  explicit CheckerContext() = default;

private:
  int ir_version_{-1};
  std::unordered_map<std::string, int> opset_imports_;
  bool is_main_graph_ = true;
  const ISchemaRegistry *schema_registry_ = OpSchemaRegistry::Instance();
  std::string model_dir_;
  bool skip_opset_compatibility_check_ = false;
  bool check_custom_domain_ = false;
};

/**
 * Tracks values visible in the current graph and in parent lexical scopes.
 */
class LexicalScopeContext final {
public:
  LexicalScopeContext() = default;
  ~LexicalScopeContext() = default;

  // Construct an instance with the lexical scope from the parent graph to allow
  // lookup of names from that scope via this_or_ancestor_graph_has.
  // The caller must ensure parent_context remains valid for the entire lifetime
  // of the new instance. Alternatively, if that cannot be guaranteed, create an
  // instance with the default constructor and populate output_names with the
  // values from the parent scope so the values are copied instead.
  LexicalScopeContext(const LexicalScopeContext &parent_context)
      : parent_context_{&parent_context} {}
  LexicalScopeContext &operator=(const LexicalScopeContext &parent_context) {
    if (this == &parent_context) {
      return *this;
    }
    parent_context_ = &parent_context;
    return *this;
  }
  LexicalScopeContext(LexicalScopeContext &&) = delete;
  LexicalScopeContext &operator=(LexicalScopeContext &&) = delete;

  void add(const std::string &name) { output_names.insert(name); }

  bool this_graph_has(const std::string &name) const { return output_names.count(name) > 0; }

  bool this_or_ancestor_graph_has(const std::string &name) const {
    return this_graph_has(name) ||
           (parent_context_ && parent_context_->this_or_ancestor_graph_has(name));
  }

  // public for backwards compatibility. please prefer the public interface of
  // this class over directly changing output_names
  std::unordered_set<std::string> output_names;

private:
  const LexicalScopeContext *parent_context_{nullptr};
};

using IR_VERSION_TYPE = decltype(Version::IR_VERSION);
void check_value_info(const ValueInfoProto &value_info, const CheckerContext & /*ctx*/);
void check_tensor(const TensorProto &tensor, const CheckerContext & /*ctx*/);
void check_sparse_tensor(const SparseTensorProto &sparse_tensor, const CheckerContext & /*ctx*/);
void check_sequence(const SequenceProto &sequence, const CheckerContext & /*ctx*/);
void check_map(const MapProto &map, const CheckerContext & /*ctx*/);
void check_optional(const OptionalProto &opt, const CheckerContext & /*ctx*/);
void check_attribute(const AttributeProto &attr, const CheckerContext & /*ctx*/,
                     const LexicalScopeContext & /*lex_ctx*/);
void check_node(const NodeProto &node, const CheckerContext & /*ctx*/,
                const LexicalScopeContext & /*lex_ctx*/);
void check_graph(const GraphProto &graph, const CheckerContext & /*ctx*/,
                 const LexicalScopeContext & /*parent_lex*/);
void check_function(const FunctionProto &function, const CheckerContext & /*ctx*/,
                    const LexicalScopeContext & /*parent_lex*/);

/**
 * Determines whether a node remains schema-compatible across two opset
 * versions.
 *
 * Compatibility means that both imported versions resolve to the same schema
 * evolution point (the same since_version), so function-local and model-level
 * imports do not disagree for the node's operator.
 *
 * @param node Identifies the operator node to validate.
 * @param ctx Provides checker settings and schema lookup configuration.
 * @param func_opset_imports Contains opset imports from the enclosing function.
 * @param model_opset_imports Contains opset imports from the parent model.
 *
 * @throws ValidationError Thrown when compatibility checks fail.
 */
ONNX_API void
check_opset_compatibility(const NodeProto &node, const CheckerContext &ctx,
                          const std::unordered_map<std::string, int> &func_opset_imports,
                          const std::unordered_map<std::string, int> &model_opset_imports);

/**
 * Validates all model-local functions declared in a model.
 *
 * @param model Supplies the model containing local functions.
 * @param ctx Provides checker settings and schema lookup configuration.
 * @param parent_lex Provides the lexical scope visible to local functions.
 *
 * @throws ValidationError Thrown when function validation fails.
 */
ONNX_API void check_model_local_functions(const ModelProto &model, const CheckerContext &ctx,
                                          const LexicalScopeContext &parent_lex);

/**
 * Detects cycles in the model-local function call graph.
 *
 * @param model Supplies the model containing local functions.
 *
 * @throws ValidationError Thrown when a function directly or indirectly
 * references itself.
 */
ONNX_API void check_function_call_cycles(const ModelProto &model);

/**
 * Validates an in-memory model protobuf.
 *
 * @param model Model to validate.
 * @param full_check When true, enables additional shape inference checks after
 * structural validation.
 * @param skip_opset_compatibility_check When true, skips schema compatibility
 * checks.
 * @param check_custom_domain When true, enables checks on custom op domains.
 *
 * @throws ValidationError Thrown when validation fails.
 */
ONNX_API void check_model(const ModelProto &model, bool full_check = false,
                          bool skip_opset_compatibility_check = false,
                          bool check_custom_domain = false);
/**
 * Validates a serialized model located at a filesystem path.
 *
 * @param model_path UTF-8 path to a serialized ModelProto.
 * @param full_check When true, enables additional shape inference checks after
 * structural validation.
 * @param skip_opset_compatibility_check When true, skips schema compatibility
 * checks.
 * @param check_custom_domain When true, enables checks on custom op domains.
 *
 * @throws ValidationError Thrown when validation fails.
 */
ONNX_API void check_model(const std::string &model_path, bool full_check = false,
                          bool skip_opset_compatibility_check = false,
                          bool check_custom_domain = false);

/**
 * Resolves and validates an external tensor data location relative to a model.
 *
 * @param base_dir Provides the model base directory used for resolution.
 * @param location Provides the external data location from TensorProto.
 * @param tensor_name Provides the tensor name used in error messages.
 *
 * @return The resolved filesystem path.
 *
 * @throws ValidationError Thrown when the location is invalid or unsafe.
 */
std::filesystem::path resolve_external_data_location(const std::string &base_dir,
                                                     const std::string &location,
                                                     const std::string &tensor_name);

/**
 * Opens external tensor data and returns a CRT file descriptor.
 *
 * @param base_dir Provides the model base directory used for resolution.
 * @param location Provides the external data location from TensorProto.
 * @param tensor_name Provides the tensor name used in error messages.
 * @param read_only Selects read-only mode when true.
 *
 * @return The opened CRT file descriptor.
 *
 * @throws ValidationError Thrown when location validation fails.
 * @throws std::runtime_error Thrown when the file cannot be opened.
 *
 * The caller owns the descriptor and must close it.
 */
int64_t open_external_data(const std::string &base_dir, const std::string &location,
                           const std::string &tensor_name, bool read_only);

/**
 * Determines whether a node belongs to an experimental domain.
 *
 * @param node Identifies the operator node to inspect.
 *
 * @return True when the node is experimental.
 */
ONNX_API bool check_is_experimental_op(const NodeProto &node);

} // namespace checker
} // namespace ONNX_LIGHT_NAMESPACE

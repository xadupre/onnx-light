// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file light_op_schema.h
 * @brief Declares the lightweight ONNX operator schema types used by
 *        ``onnx_light``.
 *
 * This header defines the core data structures that ``onnx_light`` uses to
 * describe ONNX operators without depending on the full ``onnx`` library:
 *
 * - ::ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema, a read-only record that
 *   captures a single operator at a specific opset version (name, domain,
 *   ``since_version``, documentation string, formal inputs and outputs, and
 *   type constraints).
 * - ::ONNX_LIGHT_NAMESPACE::core::schema::FormalParameter and
 *   ::ONNX_LIGHT_NAMESPACE::core::schema::TypeConstraintParam, the building blocks
 *   used to describe input/output parameters and their type constraints.
 * - ::ONNX_LIGHT_NAMESPACE::core::schema::TensorType, an enumeration of every
 *   element-tensor, sequence-tensor, and optional-tensor type used in type
 *   constraints, together with ::ONNX_LIGHT_NAMESPACE::core::schema::ToTypeString
 *   to convert it to the canonical ONNX type string (e.g. ``"tensor(float)"``).
 * - A collection of helper functions returning common type sets reused across
 *   operator schemas (``FloatTypes()``, ``AllNumericTypes()``,
 *   ``AllTensorTypes()``, ``CastTypesVer*()``, ``EqualTypesV*()``, etc.).
 * - ::ONNX_LIGHT_NAMESPACE::core::schema::StripDocs to obtain a memory-light copy
 *   of a schema list with documentation strings cleared, useful in
 *   memory-constrained environments.
 *
 * The schemas produced by the helpers in the sibling ``operator_sets_*.h``
 * headers are aggregated by ``operator_sets.h`` via
 * ``GetAllOnnxOpSchemasWithHistory()`` and consumed by both documentation
 * generators and the ``onnx_shapes`` shape inference library.
 *
 * Constructing a schema with invalid arguments throws a
 * ::ONNX_LIGHT_NAMESPACE::core::schema::SchemaError.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// ``ONNX_LIGHT_NAMESPACE`` must resolve to the ``onnx_light`` namespace (or its
// compile-time override) so that ``LightOpSchema`` has a single C++ identity
// shared with the rest of the ``onnx_core`` / ``onnx_op`` libraries. Including
// the helpers header defines the macro when no translation unit has done so yet
// while still honouring an explicit ``-DONNX_LIGHT_NAMESPACE=...`` override.
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_helpers.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/type_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::schema {

/// The standard ONNX operator domain string.
constexpr const char *kOnnxDomain = "ai.onnx";

/// Describes a single formal input or output parameter of an ONNX operator.
struct FormalParameter {
  /// Parameter name as it appears in the ONNX spec.
  std::string name;
  /// Human-readable description of the parameter.
  std::string description;
  /// Type-constraint identifier string (e.g. "T", "T1").
  std::string type;
};

/**
 * Enumeration of attribute scalar/list types supported by ONNX.
 *
 * The enumerator values mirror ``onnx::AttributeProto::AttributeType`` so an
 * ``AttributeType`` from ``onnx_op`` can be compared against (or converted
 * to) the proto enum without a lookup table. The enumeration is duplicated
 * here so that ``onnx_op`` remains free of any dependency on the full ONNX
 * schema registry.
 */
enum class AttributeType : int32_t {
  UNDEFINED = 0,
  FLOAT = 1,
  INT = 2,
  STRING = 3,
  TENSOR = 4,
  GRAPH = 5,
  FLOATS = 6,
  INTS = 7,
  STRINGS = 8,
  TENSORS = 9,
  GRAPHS = 10,
  SPARSE_TENSOR = 11,
  SPARSE_TENSORS = 12,
  TYPE_PROTO = 13,
  TYPE_PROTOS = 14,
};

/// Returns the canonical ONNX name for an ``AttributeType`` (e.g. ``"INTS"``).
inline constexpr const char *AttributeType_Name(AttributeType t) {
  switch (t) {
  case AttributeType::FLOAT:
    return "FLOAT";
  case AttributeType::INT:
    return "INT";
  case AttributeType::STRING:
    return "STRING";
  case AttributeType::TENSOR:
    return "TENSOR";
  case AttributeType::GRAPH:
    return "GRAPH";
  case AttributeType::FLOATS:
    return "FLOATS";
  case AttributeType::INTS:
    return "INTS";
  case AttributeType::STRINGS:
    return "STRINGS";
  case AttributeType::TENSORS:
    return "TENSORS";
  case AttributeType::GRAPHS:
    return "GRAPHS";
  case AttributeType::SPARSE_TENSOR:
    return "SPARSE_TENSOR";
  case AttributeType::SPARSE_TENSORS:
    return "SPARSE_TENSORS";
  case AttributeType::TYPE_PROTO:
    return "TYPE_PROTO";
  case AttributeType::TYPE_PROTOS:
    return "TYPE_PROTOS";
  case AttributeType::UNDEFINED:
  default:
    return "UNDEFINED";
  }
}

/**
 * Typed default value carried by an :class:`AttributeParam`.
 *
 * Mirrors the subset of ``onnx::AttributeProto`` value fields that can sensibly
 * be expressed as a literal default in an operator schema:
 *
 * - ``std::monostate`` &mdash; no default value (the attribute is required or
 *   has no documented default).
 * - ``int64_t`` &mdash; default for ``AttributeType::INT`` (also used for
 *   boolean-valued ``INT`` attributes; ``0``/``1``).
 * - ``double`` &mdash; default for ``AttributeType::FLOAT``.
 * - ``std::string`` &mdash; default for ``AttributeType::STRING``.
 * - ``std::vector<int64_t>``/``std::vector<double>``/``std::vector<std::string>``
 *   &mdash; defaults for ``INTS``/``FLOATS``/``STRINGS``.
 *
 * ``TENSOR``/``GRAPH``/``SPARSE_TENSOR``/``TYPE_PROTO`` attributes have no
 * literal default in practice and are therefore represented as
 * ``std::monostate``.
 */
using AttributeDefault =
    std::variant<std::monostate, int64_t, double, std::string, std::vector<int64_t>,
                 std::vector<double>, std::vector<std::string>>;

/// Returns a stable textual representation of an ``AttributeDefault`` (e.g.
/// ``"1"``, ``"0.5"``, ``"foo"``, ``"[1, 2, 3]"``, or ``""`` for monostate).
std::string AttributeDefaultRepr(const AttributeDefault &d);

/**
 * Describes a single operator attribute as exposed by LightOpSchema.
 *
 * Attribute metadata is intentionally lightweight to keep ``onnx_op`` free of
 * any dependency on the full ONNX schema registry. The ``type`` field uses
 * the ``AttributeType`` enumeration above; ``default_value`` is a typed
 * variant (see :type:`AttributeDefault`) and is ``std::monostate`` when the
 * attribute is required or has no documented default.
 */
struct AttributeParam {
  /// Attribute name as it appears in the ONNX spec.
  std::string name;
  /// Human-readable description of the attribute.
  std::string description;
  /// Attribute type (mirrors ``onnx::AttributeProto::AttributeType``).
  AttributeType type;
  /// True if the attribute is required (no default value).
  bool required;
  /// Typed default value (``std::monostate`` when required or absent).
  AttributeDefault default_value = {};
};

/// Re-exports ``onnx_proto::TensorType`` so that existing consumers of
/// ``core::schema::TensorType`` continue to compile without change.
using TensorType = onnx_proto::TensorType;

/// Specifies which tensor types are permitted for a named type parameter.
struct TypeConstraintParam {
  /// Type-parameter identifier (e.g. "T").
  std::string type_param_str;
  /// Set of tensor types that satisfy this constraint.
  std::vector<TensorType> allowed_type_strs;
  /// Human-readable description of the constraint.
  std::string description;
};

/**
 * Returns the ONNX type-string representation of a TensorType value.
 *
 * Forwards to ``onnx_proto::ToTypeString``; kept here so that existing
 * callers of ``core::schema::ToTypeString`` continue to compile unchanged.
 *
 * @param type Tensor type enumerator to convert.
 * @return Null-terminated string such as `"tensor(float)"` or
 *         `"seq(tensor(int64))"`.
 */
inline constexpr const char *ToTypeString(TensorType type) {
  return onnx_proto::ToTypeString(type);
}

/// Thrown when a LightOpSchema is constructed with invalid arguments, or when
/// ::ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema::Verify rejects a node.
class ONNX_LIGHT_CORE_API SchemaError final : public std::runtime_error {
public:
  /// Constructs a SchemaError with the given diagnostic message.
  explicit SchemaError(const std::string &message) : std::runtime_error(message) {}
};

/**
 * One concrete value supplied for a node input when calling
 * ::ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema::Verify, used to check the input
 * against the schema's type constraints. Exactly one alternative describes a given input:
 *
 * - ``ValueInfoProto`` &mdash; type/shape as it would appear in a ``GraphProto`` value_info
 *   entry. Tensor, sequence, and optional value_info types are all supported.
 * - ``core::symbolic::SymTensor`` &mdash; a single tensor descriptor, as used by the
 *   ``onnx_shapes``/``onnx_optim`` optimisation stack.
 * - ``core::symbolic::SymSequence`` &mdash; a tensor-sequence descriptor, as used by the
 *   same optimisation stack (e.g. the output of ``SequenceConstruct``).
 */
using SchemaInputValue = std::variant<ValueInfoProto, symbolic::SymTensor, symbolic::SymSequence>;

/**
 * Lightweight, read-only description of an ONNX operator schema at one
 * specific opset version.
 *
 * A LightOpSchema captures everything that documentation and validation tools
 * need about a single versioned operator: its name, domain, the opset version
 * it was introduced in, its documentation string, the formal inputs and
 * outputs, and the type constraints that govern them.
 */
class LightOpSchema {
public:
  /**
   * Describes whether evaluating the operator produces deterministic outputs.
   *
   * Mirrors ``OpSchema::NodeDeterminism`` from the full ONNX library.
   */
  enum class NodeDeterminism : uint8_t {
    /// Determinism has not been specified for this operator.
    Unknown = 0,
    /// The operator may produce different outputs for identical inputs.
    NonDeterministic = 1,
    /// The operator always produces the same outputs for identical inputs.
    Deterministic = 2,
  };

  /**
   * Constructs a schema record for a versioned ONNX operator.
   *
   * @param name Operator name (e.g. "Add").
   * @param domain Operator domain (e.g. "ai.onnx").
   * @param since_version Opset version at which this schema was introduced.
   * @param doc Documentation string (may contain Markdown).
   * @param inputs Ordered list of formal input parameters.
   * @param outputs Ordered list of formal output parameters.
   * @param type_constraints Type constraints referenced by the parameters.
   * @param has_function_implementation Whether the op has a function body.
   * @param init_doc If true (default), the documentation string is stored on
   *        the schema. When false, the @p doc argument is discarded and
   *        doc() returns an empty string. This can be used to save memory
   *        when documentation is not needed by the consumer.
   */
  LightOpSchema(std::string name, std::string domain, int since_version, std::string doc,
                std::vector<FormalParameter> inputs, std::vector<FormalParameter> outputs,
                std::vector<TypeConstraintParam> type_constraints,
                bool has_function_implementation = false, bool init_doc = true)
      : name_(std::move(name)), domain_(std::move(domain)), since_version_(since_version),
        doc_(init_doc ? std::move(doc) : std::string()), inputs_(std::move(inputs)),
        outputs_(std::move(outputs)), type_constraints_(std::move(type_constraints)), attributes_(),
        has_function_implementation_(has_function_implementation),
        min_output_(static_cast<int>(outputs_.size())),
        max_output_(static_cast<int>(outputs_.size())), deprecated_(false) {}

  /**
   * Constructs a schema record for a versioned ONNX operator with attributes.
   *
   * Same as the other constructor but also stores the operator's attribute
   * metadata, which the documentation generator uses to surface
   * cross-version attribute differences.
   */
  LightOpSchema(std::string name, std::string domain, int since_version, std::string doc,
                std::vector<FormalParameter> inputs, std::vector<FormalParameter> outputs,
                std::vector<TypeConstraintParam> type_constraints,
                std::vector<AttributeParam> attributes, bool has_function_implementation = false,
                bool init_doc = true)
      : name_(std::move(name)), domain_(std::move(domain)), since_version_(since_version),
        doc_(init_doc ? std::move(doc) : std::string()), inputs_(std::move(inputs)),
        outputs_(std::move(outputs)), type_constraints_(std::move(type_constraints)),
        attributes_(std::move(attributes)),
        has_function_implementation_(has_function_implementation),
        min_output_(static_cast<int>(outputs_.size())),
        max_output_(static_cast<int>(outputs_.size())), deprecated_(false) {}

  /// Returns the operator name.
  const std::string &name() const { return name_; }
  /// Returns the operator domain.
  const std::string &domain() const { return domain_; }
  /// Returns the opset version at which this schema was introduced.
  int since_version() const { return since_version_; }
  /// Returns the operator documentation string.
  const std::string &doc() const { return doc_; }
  /// Returns the list of formal input parameters.
  const std::vector<FormalParameter> &inputs() const { return inputs_; }
  /// Returns the list of formal output parameters.
  const std::vector<FormalParameter> &outputs() const { return outputs_; }
  /// Returns the type constraints for this schema.
  const std::vector<TypeConstraintParam> &type_constraints() const { return type_constraints_; }
  /// Returns the operator attributes (may be empty when not populated).
  const std::vector<AttributeParam> &attributes() const { return attributes_; }
  /// Returns true if the operator has a function body implementation.
  bool has_function_implementation() const { return has_function_implementation_; }

  /// Returns the minimum number of outputs supported by this operator.
  /// Defaults to ``outputs().size()``; can be overridden via
  /// ``set_min_output`` for operators with variadic outputs.
  int min_output() const { return min_output_; }
  /// Returns the maximum number of outputs supported by this operator.
  /// Defaults to ``outputs().size()``; can be overridden via
  /// ``set_max_output`` for operators with variadic outputs (use
  /// ``std::numeric_limits<int>::max()`` for unbounded variadic outputs).
  int max_output() const { return max_output_; }
  /// Returns true if this versioned operator is deprecated.
  bool deprecated() const { return deprecated_; }
  /// Returns the operator's node determinism (``Unknown`` when unspecified).
  NodeDeterminism node_determinism() const { return node_determinism_; }
  /// Returns true if the operator is explicitly marked non-deterministic.
  bool non_deterministic() const { return node_determinism_ == NodeDeterminism::NonDeterministic; }

  /// Sets the minimum number of outputs. Returns *this for chaining.
  LightOpSchema &set_min_output(int v) {
    min_output_ = v;
    return *this;
  }
  /// Sets the maximum number of outputs. Returns *this for chaining.
  LightOpSchema &set_max_output(int v) {
    max_output_ = v;
    return *this;
  }
  /// Marks this operator as deprecated. Returns *this for chaining.
  LightOpSchema &set_deprecated(bool v = true) {
    deprecated_ = v;
    return *this;
  }
  /// Sets the operator's node determinism. Returns *this for chaining.
  LightOpSchema &set_node_determinism(NodeDeterminism v) {
    node_determinism_ = v;
    return *this;
  }

  /**
   * Verifies that @p node is a valid instantiation of this operator schema, throwing
   * ::ONNX_LIGHT_NAMESPACE::core::schema::SchemaError on the first violation found.
   *
   * The checks performed are, in order:
   *   - @p node is not deprecated (`deprecated()` is false).
   *   - `node.op_type()` matches `name()`, and `node.domain()` matches `domain()` (an empty
   *     node domain is treated as ::ONNX_LIGHT_NAMESPACE::core::schema::kOnnxDomain).
   *   - `node.output_size()` lies within [`min_output()`, `max_output()`].
   *   - Every attribute in `node.attribute()` is recognized (declared in `attributes()`, or its
   *     name starts with `"__"`, an internal-symbol convention that is always accepted), has a
   *     `type()` matching the declared ::ONNX_LIGHT_NAMESPACE::core::schema::AttributeType, and
   *     every required attribute (`AttributeParam::required`) is present.
   *   - When @p inputs is non-null, each populated (non-``std::nullopt``) entry is resolved to a
   *     ::ONNX_LIGHT_NAMESPACE::onnx_proto::TensorType and, if the corresponding formal input's
   *     `type` names one of `type_constraints()`, checked for membership in that constraint's
   *     `allowed_type_strs`. Entries beyond `inputs().size()` are checked against the last formal
   *     input (variadic convention), and entries whose concrete type cannot be determined (e.g. a
   *     ``ValueInfoProto`` with a `map_type` or `opaque_type`) are silently skipped, since
   *     LightOpSchema does not model every ONNX type category.
   *
   * Input arity itself is intentionally not bounded here: unlike outputs, LightOpSchema does not
   * track per-parameter Optional/Variadic metadata for inputs, so @p node may declare fewer or
   * more inputs than `inputs().size()` without being rejected.
   *
   * @param node Node to verify against this schema. `node.op_type()`/`node.domain()` are expected
   *        to already match this schema (see above); callers are responsible for having selected
   *        the schema version matching the node's opset import.
   * @param inputs Optional, one entry per node input, describing its concrete type for
   *        type-constraint checking. When null (the default), no input type checking is
   *        performed.
   */
  void Verify(const NodeProto &node,
              const std::vector<std::optional<SchemaInputValue>> *inputs = nullptr) const;

private:
  std::string name_;
  std::string domain_;
  int since_version_;
  std::string doc_;
  std::vector<FormalParameter> inputs_;
  std::vector<FormalParameter> outputs_;
  std::vector<TypeConstraintParam> type_constraints_;
  std::vector<AttributeParam> attributes_;
  bool has_function_implementation_;
  int min_output_;
  int max_output_;
  bool deprecated_;
  NodeDeterminism node_determinism_ = NodeDeterminism::Unknown;
};

/// Returns floating-point tensor types (float16, float, double, bfloat16).
std::vector<TensorType> FloatTypes();
/// Returns numeric types used in reduction ops (excludes low-precision floats).
std::vector<TensorType> NumericTypesForMathReduction();
/// Returns numeric types used in reduction ops for IR version 4 and later.
std::vector<TensorType> NumericTypesForMathReductionIr4();
/// Returns all numeric (integer and floating-point) tensor types.
std::vector<TensorType> AllNumericTypes();
/// Returns all numeric tensor types for IR version 4 and later.
std::vector<TensorType> AllNumericTypesIr4();
/// Returns all scalar tensor types (no sequence types).
std::vector<TensorType> AllTensorTypes();
/// Returns all sequence-of-tensor types.
std::vector<TensorType> AllTensorSequenceTypes();
/// Returns all optional tensor and optional sequence tensor types.
/// Matches `OpSchema::all_optional_types()` from the full ONNX library.
std::vector<TensorType> AllOptionalTypes();
/// Returns the Cast input/output types valid for opset versions 1 and 6.
std::vector<TensorType> CastTypesVer1And6();
/// Returns the Cast input/output types valid from opset version 9.
std::vector<TensorType> CastTypesVer9();
/// Returns the Cast input/output types valid from opset version 13.
std::vector<TensorType> CastTypesVer13();
/// Returns the Cast input/output types valid from opset version 19.
std::vector<TensorType> CastTypesVer19();
/// Returns the Cast input/output types valid from opset version 21.
std::vector<TensorType> CastTypesVer21();
/// Returns the Cast input/output types valid from opset version 23.
std::vector<TensorType> CastTypesVer23();
/// Returns the Cast input/output types valid from opset version 24.
std::vector<TensorType> CastTypesVer24();
/// Returns the Cast input/output types valid from opset version 25.
std::vector<TensorType> CastTypesVer25();
std::vector<TensorType> CastTypesVer28();
/// Returns the Equal input types valid for opset versions 1 and 7.
std::vector<TensorType> EqualTypesV1V7();
/// Returns the Equal input types valid from opset version 11.
std::vector<TensorType> EqualTypesV11();
/// Returns the Equal input types valid from opset version 13.
std::vector<TensorType> EqualTypesV13();
/// Returns the Equal input types valid from opset version 19.
std::vector<TensorType> EqualTypesV19();
/// Returns the Concat input/output types valid for opset version 1
/// (float16, float, double).
std::vector<TensorType> ConcatTypesVer1();
/// Returns the Concat input/output types valid for opset versions 4 and 11
/// (matches `OpSchema::all_tensor_types()` from the full ONNX library).
std::vector<TensorType> ConcatTypesVer4And11();
/// Returns the Concat input/output types valid from opset version 13
/// (matches `OpSchema::all_tensor_types_ir4()` from the full ONNX library).
std::vector<TensorType> ConcatTypesVer13();

/**
 * Returns a copy of @p schemas with all documentation strings replaced by an
 * empty string. Useful when callers want the schema metadata but do not need
 * documentation, saving memory in memory-constrained environments.
 */
std::vector<LightOpSchema> StripDocs(const std::vector<LightOpSchema> &schemas);

/**
 * Type of a builder function that produces the versioned schema history for a
 * single ONNX operator. Each per-domain ``GetAllOnnxOp*SchemasWithHistory``
 * function maintains a static ``std::map<std::string, SchemaBuilder>`` keyed by
 * ``op_type`` so that, when a caller requests a specific operator, only the
 * matching builder runs instead of constructing the entire domain. This mirrors
 * the dispatch-table pattern used by the shape-inference subsystem.
 */
using SchemaBuilder = std::function<std::vector<LightOpSchema>()>;

/**
 * Invokes builders from a name → @ref SchemaBuilder map and concatenates their
 * results. When @p op_type is empty, every builder is invoked (in alphabetical
 * order of keys when @p builders is a ``std::map``). When @p op_type is
 * non-empty, only the builder whose key equals @p op_type is invoked (returning
 * an empty vector if absent). If @p init_doc is false, the resulting schemas
 * have their documentation strings stripped (see ::StripDocs).
 */
std::vector<LightOpSchema>
CollectSchemasFromBuilders(const std::map<std::string, SchemaBuilder> &builders,
                           const std::string &op_type, bool init_doc);

} // namespace ONNX_LIGHT_NAMESPACE::core::schema

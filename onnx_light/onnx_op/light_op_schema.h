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
 * - ::ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema, a read-only record that
 *   captures a single operator at a specific opset version (name, domain,
 *   ``since_version``, documentation string, formal inputs and outputs, and
 *   type constraints).
 * - ::ONNX_LIGHT_NAMESPACE::onnx_op::FormalParameter and
 *   ::ONNX_LIGHT_NAMESPACE::onnx_op::TypeConstraintParam, the building blocks
 *   used to describe input/output parameters and their type constraints.
 * - ::ONNX_LIGHT_NAMESPACE::onnx_op::TensorType, an enumeration of every
 *   element-tensor, sequence-tensor, and optional-tensor type used in type
 *   constraints, together with ::ONNX_LIGHT_NAMESPACE::onnx_op::ToTypeString
 *   to convert it to the canonical ONNX type string (e.g. ``"tensor(float)"``).
 * - A collection of helper functions returning common type sets reused across
 *   operator schemas (``FloatTypes()``, ``AllNumericTypes()``,
 *   ``AllTensorTypes()``, ``CastTypesVer*()``, ``EqualTypesV*()``, etc.).
 * - ::ONNX_LIGHT_NAMESPACE::onnx_op::StripDocs to obtain a memory-light copy
 *   of a schema list with documentation strings cleared, useful in
 *   memory-constrained environments.
 *
 * The schemas produced by the helpers in the sibling ``operator_sets_*.h``
 * headers are aggregated by ``operator_sets.h`` via
 * ``GetAllOnnxOpSchemasWithHistory()`` and consumed by both documentation
 * generators and the ``onnx_optim`` shape inference library.
 *
 * Constructing a schema with invalid arguments throws a
 * ::ONNX_LIGHT_NAMESPACE::onnx_op::SchemaError.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

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
 * Identifies an element or sequence tensor type supported by onnx-light.
 *
 * Each enumerator corresponds to a concrete ONNX element type or to a
 * sequence-of-tensor type used in type-constraint definitions.
 */
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
  kOptSeqBool,
  kOptSeqString,
  kOptSeqUint8,
  kOptSeqUint16,
  kOptSeqUint32,
  kOptSeqUint64,
  kOptSeqInt8,
  kOptSeqInt16,
  kOptSeqInt32,
  kOptSeqInt64,
  kOptSeqFloat16,
  kOptSeqFloat,
  kOptSeqDouble,
  kOptSeqComplex64,
  kOptSeqComplex128,
  kOptBool,
  kOptString,
  kOptUint8,
  kOptUint16,
  kOptUint32,
  kOptUint64,
  kOptInt8,
  kOptInt16,
  kOptInt32,
  kOptInt64,
  kOptFloat16,
  kOptFloat,
  kOptDouble,
  kOptComplex64,
  kOptComplex128,
  kUndefined,
};

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
 * @param type Tensor type enumerator to convert.
 * @return Null-terminated string such as `"tensor(float)"` or
 *         `"seq(tensor(int64))"`.
 */
const char *ToTypeString(TensorType type);

/// Thrown when a LightOpSchema is constructed with invalid arguments.
class SchemaError final : public std::runtime_error {
public:
  /// Constructs a SchemaError with the given diagnostic message.
  explicit SchemaError(const std::string &message) : std::runtime_error(message) {}
};

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
        outputs_(std::move(outputs)), type_constraints_(std::move(type_constraints)),
        has_function_implementation_(has_function_implementation) {}

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
  /// Returns true if the operator has a function body implementation.
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

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

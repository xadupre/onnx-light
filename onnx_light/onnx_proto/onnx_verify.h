#pragma once

#include "onnx.h"
#include "onnx_helper.h"

#include <cstddef>
#include <string>
#include <unordered_set>

/**
 * @file onnx_verify.h
 * @brief Schema-free structural validation for onnx_proto messages.
 *
 * The functions declared here validate that a protobuf is internally
 * consistent on its own terms: required fields are set, names are unique,
 * the graph is in single static assignment (SSA) form and topologically
 * sorted, tensor payload sizes match the declared shape/dtype, and so on.
 * None of these checks needs an operator-schema registry: nothing here
 * depends on the semantics of a specific ``op_type``.
 *
 * For schema-aware validation (operator input/output arity, attribute
 * constraints, type/shape inference), use ``onnx_lib::checker::check_model()``
 * instead (declared in ``onnx_lib/checker.h``).
 *
 * Every function below throws ``std::invalid_argument`` (via
 * ``EXT_THROW_INVALID`` / ``EXT_ENFORCE_INVALID``) with a descriptive message
 * on the first violation found.
 */

namespace ONNX_LIGHT_NAMESPACE {

/** Returns true when @p data_type is one of the supported affine storage types. */
inline constexpr bool IsAffineStorageType(TensorProto::DataType data_type) {
  return data_type == TensorProto::INT8 || data_type == TensorProto::UINT8 ||
         data_type == TensorProto::INT4 || data_type == TensorProto::UINT4;
}

/** Describes the validated payload geometry of one ``EncodedValueProto``. */
struct EncodedValueLayout {
  /** Resolved structured root, or null for the affine branch. */
  const StructTypeProto *root = nullptr;
  /** Affine descriptor, or null for the structured branch. */
  const AffineLayoutProto *affine = nullptr;
  /** Size of one record in bits: the element size, or the storage width for the affine branch. */
  uint64_t element_bits = 0;
  /** Byte extent of the payload, inline or declared by the external metadata. */
  uint64_t payload_bytes = 0;
  /** Number of records derived from the payload extent, or logical elements for affine. */
  uint64_t record_count = 0;
  /** True when the payload lives outside the message. */
  bool external = false;
  /** True when a ``type_ref`` was left unresolved because no catalogue was available. */
  bool unresolved_reference = false;
  /**
   * True when the payload bytes themselves were inspected, which only an inline
   * ``raw_data`` payload allows. For an external payload only the metadata is
   * validated: ``payload_bytes`` is the *declared* extent, the backing file is
   * neither opened nor measured, and content rules that need the bytes (the
   * affine 4-bit nibble padding in particular) are not checked. Load the payload
   * and re-validate before treating an external value as fully checked.
   */
  bool content_verified = false;
};

/**
 * Maximum nesting depth accepted while walking a declaration. Every nested
 * ``TypeProto`` and every ``type_ref`` expansion counts as one level, so a
 * hostile or accidentally recursive catalogue cannot exhaust the stack.
 */
inline constexpr uint32_t kMaxStructTypeDepth = 64;

/**
 * Resolves ``type_id`` references declared by ``ModelProto.struct_types``.
 *
 * The catalogue borrows the model it was built from: it must not outlive that
 * model, and the model must not be mutated while it is in use. A catalogue
 * holds a handful of declarations, so lookups scan them linearly instead of
 * paying for a hash map.
 */
class ONNX_LIGHT_PROTO_API StructTypeCatalogue {
public:
  /** Builds an empty catalogue resolving no identity. */
  StructTypeCatalogue() = default;

  /**
   * Indexes and validates every declaration of @p model.
   *
   * Each entry must carry a nonzero ``type_id`` unique in the model and a
   * concrete kind; references must resolve and must not form a cycle.
   *
   * @param model Model owning the declarations, borrowed by the catalogue.
   *
   * @throws std::invalid_argument Thrown when a declaration is malformed.
   */
  void Build(const ModelProto &model);

  /** Returns the declaration for @p type_id, or null when the model declares no such identity. */
  inline const StructTypeProto *Find(uint64_t type_id) const {
    if (model_ == nullptr || type_id == 0) {
      return nullptr;
    }
    const utils::RepeatedProtoField<StructTypeProto> &declarations = model_->ref_struct_types();
    for (size_t i = 0; i < declarations.size(); ++i) {
      if (declarations[i].has_type_id() && declarations[i].ref_type_id() == type_id) {
        return &declarations[i];
      }
    }
    return nullptr;
  }

  /** Returns the number of indexed declarations. */
  inline size_t size() const { return model_ == nullptr ? 0 : model_->ref_struct_types().size(); }

  /** Returns true when the catalogue indexes no declaration. */
  inline bool empty() const { return size() == 0; }

  /**
   * Returns the declaration @p type denotes, or @p type itself when it is not a reference.
   *
   * @throws std::invalid_argument Thrown when the reference does not resolve.
   */
  inline const StructTypeProto &Resolve(const StructTypeProto &type) const {
    if (type.kind_case() != StructTypeProto::kTypeRef) {
      return type;
    }
    const StructTypeProto *declaration = Find(type.ref_type_ref());
    EXT_ENFORCE_INVALID(declaration != nullptr,
                        "StructTypeProto references a type_id the model does not declare.");
    return *declaration;
  }

  /**
   * Computes the fixed size of @p type in bits with checked ``uint64_t`` arithmetic.
   *
   * Constants contribute zero bits, arrays and bit packings are tight and
   * fields follow declaration order.
   *
   * @param type Type whose layout is measured.
   * @param bits Receives the size in bits on success.
   * @param reason Optional destination for a human-readable failure cause.
   *
   * Returns: True when the type has a fixed inline layout, false otherwise.
   */
  bool FixedBitSize(const StructTypeProto &type, uint64_t &bits,
                    std::string *reason = nullptr) const;

  /**
   * Returns true when @p type can be the root of a byte-encoded payload, that
   * is when its size is fixed, strictly positive and a whole number of bytes.
   *
   * @param type Type to test.
   * @param bytes Receives the size of one record in bytes on success.
   * @param reason Optional destination for a human-readable failure cause.
   */
  inline bool IsByteEncodable(const StructTypeProto &type, uint64_t &bytes,
                              std::string *reason = nullptr) const {
    uint64_t bits = 0;
    if (!FixedBitSize(type, bits, reason)) {
      return false;
    }
    if (bits == 0 || bits % 8 != 0) {
      if (reason != nullptr) {
        *reason = "the encoded root needs a strictly positive size divisible by eight";
      }
      return false;
    }
    bytes = bits / 8;
    return true;
  }

  /**
   * Validates a ``TypeProto``, including any nested ``struct_type``.
   *
   * Dynamic tensor dimensions, sequences, maps and optional values stay valid
   * types here; only byte encoding requires a fixed layout.
   *
   * @param type Type to validate.
   *
   * @throws std::invalid_argument Thrown when validation fails.
   */
  void ValidateType(const TypeProto &type) const;

  /**
   * Validates one ``EncodedValueProto`` and returns its payload geometry.
   *
   * Checks the layout choice, raw/external exclusivity, external metadata,
   * byte-encodability of the structured root and payload divisibility, or the
   * frozen affine contract (storage type, parameter types, axis and block
   * size, logical type and shape, code byte length and nibble padding).
   *
   * @param value Value to validate.
   * @param require_resolved_reference When false, a ``type_ref`` that this
   * catalogue cannot resolve suspends the size checks instead of failing; used
   * to validate a graph before its model catalogue is known.
   *
   * Returns: The validated layout.
   *
   * @throws std::invalid_argument Thrown when validation fails.
   */
  EncodedValueLayout ValidateEncodedValue(const EncodedValueProto &value,
                                          bool require_resolved_reference = true) const;

private:
  const ModelProto *model_ = nullptr;
};

/**
 * Validates a ValueInfoProto.
 *
 * @param value_info Value information to validate.
 * @param is_main_graph When false (subgraph input/output), the ``type`` field
 * is not required to be present, mirroring the relaxed constraint that
 * applies to control-flow subgraphs.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyValueInfo(const ValueInfoProto &value_info,
                                          bool is_main_graph = true);

/**
 * Validates a TensorProto: data_type is set, at most one payload field is
 * populated, and the populated field matches the declared data_type.
 *
 * @param tensor Tensor to validate.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyTensor(const TensorProto &tensor);

/**
 * Validates a SparseTensorProto: ``values`` and ``indices`` are individually
 * valid tensors and ``indices`` uses the required INT64 data type.
 *
 * @param sparse_tensor Sparse tensor to validate.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifySparseTensor(const SparseTensorProto &sparse_tensor);

/**
 * Validates an AttributeProto: exactly one value field is populated and it
 * matches the declared ``type``; nested graphs/tensors are recursively
 * validated.
 *
 * @param attribute Attribute to validate.
 * @param in_function_body True when the attribute belongs to a node inside a
 * FunctionProto body, in which case ``ref_attr_name`` is permitted.
 * @param scope Names visible to the attribute's nested subgraph (if any),
 * i.e. every name defined so far in the enclosing graph. Used to validate
 * that control-flow body subgraphs (Attribute.g / Attribute.graphs) can
 * legally reference outer-scope values.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyAttribute(const AttributeProto &attribute, bool in_function_body,
                                          const std::unordered_set<std::string> &scope);

/**
 * Catalogue-aware overload of VerifyAttribute(): additionally resolves the
 * ``type_ref`` identities used by ``TYPE_PROTO``/``TYPE_PROTOS`` attributes and
 * by nested subgraphs against @p struct_types.
 *
 * @param struct_types Catalogue resolving ``type_ref`` identities; when null the
 * attribute is checked on its own terms and unresolved references are tolerated.
 * @param attribute Attribute to validate.
 * @param in_function_body True when the attribute belongs to a node inside a
 * FunctionProto body.
 * @param scope Names visible to the attribute's nested subgraph (if any).
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyAttribute(const StructTypeCatalogue *struct_types,
                                          const AttributeProto &attribute, bool in_function_body,
                                          const std::unordered_set<std::string> &scope);

/**
 * Validates a NodeProto: ``op_type`` is set, the node has at least one input
 * or output, attribute names are unique, and each attribute is valid.
 *
 * @param node Node to validate.
 * @param in_function_body True when ``node`` belongs to a FunctionProto body.
 * @param scope Names visible to the node, forwarded to VerifyAttribute() for
 * nested subgraph validation.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyNode(const NodeProto &node, bool in_function_body,
                                     const std::unordered_set<std::string> &scope);

/**
 * Catalogue-aware overload of VerifyNode(): forwards @p struct_types to
 * VerifyAttribute() so typed attributes and nested subgraphs resolve their
 * ``type_ref`` identities.
 *
 * @param struct_types Catalogue resolving ``type_ref`` identities; may be null.
 * @param node Node to validate.
 * @param in_function_body True when ``node`` belongs to a FunctionProto body.
 * @param scope Names visible to the node.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyNode(const StructTypeCatalogue *struct_types, const NodeProto &node,
                                     bool in_function_body,
                                     const std::unordered_set<std::string> &scope);

/**
 * Validates a GraphProto: inputs/outputs/initializers have unique names (across
 * dense, sparse and encoded initializers), the graph is in SSA form, nodes are
 * topologically sorted (every input is produced by a prior node, initializer,
 * graph input, or outer scope), every declared output is actually produced, and
 * every encoded initializer describes a valid flat byte layout.
 *
 * @param graph Graph to validate.
 * @param is_main_graph When false, relaxes the ValueInfoProto ``type``
 * requirement on graph inputs/outputs (subgraphs may omit it).
 * @param in_function_body True when ``graph`` is nested inside a
 * FunctionProto body.
 * @param outer_scope Names visible from an enclosing graph (used when
 * validating a control-flow body subgraph); may be null for the main graph.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyGraph(const GraphProto &graph, bool is_main_graph = true,
                                      bool in_function_body = false,
                                      const std::unordered_set<std::string> *outer_scope = nullptr);

/**
 * Catalogue-aware overload of VerifyGraph().
 *
 * @param struct_types Catalogue resolving the ``type_ref`` identities used by
 * encoded initializers and structured types. When null the graph is checked on
 * its own terms and a reference the graph cannot resolve suspends the layout
 * checks instead of failing; VerifyModel() supplies the model catalogue.
 * @param graph Graph to validate.
 * @param is_main_graph When false, relaxes the ValueInfoProto ``type``
 * requirement on graph inputs/outputs (subgraphs may omit it).
 * @param in_function_body True when ``graph`` is nested inside a FunctionProto body.
 * @param outer_scope Names visible from an enclosing graph; may be null.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyGraph(const StructTypeCatalogue *struct_types,
                                      const GraphProto &graph, bool is_main_graph = true,
                                      bool in_function_body = false,
                                      const std::unordered_set<std::string> *outer_scope = nullptr);

/**
 * Validates a FunctionProto: ``name`` is set, inputs are uniquely named,
 * nodes are topologically sorted with respect to the function's inputs, and
 * every declared output is produced.
 *
 * @param function Function to validate.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyFunction(const FunctionProto &function);

/**
 * Catalogue-aware overload of VerifyFunction(): validates the function's
 * ``value_info`` types and every node attribute against @p struct_types.
 *
 * @param struct_types Catalogue resolving ``type_ref`` identities; may be null.
 * @param function Function to validate.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyFunction(const StructTypeCatalogue *struct_types,
                                         const FunctionProto &function);

/**
 * Validates an in-memory ModelProto without requiring an operator-schema
 * registry: ``graph`` is present, at least one opset is imported with unique
 * domains, ``struct_types`` declares unique nonzero identities without
 * unresolved references or cycles, the main graph is structurally valid, and
 * model-local functions are individually valid and uniquely identified by
 * (domain, name, overload).
 *
 * This performs IR-level structural checks only; it does not check operator
 * input/output arity or attribute constraints, and it does not run shape
 * inference. Use ``onnx_lib::checker::check_model()`` for that.
 *
 * @param model Model to validate.
 *
 * @throws std::invalid_argument Thrown when a structural inconsistency is found.
 */
ONNX_LIGHT_PROTO_API void VerifyModel(const ModelProto &model);

} // namespace ONNX_LIGHT_NAMESPACE

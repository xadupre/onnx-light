// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file shape_inference.h
 * @brief Declares interfaces and helper utilities for operator shape inference.
 *
 * This header defines the abstract contexts consumed by OpSchema
 * TypeAndShapeInferenceFunction callbacks and the helper functions used by
 * schema implementations to propagate and merge element-type and shape data.
 *
 * Key types:
 * - @ref onnx::ShapeInferenceOptions – runtime knobs for the inference pass.
 * - @ref onnx::InferenceContext – abstract view of node inputs/outputs for
 *   type-and-shape inference callbacks registered on @c OpSchema.
 * - @ref onnx::DataPropagationContext – abstract view for data-propagation
 *   callbacks that carry statically known integer tensor values.
 * - @ref onnx::GraphInferencer – delegate for inferring subgraph attributes.
 *
 * Convenience helpers (ONNX-compatible inline functions):
 * - @ref onnx::propagateElemTypeFromInputToOutput
 * - @ref onnx::updateOutputElemType
 * - @ref onnx::updateOutputShape
 * - @ref onnx::bidirectionalBroadcastShapeInference
 * - @ref onnx::propagateShapeAndTypeFromFirstInput
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "onnx/common/common.h"
#include "onnx/common/proto_utils.h"

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Stores runtime options controlling schema-level shape inference.
 *
 * Pass an instance of this struct to InferShapes() or node-level inference
 * entry points to tune error-handling and data-propagation behaviour.
 */
struct ShapeInferenceOptions {
  /// When @c true, validates that input/output element types are mutually consistent.
  bool check_type;
  /// Error handling mode: @c 1 throws on every node-level error; @c 0 continues where possible.
  int error_mode;
  /// When @c true, activates limited constant-data propagation for operators that need it.
  bool enable_data_propagation;

  /**
   * Constructs a ShapeInferenceOptions with explicit control flags.
   * @param check_type_val    Enable element-type consistency checks.
   * @param strict_mode_val   Error mode (1 = strict, 0 = lenient).
   * @param data_prop_val     Enable limited data propagation.
   */
  explicit ShapeInferenceOptions(bool check_type_val = false, int strict_mode_val = 0,
                                 bool data_prop_val = false)
      : check_type(check_type_val), error_mode(strict_mode_val),
        enable_data_propagation(data_prop_val) {}
};

/**
 * Provides graph-level inference for attributes containing subgraphs.
 *
 * Operator shape-inference functions that encounter @c GRAPH-typed attributes
 * call getGraphAttributeInferencer() on @ref InferenceContext to obtain a
 * GraphInferencer for that attribute, then invoke doInferencing() to obtain
 * output types for the subgraph.  Concrete implementations are supplied by
 * the inference engine.
 */
class GraphInferencer {
public:
  /// Destroys the inferencer.
  virtual ~GraphInferencer() = default;
  /**
   * Infers output types for a subgraph attribute.
   *
   * @param input_types  Element-type and shape information for each subgraph input;
   *                     entries may be @c nullptr when a type is unknown.
   * @param input_data   Optional constant tensor data for subgraph inputs that carry
   *                     statically known values; entries may be @c nullptr.
   * @return A vector of pointers to inferred output TypeProto values, one per
   *         subgraph output.  Returns an empty vector by default (no inference).
   */
  virtual std::vector<const TypeProto *>
  doInferencing(const std::vector<const TypeProto *> & /*input_types*/,
                const std::vector<const TensorProto *> & /*input_data*/) {
    return {};
  }
};

/**
 * Represents shape/type inference failures with optional contextual details.
 *
 * Thrown by @ref fail_type_inference and @ref fail_shape_inference when the
 * inference function detects an inconsistency.  The error message is prefixed
 * with either @c [TypeInferenceError] or @c [ShapeInferenceError] so callers
 * can distinguish the two categories.
 */
struct InferenceError final : public std::runtime_error {
  using std::runtime_error::runtime_error;

  /**
   * Constructs an InferenceError with the given message.
   * @param message Human-readable description of the inference failure.
   */
  explicit InferenceError(const std::string &message) : std::runtime_error(message) {}

  /**
   * Returns the error message, with context appended when AppendContext() was called.
   * @return Null-terminated error string; includes context section when set.
   */
  const char *what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  /**
   * Appends contextual information describing where the error occurred.
   *
   * Typically invoked by the inference engine to attach the node or graph
   * name to the base error message before propagating it upward.
   *
   * @param context A human-readable description of the error location (e.g.,
   *                the operator name and node display name).
   */
  void AppendContext(const std::string &context) {
    expanded_message_ =
        ONNX_LIGHT_NAMESPACE::MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

private:
  std::string expanded_message_;
};

/**
 * @def fail_type_inference(...)
 * Throws an InferenceError tagged as a type-inference failure.
 *
 * Use this macro inside TypeAndShapeInferenceFunction callbacks to signal that
 * the input element types are incompatible with the operator's type constraints.
 * The message is prefixed with @c "[TypeInferenceError]" so it can be
 * distinguished from shape-inference errors at the catch site.
 *
 * Example:
 * @code
 * if (!ctx.getInputType(0)->has_tensor_type())
 *   fail_type_inference("input 0 must be a tensor");
 * @endcode
 */
#define fail_type_inference(...)                                                                   \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::InferenceError(                                              \
      ONNX_LIGHT_NAMESPACE::MakeString("[TypeInferenceError] ", __VA_ARGS__)));

/**
 * @def fail_shape_inference(...)
 * Throws an InferenceError tagged as a shape-inference failure.
 *
 * Use this macro inside TypeAndShapeInferenceFunction callbacks to signal that
 * the inferred output shape is inconsistent with declared constraints or with
 * the observed input shapes.  The message is prefixed with
 * @c "[ShapeInferenceError]".
 *
 * Example:
 * @code
 * if (!hasNInputShapes(ctx, 1))
 *   fail_shape_inference("input 0 must have a known shape");
 * @endcode
 */
#define fail_shape_inference(...)                                                                  \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::InferenceError(                                              \
      ONNX_LIGHT_NAMESPACE::MakeString("[ShapeInferenceError] ", __VA_ARGS__)));

/**
 * Supplies inputs, outputs, and attributes to operator type-and-shape inferencers.
 *
 * An @c InferenceContext is created by the inference engine for each node and
 * passed to the TypeAndShapeInferenceFunction registered on the node's
 * @c OpSchema.  Implementations are provided by the engine; schema authors only
 * consume the interface.
 *
 * Typical usage inside an inference function:
 * @code
 * void MyOpInference(InferenceContext& ctx) {
 *   propagateElemTypeFromInputToOutput(ctx, 0, 0);   // copy elem type
 *   if (hasNInputShapes(ctx, 1))
 *     updateOutputShape(ctx, 0, ctx.getInputType(0)->tensor_type().shape());
 * }
 * @endcode
 */
struct InferenceContext {
  /**
   * Returns the attribute with the given name, or @c nullptr if absent.
   * @param name Attribute name as it appears in the operator schema.
   * @return Pointer to the AttributeProto, or @c nullptr when not present.
   */
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  /**
   * Returns the number of node inputs.
   * @return Count of inputs declared for this node (including optional absent ones).
   */
  virtual size_t getNumInputs() const = 0;
  /**
   * Returns the input type for the given index, or @c nullptr when unknown.
   * @param index Zero-based input index.
   * @return Pointer to the TypeProto, or @c nullptr when the type is unavailable.
   */
  virtual const TypeProto *getInputType(size_t index) const = 0;
  /**
   * Returns static input tensor data for the given index when available.
   *
   * Provides access to constant-folded tensor values that the data-propagation
   * pass has computed for earlier nodes in the graph.
   *
   * @param index Zero-based input index.
   * @return Pointer to the TensorProto, or @c nullptr when the value is unknown.
   */
  virtual const TensorProto *getInputData(size_t index) const = 0;
  /**
   * Returns the number of node outputs.
   * @return Count of outputs declared for this node.
   */
  virtual size_t getNumOutputs() const = 0;
  /**
   * Returns a mutable output type for the given index.
   *
   * Inference functions write their results through this pointer.  The
   * engine allocates the TypeProto; callers must not delete it.
   *
   * @param index Zero-based output index.
   * @return Mutable pointer to the TypeProto, or @c nullptr when unavailable.
   */
  virtual TypeProto *getOutputType(size_t index) = 0;
  /**
   * Returns an inferencer for the graph-typed attribute with the given name.
   *
   * Operators with @c GRAPH attributes (e.g., @c If, @c Loop, @c Scan) use
   * this method to obtain a GraphInferencer and recursively propagate types
   * into the subgraph body.
   *
   * @param attribute_name Name of the @c GRAPH-typed attribute.
   * @return Pointer to the GraphInferencer, or @c nullptr when not applicable.
   */
  virtual GraphInferencer *getGraphAttributeInferencer(const std::string &attribute_name) = 0;
  /// Destroys the context.
  virtual ~InferenceContext() = default;
  /**
   * Returns @c true when the input at @p index has a known type.
   * @param index Zero-based input index.
   * @return @c true if getInputType(index) is non-null; @c false otherwise.
   */
  virtual bool hasInput(size_t index) const { return getInputType(index) != nullptr; }
  /**
   * Returns @c true when the output at @p index has a TypeProto allocated.
   * @param index Zero-based output index.
   * @return @c true if getOutputType(index) is non-null; @c false otherwise.
   */
  virtual bool hasOutput(size_t index) { return getOutputType(index) != nullptr; }
  /**
   * Returns sparse input data for the given index when available.
   * @param index Zero-based input index.
   * @return Pointer to the SparseTensorProto, or @c nullptr (default) when unavailable.
   */
  virtual const SparseTensorProto *getInputSparseData(size_t /*index*/) const { return nullptr; }
  /**
   * Returns symbolic (data-propagation) input for the given index when available.
   *
   * The inference engine stores statically inferred integer values for shape
   * tensors here so that downstream operators (e.g., @c Reshape) can use them
   * during shape inference.
   *
   * @param index Zero-based input index.
   * @return Pointer to the TensorShapeProto encoding the known values, or
   *         @c nullptr (default) when no symbolic data is available.
   */
  virtual const TensorShapeProto *getSymbolicInput(size_t /*index*/) const { return nullptr; }
  /**
   * Returns a display name for use in diagnostic messages.
   * @return A human-readable identifier (e.g., node name), or empty string.
   */
  virtual std::string getDisplayName() const { return ""; }
};

/**
 * Supplies tensor-shape constants to data-propagation functions.
 *
 * DataPropagationContext is passed to the DataPropagationFunction registered
 * on @c OpSchema.  Operators that produce shape tensors (e.g., @c Shape,
 * @c Gather applied to a shape) use addOutputData() to store statically known
 * values so that subsequent operators (@c Reshape, @c Expand) can consume them
 * during shape inference via InferenceContext::getSymbolicInput().
 *
 * @note Only integer-valued tensors can be propagated; float or string tensors
 *       are not supported.
 */
struct DataPropagationContext {
  /**
   * Returns the attribute with the given name, or @c nullptr if absent.
   * @param name Attribute name as it appears in the operator schema.
   * @return Pointer to the AttributeProto, or @c nullptr when not present.
   */
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  /**
   * Returns the number of node inputs.
   * @return Count of inputs declared for this node.
   */
  virtual size_t getNumInputs() const = 0;
  /**
   * Returns the input type for the given index, or @c nullptr when unknown.
   * @param index Zero-based input index.
   * @return Pointer to the TypeProto, or @c nullptr when unavailable.
   */
  virtual const TypeProto *getInputType(size_t index) const = 0;
  /**
   * Returns the number of node outputs.
   * @return Count of outputs declared for this node.
   */
  virtual size_t getNumOutputs() const = 0;
  /**
   * Returns the output type for the given index, or @c nullptr when unknown.
   * @param index Zero-based output index.
   * @return Pointer to the TypeProto, or @c nullptr when unavailable.
   */
  virtual const TypeProto *getOutputType(size_t index) const = 0;
  /**
   * Returns propagated input shape data for the given index, or @c nullptr when unknown.
   *
   * Retrieves integer values previously stored by an upstream operator's
   * DataPropagationFunction via addOutputData().
   *
   * @param index Zero-based input index.
   * @return Pointer to a TensorShapeProto encoding the element values, or
   *         @c nullptr when no propagated data is available.
   */
  virtual const TensorShapeProto *getInputData(size_t index) = 0;
  /**
   * Stores propagated integer tensor data for the given output index.
   *
   * Called by the data-propagation function to record statically computed
   * values.  The engine will make these available to downstream nodes via
   * InferenceContext::getSymbolicInput().
   *
   * @param index Zero-based output index.
   * @param tp    TensorShapeProto encoding the element values (moved in).
   */
  virtual void addOutputData(size_t index, TensorShapeProto &&tp) = 0;
  /// Destroys the context.
  virtual ~DataPropagationContext() = default;
};

/// Signature for type-and-shape inference callbacks registered on @c OpSchema.
using InferenceFunction = std::function<void(InferenceContext &)>;
/// Signature for data-propagation callbacks registered on @c OpSchema.
using DataPropagationFunction = std::function<void(DataPropagationContext &)>;

/// No-op inference callback used as a default placeholder for operators without inference.
inline void dummyInferenceFunction(InferenceContext & /*unused*/) {}

/// No-op data propagation callback used as a default placeholder for operators without propagation.
inline void dummyDataPropagationFunction(DataPropagationContext & /*unused*/) {}

/**
 * @name ONNX-compatible helper APIs
 * These helpers mirror onnx/defs/shape_inference.h while using onnx_light's
 * nanopb-based TypeProto API.  They are intended for use inside
 * TypeAndShapeInferenceFunction callbacks registered on @c OpSchema.
 *
 * All helpers are inline and depend only on the abstract InferenceContext
 * interface, so they can be called from any schema-registration translation
 * unit without linking against a separate inference library.
 */
/// @{

/**
 * Copies the element type of input @p inputIndex to output @p outputIndex.
 *
 * Handles tensor, sequence, optional, and map input types.
 *
 * @param ctx         Inference context supplying input and output type information.
 * @param inputIndex  Zero-based index of the source input.
 * @param outputIndex Zero-based index of the destination output.
 */
void propagateElemTypeFromInputToOutput(InferenceContext &ctx, size_t inputIndex,
                                        size_t outputIndex);

/**
 * Sets the element type of output @p outputIndex to @p elemType.
 *
 * Does nothing if the output TypeProto is unavailable.
 *
 * @param ctx         Inference context supplying output type information.
 * @param outputIndex Zero-based index of the output to update.
 * @param elemType    TensorProto::DataType value to assign.
 */
inline void updateOutputElemType(InferenceContext &ctx, size_t outputIndex, int32_t elemType) {
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().set_elem_type(elemType);
}

/**
 * Returns @c true when the first @p n inputs all have known tensor shapes.
 *
 * Convenience predicate used to guard shape propagation code that requires
 * all leading inputs to carry rank and dimension information.
 *
 * @param ctx Inference context supplying input type information.
 * @param n   Number of leading inputs to check.
 * @return @c true if every input in [0, n) has a tensor type with a shape;
 *         @c false as soon as one is missing.
 */
inline bool hasNInputShapes(const InferenceContext &ctx, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const auto *t = ctx.getInputType(i);
    if (!t || !t->has_tensor_type() || !t->tensor_type().has_shape()) {
      return false;
    }
  }
  return true;
}

/**
 * Returns @c true when input @p n has a known tensor shape.
 *
 * @tparam CTX  An inference context type exposing @c getInputType(size_t).
 * @param ctx   Inference context supplying input type information.
 * @param n     Zero-based input index.
 * @return @c true if the input has a tensor type with a shape present.
 */
template <typename CTX> inline bool hasInputShape(const CTX &ctx, size_t n) {
  const auto *t = ctx.getInputType(n);
  return t && t->has_tensor_type() && t->tensor_type().has_shape();
}

/**
 * Returns a reference to the tensor shape of input @p n.
 *
 * @pre @c hasInputShape(ctx, n) must be @c true; behaviour is undefined otherwise.
 *
 * @param ctx Inference context supplying input type information.
 * @param n   Zero-based input index.
 * @return Const reference to the TensorShapeProto of the input tensor.
 */
inline const TensorShapeProto &getInputShape(const InferenceContext &ctx, size_t n) {
  return ctx.getInputType(n)->tensor_type().shape();
}

/**
 * Returns the value of integer attribute @p attributeName, or @p defaultValue when absent.
 *
 * @param ctx            Inference context supplying attribute access.
 * @param attributeName  Name of the @c INT-typed attribute to look up.
 * @param defaultValue   Value returned when the attribute is absent or not an integer.
 * @return The attribute's integer value, or @p defaultValue.
 */
inline int64_t getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                            int64_t defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::INT) {
    return defaultValue;
  }
  return attr->i();
}

/**
 * Returns the value of float attribute @p attributeName, or @p defaultValue when absent.
 *
 * @param ctx            Inference context supplying attribute access.
 * @param attributeName  Name of the @c FLOAT-typed attribute to look up.
 * @param defaultValue   Value returned when the attribute is absent or not a float.
 * @return The attribute's float value, or @p defaultValue.
 */
inline float getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                          float defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::FLOAT) {
    return defaultValue;
  }
  return attr->f();
}

/**
 * Returns the value of string attribute @p attributeName, or @p defaultValue when absent.
 *
 * @param ctx            Inference context supplying attribute access.
 * @param attributeName  Name of the @c STRING-typed attribute to look up.
 * @param defaultValue   Value returned when the attribute is absent or not a string.
 * @return The attribute's string value, or @p defaultValue.
 */
inline std::string getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                                const std::string &defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::STRING) {
    return defaultValue;
  }
  return attr->ref_s().as_string();
}

/**
 * Propagates the @c INT attribute @p attributeName as the element type of output @p outputIndex.
 *
 * Typically used by operators like @c Cast that express their output type via an attribute.
 * Does nothing if the attribute is absent or is not an integer.
 *
 * @param ctx            Inference context supplying attribute and output type access.
 * @param attributeName  Name of the @c INT-typed data-type attribute (e.g., @c "to").
 * @param outputIndex    Zero-based index of the output whose element type is set.
 */
inline void propagateElemTypeFromAttributeToOutput(InferenceContext &ctx,
                                                   const std::string &attributeName,
                                                   size_t outputIndex) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::INT) {
    return;
  }
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().set_elem_type(static_cast<int>(attr->i()));
}

/**
 * Copies the given @p shape into the tensor shape of output @p outputIndex.
 *
 * Does nothing if the output TypeProto is unavailable.
 *
 * @param ctx         Inference context supplying output type access.
 * @param outputIndex Zero-based index of the output to update.
 * @param shape       Source shape to copy.
 */
inline void updateOutputShape(InferenceContext &ctx, size_t outputIndex,
                              const TensorShapeProto &shape) {
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().shape().CopyFrom(shape);
}

/**
 * Returns the tensor element type of @p type, or @c -1 when unavailable.
 *
 * Returns @c -1 (not @c TensorProto::UNDEFINED) for non-tensor types or
 * when the element type field is not set.
 *
 * @param type TypeProto to inspect.
 * @return The @c TensorProto::DataType integer, or @c -1.
 */
inline int32_t getTensorElementType(const TypeProto &type) {
  if (!type.has_tensor_type() || !type.tensor_type().has_elem_type()) {
    return -1;
  }
  return static_cast<int32_t>(type.tensor_type().elem_type());
}

/**
 * Copies the full type and shape from input 0 to output 0.
 *
 * Convenience helper for unary operators that pass their input type
 * unchanged (e.g., @c Identity, @c Relu).  Does nothing when either
 * the input or output TypeProto is unavailable.
 *
 * @param ctx Inference context supplying input and output type access.
 */
inline void propagateShapeAndTypeFromFirstInput(InferenceContext &ctx) {
  const auto *in_type = ctx.getInputType(0);
  if (!in_type) {
    return;
  }
  auto *out_type = ctx.getOutputType(0);
  if (!out_type) {
    return;
  }
  out_type->CopyFrom(*in_type);
}

/**
 * Infers the output shape from two input shapes using NumPy-style broadcasting semantics.
 *
 * Implements the bidirectional broadcast rule: shapes are right-aligned and
 * each output dimension is the maximum of the two corresponding input
 * dimensions.  A dimension of 1 in either input is broadcast to match the
 * other.  Symbolic (string-param) dimensions are preserved when they match;
 * otherwise the output dimension is left unset (unknown).
 *
 * @param shape1        First input shape.
 * @param shape2        Second input shape.
 * @param output_shape  Destination shape; new dimensions are appended.
 */
inline void bidirectionalBroadcastShapeInference(const TensorShapeProto &shape1,
                                                 const TensorShapeProto &shape2,
                                                 TensorShapeProto &output_shape) {
  const auto &dims1 = shape1.dim();
  const auto &dims2 = shape2.dim();
  const size_t rank1 = dims1.size();
  const size_t rank2 = dims2.size();
  const size_t out_rank = std::max(rank1, rank2);
  for (size_t i = 0; i < out_rank; ++i) {
    // Align dimensions from the right (NumPy-style: extend on the left with 1s).
    const bool has1 = i >= out_rank - rank1;
    const bool has2 = i >= out_rank - rank2;
    const size_t idx1 = has1 ? i - (out_rank - rank1) : static_cast<size_t>(0);
    const size_t idx2 = has2 ? i - (out_rank - rank2) : static_cast<size_t>(0);
    auto *out_dim = output_shape.add_dim();
    const bool d1_is_one = has1 && dims1[idx1].has_dim_value() && dims1[idx1].dim_value() == 1;
    const bool d2_is_one = has2 && dims2[idx2].has_dim_value() && dims2[idx2].dim_value() == 1;
    if (!has1) {
      // Shape1 is shorter: this position is effectively 1, so output = shape2's dim.
      out_dim->CopyFrom(dims2[idx2]);
    } else if (!has2) {
      // Shape2 is shorter: output = shape1's dim.
      out_dim->CopyFrom(dims1[idx1]);
    } else if (d1_is_one) {
      out_dim->CopyFrom(dims2[idx2]);
    } else if (d2_is_one) {
      out_dim->CopyFrom(dims1[idx1]);
    } else if (dims1[idx1].has_dim_value() && dims2[idx2].has_dim_value()) {
      // Both concrete and neither is 1 — they must be equal for valid broadcast.
      out_dim->set_dim_value(dims1[idx1].dim_value());
    }
    // else: at least one dim is symbolic/unknown → leave out_dim without a value.
  }
}
/// @}

// ---------------------------------------------------------------------------
// Forward declarations for non-inline shape-inference helpers defined in
// shape_inference.cc. These are required by the full operator definition files
// (controlflow, sequence, etc.) that are compiled as part of lib_onnx_cpp.
// ---------------------------------------------------------------------------

/// Merges shape information from @p source_shape into @p target_type.
void mergeInShapeInfo(const TensorShapeProto &source_shape, TypeProto_Tensor &target_type);
/// Merges shape information from @p source_shape into @p target_type (sparse tensor).
void mergeInShapeInfo(const TensorShapeProto &source_shape, TypeProto_SparseTensor &target_type);
/// Merges shape information from @p source into @p target (tensor-to-tensor).
void mergeInShapeInfo(const TypeProto_Tensor &source, TypeProto_Tensor &target);
/// Merges shape information from @p source into @p target (sparse-tensor-to-sparse-tensor).
void mergeInShapeInfo(const TypeProto_SparseTensor &source, TypeProto_SparseTensor &target);

/// Unions shape information from @p source_shape into @p target_type (tensor).
void UnionShapeInfo(const TensorShapeProto &source_shape, TypeProto_Tensor &target_type);
/// Unions shape information from @p source_shape into @p target_type (sparse tensor).
void UnionShapeInfo(const TensorShapeProto &source_shape, TypeProto_SparseTensor &target_type);
/// Unions type information from @p source_type into @p target_type.
void UnionTypeInfo(const TypeProto &source_type, TypeProto &target_type);

/// Propagates element type from @p input_type to @p output_type with validation.
void propagateElemTypeWithValidation(const TypeProto *input_type, TypeProto *output_type);

} // namespace ONNX_LIGHT_NAMESPACE

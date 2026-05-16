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
 */
struct ShapeInferenceOptions {
  /// Checks input/output type consistency when true.
  bool check_type;
  /// Controls node-level error behavior (1 throws immediately, 0 continues when possible).
  int error_mode;
  /// Enables limited data propagation used by some operators for shape computation.
  bool enable_data_propagation;
  /// Initializes shape inference options.
  explicit ShapeInferenceOptions(bool check_type_val = false, int strict_mode_val = 0,
                                 bool data_prop_val = false)
      : check_type(check_type_val), error_mode(strict_mode_val),
        enable_data_propagation(data_prop_val) {}
};

/**
 * Provides graph-level inference for attributes containing subgraphs.
 */
class GraphInferencer {
public:
  /// Destroys the inferencer.
  virtual ~GraphInferencer() = default;
  /**
   * Infers output types for a subgraph attribute.
   * @param input_types Input type information.
   * @param input_data Optional constant tensor input data.
   * @return Inferred output types.
   */
  virtual std::vector<const TypeProto *>
  doInferencing(const std::vector<const TypeProto *> & /*input_types*/,
                const std::vector<const TensorProto *> & /*input_data*/) {
    return {};
  }
};

/**
 * Represents shape/type inference failures with optional contextual details.
 */
struct InferenceError final : public std::runtime_error {
  using std::runtime_error::runtime_error;

  /// Initializes the error with a message.
  explicit InferenceError(const std::string &message) : std::runtime_error(message) {}

  /// Returns the contextualized message when available.
  const char *what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  /// Appends contextual information describing where the error occurred.
  void AppendContext(const std::string &context) {
    expanded_message_ =
        ONNX_LIGHT_NAMESPACE::MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

private:
  std::string expanded_message_;
};

/// Throws an InferenceError tagged as a type-inference failure.
#define fail_type_inference(...)                                                                   \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::InferenceError(                                              \
      ONNX_LIGHT_NAMESPACE::MakeString("[TypeInferenceError] ", __VA_ARGS__)));

/// Throws an InferenceError tagged as a shape-inference failure.
#define fail_shape_inference(...)                                                                  \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::InferenceError(                                              \
      ONNX_LIGHT_NAMESPACE::MakeString("[ShapeInferenceError] ", __VA_ARGS__)));

/**
 * Supplies inputs, outputs, and attributes to operator type-and-shape inferencers.
 */
struct InferenceContext {
  /// Returns the attribute with the given name, or nullptr if absent.
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  /// Returns the number of node inputs.
  virtual size_t getNumInputs() const = 0;
  /// Returns the input type for index, or nullptr if unknown.
  virtual const TypeProto *getInputType(size_t index) const = 0;
  /// Returns static input tensor data for index when available.
  virtual const TensorProto *getInputData(size_t index) const = 0;
  /// Returns the number of node outputs.
  virtual size_t getNumOutputs() const = 0;
  /// Returns a mutable output type for index, or nullptr if unavailable.
  virtual TypeProto *getOutputType(size_t index) = 0;
  /// Returns an inferencer for a graph-typed attribute.
  virtual GraphInferencer *getGraphAttributeInferencer(const std::string &attribute_name) = 0;
  /// Destroys the context.
  virtual ~InferenceContext() = default;
  /// Returns whether the input type at index is available.
  virtual bool hasInput(size_t index) const { return getInputType(index) != nullptr; }
  /// Returns whether the output type at index is available.
  virtual bool hasOutput(size_t index) { return getOutputType(index) != nullptr; }
  /// Returns sparse input data for index when available.
  virtual const SparseTensorProto *getInputSparseData(size_t /*index*/) const { return nullptr; }
  /// Returns symbolic input data for index when available.
  virtual const TensorShapeProto *getSymbolicInput(size_t /*index*/) const { return nullptr; }
  /// Returns a display name used in diagnostic messages.
  virtual std::string getDisplayName() const { return ""; }
};

/**
 * Supplies tensor-shape constants to data-propagation functions.
 */
struct DataPropagationContext {
  /// Returns the attribute with the given name, or nullptr if absent.
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  /// Returns the number of node inputs.
  virtual size_t getNumInputs() const = 0;
  /// Returns the input type for index, or nullptr if unknown.
  virtual const TypeProto *getInputType(size_t index) const = 0;
  /// Returns the number of node outputs.
  virtual size_t getNumOutputs() const = 0;
  /// Returns the output type for index, or nullptr if unknown.
  virtual const TypeProto *getOutputType(size_t index) const = 0;
  /// Returns propagated input shape data for index, or nullptr if unknown.
  virtual const TensorShapeProto *getInputData(size_t index) = 0;
  /// Publishes propagated output shape data for index.
  virtual void addOutputData(size_t index, TensorShapeProto &&tp) = 0;
  /// Destroys the context.
  virtual ~DataPropagationContext() = default;
};

/// Signature for type-and-shape inference callbacks.
using InferenceFunction = std::function<void(InferenceContext &)>;
/// Signature for data-propagation callbacks.
using DataPropagationFunction = std::function<void(DataPropagationContext &)>;

/// No-op inference callback used as a default placeholder.
inline void dummyInferenceFunction(InferenceContext & /*unused*/) {}

/// No-op data propagation callback used as a default placeholder.
inline void dummyDataPropagationFunction(DataPropagationContext & /*unused*/) {}

/**
 * @name ONNX-compatible helper APIs
 * These helpers mirror onnx/defs/shape_inference.h while using onnx_light's
 * nanopb-based TypeProto API.
 */
/// @{

/// Propagates the input element type from inputIndex to outputIndex.
inline void propagateElemTypeFromInputToOutput(InferenceContext &ctx, size_t inputIndex,
                                               size_t outputIndex) {
  const auto *in_type = ctx.getInputType(inputIndex);
  if (!in_type || !in_type->has_tensor_type()) {
    return;
  }
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().set_elem_type(static_cast<int>(in_type->tensor_type().elem_type()));
}

/// Sets the output element type at outputIndex.
inline void updateOutputElemType(InferenceContext &ctx, size_t outputIndex, int32_t elemType) {
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().set_elem_type(elemType);
}

/// Returns true when the first n inputs have tensor shapes.
inline bool hasNInputShapes(const InferenceContext &ctx, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const auto *t = ctx.getInputType(i);
    if (!t || !t->has_tensor_type() || !t->tensor_type().has_shape()) {
      return false;
    }
  }
  return true;
}

/// Returns true when input n has a tensor shape.
template <typename CTX> inline bool hasInputShape(const CTX &ctx, size_t n) {
  const auto *t = ctx.getInputType(n);
  return t && t->has_tensor_type() && t->tensor_type().has_shape();
}

/// Returns the tensor shape of input n.
inline const TensorShapeProto &getInputShape(const InferenceContext &ctx, size_t n) {
  return ctx.getInputType(n)->tensor_type().shape();
}

/** Returns int64 attribute attributeName or defaultValue when absent or not AttributeProto::INT. */
inline int64_t getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                            int64_t defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::INT) {
    return defaultValue;
  }
  return attr->i();
}

/** Returns float attribute attributeName or defaultValue when absent or not AttributeProto::FLOAT.
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
 * Returns string attribute attributeName or defaultValue when absent or not
 * AttributeProto::STRING.
 */
inline std::string getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                                const std::string &defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::STRING) {
    return defaultValue;
  }
  return attr->ref_s().as_string();
}

/// Propagates the int attribute named by attributeName to the output element type.
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

/// Copies shape information from the shape parameter to outputIndex.
inline void updateOutputShape(InferenceContext &ctx, size_t outputIndex,
                              const TensorShapeProto &shape) {
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().shape().CopyFrom(shape);
}

/** Returns tensor element type, or -1 when unavailable. */
inline int32_t getTensorElementType(const TypeProto &type) {
  if (!type.has_tensor_type() || !type.tensor_type().has_elem_type()) {
    return -1;
  }
  return static_cast<int32_t>(type.tensor_type().elem_type());
}

/// Copies full type information from input 0 to output 0.
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

/** Infers bidirectional broadcast shape using NumPy semantics. */
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

} // namespace ONNX_LIGHT_NAMESPACE

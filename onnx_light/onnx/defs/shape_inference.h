// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "onnx/common/common.h"
#include "onnx/common/proto_utils.h"

namespace ONNX_LIGHT_NAMESPACE {

struct ShapeInferenceOptions {
  // Checks the type-equality for input and output.
  bool check_type;
  // 1: Will throw any node-level shape inference errors.
  // 0: Won't throw node-level errors but other errors (like merging shapes) are thrown.
  int error_mode;
  // Enables data propagation for limited operators to perform shape computation.
  bool enable_data_propagation;
  explicit ShapeInferenceOptions(bool check_type_val = false, int strict_mode_val = 0,
                                 bool data_prop_val = false)
      : check_type(check_type_val), error_mode(strict_mode_val),
        enable_data_propagation(data_prop_val) {}
};

class GraphInferencer {
public:
  virtual ~GraphInferencer() = default;
  virtual std::vector<const TypeProto *>
  doInferencing(const std::vector<const TypeProto *> & /*input_types*/,
                const std::vector<const TensorProto *> & /*input_data*/) {
    return {};
  }
};

struct InferenceError final : public std::runtime_error {
  using std::runtime_error::runtime_error;

  explicit InferenceError(const std::string &message) : std::runtime_error(message) {}

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

#define fail_type_inference(...)                                                                   \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::InferenceError(                                              \
      ONNX_LIGHT_NAMESPACE::MakeString("[TypeInferenceError] ", __VA_ARGS__)));

#define fail_shape_inference(...)                                                                  \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::InferenceError(                                              \
      ONNX_LIGHT_NAMESPACE::MakeString("[ShapeInferenceError] ", __VA_ARGS__)));

struct InferenceContext {
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto *getInputType(size_t index) const = 0;
  virtual const TensorProto *getInputData(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto *getOutputType(size_t index) = 0;
  virtual GraphInferencer *getGraphAttributeInferencer(const std::string &attribute_name) = 0;
  virtual ~InferenceContext() = default;
  virtual bool hasInput(size_t index) const { return getInputType(index) != nullptr; }
  virtual bool hasOutput(size_t index) { return getOutputType(index) != nullptr; }
  virtual const SparseTensorProto *getInputSparseData(size_t /*index*/) const { return nullptr; }
  virtual const TensorShapeProto *getSymbolicInput(size_t /*index*/) const { return nullptr; }
  virtual std::string getDisplayName() const { return ""; }
};

struct DataPropagationContext {
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto *getInputType(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual const TypeProto *getOutputType(size_t index) const = 0;
  virtual const TensorShapeProto *getInputData(size_t index) = 0;
  virtual void addOutputData(size_t index, TensorShapeProto &&tp) = 0;
  virtual ~DataPropagationContext() = default;
};

using InferenceFunction = std::function<void(InferenceContext &)>;
using DataPropagationFunction = std::function<void(DataPropagationContext &)>;

inline void dummyInferenceFunction(InferenceContext & /*unused*/) {}

inline void dummyDataPropagationFunction(DataPropagationContext & /*unused*/) {}

// ---------------------------------------------------------------------------
// Onnx_light-compatible helpers for operator type-and-shape inference.
// These provide the same interface as onnx/defs/shape_inference.h from the
// full ONNX library, but use onnx_light's nanopb TypeProto API instead of
// the google::protobuf API (no value_case(), mutable_*(), etc.).
// ---------------------------------------------------------------------------

// Propagates the element type from the input at inputIndex to the output at outputIndex.
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
  out_type->tensor_type().set_elem_type(
      static_cast<int>(in_type->tensor_type().elem_type()));
}

// Sets the element type of the output at outputIndex.
inline void updateOutputElemType(InferenceContext &ctx, size_t outputIndex, int32_t elemType) {
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().set_elem_type(elemType);
}

// Returns true if all n inputs have a tensor type with a shape.
inline bool hasNInputShapes(const InferenceContext &ctx, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const auto *t = ctx.getInputType(i);
    if (!t || !t->has_tensor_type() || !t->tensor_type().has_shape()) {
      return false;
    }
  }
  return true;
}

// Returns true if the input at index n has a tensor type with a shape.
template <typename CTX>
inline bool hasInputShape(const CTX &ctx, size_t n) {
  const auto *t = ctx.getInputType(n);
  return t && t->has_tensor_type() && t->tensor_type().has_shape();
}

// Returns the shape of the input tensor at index n.
inline const TensorShapeProto &getInputShape(const InferenceContext &ctx, size_t n) {
  return ctx.getInputType(n)->tensor_type().shape();
}

// Returns the value of the int64 attribute named attributeName, or defaultValue if absent.
inline int64_t getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                            int64_t defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::INT) {
    return defaultValue;
  }
  return attr->i();
}

// Returns the value of the float attribute named attributeName, or defaultValue if absent.
inline float getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                          float defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::FLOAT) {
    return defaultValue;
  }
  return attr->f();
}

// Returns the value of the string attribute named attributeName, or defaultValue if absent.
inline std::string getAttribute(const InferenceContext &ctx, const std::string &attributeName,
                                const std::string &defaultValue) {
  const auto *attr = ctx.getAttribute(attributeName);
  if (!attr || attr->type() != AttributeProto::STRING) {
    return defaultValue;
  }
  return attr->ref_s().as_string();
}

// Propagates the element type specified by an "output_dtype" attribute to the output.
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

// Copies the shape from the input to the output.
inline void updateOutputShape(InferenceContext &ctx, size_t outputIndex,
                              const TensorShapeProto &shape) {
  auto *out_type = ctx.getOutputType(outputIndex);
  if (!out_type) {
    return;
  }
  out_type->tensor_type().shape().CopyFrom(shape);
}

// Returns the element type of the tensor type, or -1 if not a tensor type.
inline int32_t getTensorElementType(const TypeProto &type) {
  if (!type.has_tensor_type() || !type.tensor_type().has_elem_type()) {
    return -1;
  }
  return static_cast<int32_t>(type.tensor_type().elem_type());
}

// Copies both type and shape from input 0 to output 0.
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

// Stub for bidirectional broadcast shape inference.
// This stub is invoked when both inputs have shapes, but produces no output
// shape (onnx_light does not implement full broadcast logic here).  It exists
// solely to satisfy the linker / API compatibility with code that calls it.
inline void bidirectionalBroadcastShapeInference(const TensorShapeProto & /*shape1*/,
                                                  const TensorShapeProto & /*shape2*/,
                                                  TensorShapeProto & /*output_shape*/) {}

} // namespace ONNX_LIGHT_NAMESPACE

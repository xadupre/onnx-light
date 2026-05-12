// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>

#include "onnx/common/common.h"
#include "onnx/common/proto_utils.h"

namespace ONNX_LIGHT_NAMESPACE {

class GraphInferencer {
public:
  // Performs shape inferencing on the contained graph and returns output types.
  virtual std::vector<const TypeProto *>
  doInferencing(const std::vector<const TypeProto *> &inputTypes,
                const std::vector<const TensorProto *> &inputData) = 0;
  virtual ~GraphInferencer() = default;
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
  virtual bool hasInput(size_t index) const {
    return (index < getNumInputs()) && (getInputType(index) != nullptr);
  }
  virtual const TensorProto *getInputData(size_t index) const = 0;
  virtual const SparseTensorProto *getInputSparseData(size_t index) const = 0;
  virtual const TensorShapeProto *getSymbolicInput(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto *getOutputType(size_t index) = 0;
  virtual bool hasOutput(size_t index) {
    return (index < getNumOutputs()) && (getOutputType(index) != nullptr);
  }
  virtual GraphInferencer *getGraphAttributeInferencer(const std::string &attribute_name) = 0;
  virtual std::string getDisplayName() const { return ""; }
  virtual ~InferenceContext() = default;
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

// Options controlling the behaviour of shape and type inference.
struct ShapeInferenceOptions {
  // Checks type equality for inputs and outputs.
  bool check_type;
  // 1: throw any node-level shape-inference error; 0: skip node-level errors.
  int error_mode;
  // Enables data propagation for a limited set of operators.
  bool enable_data_propagation;

  explicit ShapeInferenceOptions(bool check_type_val = false, int strict_mode_val = 0,
                                 bool data_prop_val = false)
      : check_type(check_type_val), error_mode(strict_mode_val),
        enable_data_propagation(data_prop_val) {}
};

// Maintains a SymbolTable for symbolic shape inference.
class SymbolTable {
public:
  // Adds existing symbols from a main graph or subgraph.
  virtual void addFromGraph(const GraphProto &g) = 0;
  // Creates a new unique symbol with the default prefix "unk__".
  std::string createNew() { return createNew("unk__"); }
  // Creates a new unique symbol with the given prefix.
  virtual std::string createNew(const std::string &symbol_prefix) = 0;
  virtual ~SymbolTable() = default;
};

} // namespace ONNX_LIGHT_NAMESPACE

// Protobuf compatibility: map google::protobuf::RepeatedPtrField<T> to the
// onnx_light equivalent utils::RepeatedProtoField<T>.  This allows vendored
// ONNX sources that reference the protobuf type to compile unchanged.
namespace google {
namespace protobuf {
template <class T>
using RepeatedPtrField = ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<T>;
} // namespace protobuf
} // namespace google

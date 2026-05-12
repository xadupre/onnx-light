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
  virtual bool hasOutput(size_t index) {
    return getOutputType(index) != nullptr;
  }
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

} // namespace ONNX_LIGHT_NAMESPACE

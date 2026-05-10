// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>

#include "onnx/common/common.h"
#include "onnx/common/proto_utils.h"

namespace ONNX_NAMESPACE {

class GraphInferencer {
public:
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
        ONNX_NAMESPACE::MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

private:
  std::string expanded_message_;
};

#define fail_type_inference(...)                                                                   \
  ONNX_THROW_EX(ONNX_NAMESPACE::InferenceError(                                                    \
      ONNX_NAMESPACE::MakeString("[TypeInferenceError] ", __VA_ARGS__)));

#define fail_shape_inference(...)                                                                  \
  ONNX_THROW_EX(ONNX_NAMESPACE::InferenceError(                                                    \
      ONNX_NAMESPACE::MakeString("[ShapeInferenceError] ", __VA_ARGS__)));

struct InferenceContext {
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto *getInputType(size_t index) const = 0;
  virtual const TensorProto *getInputData(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto *getOutputType(size_t index) = 0;
  virtual GraphInferencer *getGraphAttributeInferencer(const std::string &attribute_name) = 0;
  virtual ~InferenceContext() = default;
};

struct DataPropagationContext {
  virtual const AttributeProto *getAttribute(const std::string &name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto *getInputType(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual const TypeProto *getOutputType(size_t index) const = 0;
  virtual ~DataPropagationContext() = default;
};

using InferenceFunction = std::function<void(InferenceContext &)>;
using DataPropagationFunction = std::function<void(DataPropagationContext &)>;

inline void dummyInferenceFunction(InferenceContext & /*unused*/) {}

inline void dummyDataPropagationFunction(DataPropagationContext & /*unused*/) {}

} // namespace ONNX_NAMESPACE

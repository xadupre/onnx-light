// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace shape_inference {

/// Adapts a node and its inputs to the `InferenceContext` interface.
///
/// This lightweight implementation is intended for node-level shape inference and
/// does not depend on `shape_inference/implementation.cc`.
class NodeInferenceContextImpl : public InferenceContext {
public:
  /// Initializes the context with a node and optional maps for input values.
  NodeInferenceContextImpl(
      NodeProto &n, const std::unordered_map<std::string, TypeProto> &input_types,
      const std::unordered_map<std::string, TensorProto> &input_data,
      const std::unordered_map<std::string, SparseTensorProto> &input_sparse_data) {
    for (size_t i = 0; i < n.ref_attribute().size(); ++i) {
      const AttributeProto &attr = n.ref_attribute()[i];
      attributes_by_name_[attr.ref_name().as_string()] = &attr;
    }
    for (size_t i = 0; i < n.ref_input().size(); ++i) {
      const std::string input_str = n.ref_input()[i].as_string();
      auto it = input_types.find(input_str);
      all_input_types_.push_back(it != input_types.end() ? &it->second : nullptr);
      auto data_it = input_data.find(input_str);
      all_input_data_.push_back(data_it != input_data.end() ? &data_it->second : nullptr);
      auto sparse_it = input_sparse_data.find(input_str);
      all_input_sparse_data_.push_back(sparse_it != input_sparse_data.end() ? &sparse_it->second
                                                                            : nullptr);
    }
    all_output_types_.resize(n.ref_output().size());
  }

  const AttributeProto *getAttribute(const std::string &name) const override {
    auto it = attributes_by_name_.find(name);
    return it == attributes_by_name_.end() ? nullptr : it->second;
  }

  size_t getNumInputs() const override { return all_input_types_.size(); }

  const TypeProto *getInputType(size_t index) const override {
    return index < all_input_types_.size() ? all_input_types_[index] : nullptr;
  }

  const TensorProto *getInputData(size_t index) const override {
    return index < all_input_data_.size() ? all_input_data_[index] : nullptr;
  }

  const SparseTensorProto *getInputSparseData(size_t index) const override {
    return index < all_input_sparse_data_.size() ? all_input_sparse_data_[index] : nullptr;
  }

  size_t getNumOutputs() const override { return all_output_types_.size(); }

  TypeProto *getOutputType(size_t index) override {
    return index < all_output_types_.size() ? &all_output_types_[index] : nullptr;
  }

  GraphInferencer *getGraphAttributeInferencer(const std::string &) override { return nullptr; }

  /// Stores inferred output types corresponding to node outputs.
  std::vector<TypeProto> all_output_types_;

private:
  std::unordered_map<std::string, const AttributeProto *> attributes_by_name_;
  std::vector<const TypeProto *> all_input_types_;
  std::vector<const TensorProto *> all_input_data_;
  std::vector<const SparseTensorProto *> all_input_sparse_data_;
};

} // namespace shape_inference
} // namespace ONNX_LIGHT_NAMESPACE

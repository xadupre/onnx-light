// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnx_lib/common/assertions.h"
#include "onnx_lib/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

class MeanVarianceNormalizationAdapter : public Adapter {
protected:
  MeanVarianceNormalizationAdapter(const OpSetID &initial_version, const OpSetID &target_version)
      : Adapter("MeanVarianceNormalization", initial_version, target_version) {}

  static TensorProto_DataType get_input_type(Node *node) {
    ONNX_ASSERTM(node->inputs().size() == 1,
                 "MeanVarianceNormalization must have exactly one input.")
    ONNX_ASSERTM(node->outputs().size() == 1,
                 "MeanVarianceNormalization must have exactly one output.")
    ONNX_ASSERTM(node->inputs()[0]->node()->kind() != kUndefined,
                 "MeanVarianceNormalization requires a nonempty input.")
    const auto elem_type = node->inputs()[0]->elemType();
    const auto input_type = static_cast<TensorProto_DataType>(
        elem_type == TensorProto_DataType_UNDEFINED ? node->output()->elemType() : elem_type);
    ONNX_ASSERTM(
        input_type == TensorProto_DataType_FLOAT16 || input_type == TensorProto_DataType_FLOAT ||
            input_type == TensorProto_DataType_DOUBLE ||
            input_type == TensorProto_DataType_BFLOAT16,
        "MeanVarianceNormalization conversion requires a known floating-point input type, got ",
        input_type, ". Annotate the input or output tensor type before version conversion.")
    return input_type;
  }

  static Node *make_float_constant(const std::shared_ptr<Graph> &graph, Node *before, float value) {
    Tensor tensor;
    tensor.elem_type() = TensorProto_DataType_FLOAT;
    tensor.sizes() = {};
    tensor.floats() = {value};
    Node *constant = graph->create(kConstant);
    constant->t_(kvalue, tensor);
    constant->insertBefore(before);
    constant->output()->setElemType(TensorProto_DataType_FLOAT);
    return constant;
  }

  static Node *make_float_reference_constant(const std::shared_ptr<Graph> &graph, Node *before,
                                             const std::string &ref_attr_name) {
    Node *constant = graph->create(kConstant);
    AttributeProto value;
    value.set_name("value_float");
    value.set_type(AttributeProto::FLOAT);
    value.set_ref_attr_name(ref_attr_name);
    constant->ref_(Symbol("value_float"), value);
    constant->insertBefore(before);
    constant->output()->setElemType(TensorProto_DataType_FLOAT);
    return constant;
  }

  static Node *make_axes_constant(const std::shared_ptr<Graph> &graph, Node *before,
                                  const std::vector<int64_t> &axes,
                                  const std::string &ref_attr_name = {}, bool linked = false) {
    Node *constant = graph->create(kConstant);
    if (linked) {
      AttributeProto value;
      value.set_name("value_ints");
      value.set_type(AttributeProto::INTS);
      value.set_ref_attr_name(ref_attr_name);
      constant->ref_(Symbol("value_ints"), value);
    } else {
      Tensor tensor;
      tensor.elem_type() = TensorProto_DataType_INT64;
      tensor.sizes() = {static_cast<int64_t>(axes.size())};
      tensor.int64s() = axes;
      constant->t_(kvalue, tensor);
    }
    constant->insertBefore(before);
    constant->output()->setElemType(TensorProto_DataType_INT64);
    return constant;
  }

  static Value *cast_to(const std::shared_ptr<Graph> &graph, Node *before, Value *input,
                        TensorProto_DataType elem_type) {
    if (input->elemType() == elem_type)
      return input;
    Node *cast = graph->create(kCast);
    cast->i_(kto, elem_type);
    cast->addInput(input);
    cast->insertBefore(before);
    cast->output()->setElemType(elem_type);
    return cast->output();
  }

  static Node *make_binary(const std::shared_ptr<Graph> &graph, Node *before, Symbol op_type,
                           Value *lhs, Value *rhs, TensorProto_DataType elem_type) {
    Node *result = graph->create(op_type);
    result->addInput(lhs);
    result->addInput(rhs);
    result->insertBefore(before);
    result->output()->setElemType(elem_type);
    return result;
  }

  static Node *make_reduce_mean(const std::shared_ptr<Graph> &graph, Node *before, Value *input,
                                Value *axes, TensorProto_DataType elem_type) {
    Node *result = graph->create(kReduceMean);
    result->addInput(input);
    result->addInput(axes);
    result->insertBefore(before);
    result->output()->setElemType(elem_type);
    return result;
  }

  static Node *make_sqrt(const std::shared_ptr<Graph> &graph, Node *before, Value *input,
                         TensorProto_DataType elem_type) {
    Node *result = graph->create(kSqrt);
    result->addInput(input);
    result->insertBefore(before);
    result->output()->setElemType(elem_type);
    return result;
  }

  static Node *replace_node(Node *node, Node *replacement) {
    Value *old_output = node->output();
    Value *new_output = replacement->output();
    Graph *graph = old_output->owningGraph();
    const std::string original_name = old_output->uniqueName();
    if (old_output->elemType() != TensorProto_DataType_UNDEFINED)
      new_output->setElemType(old_output->elemType());
    if (old_output->has_sizes())
      new_output->setSizes(old_output->sizes());
    else
      new_output->wipeSizes();
    if (old_output->has_doc_string())
      new_output->setDocString(old_output->docString());
    // Preserve capture names without renaming bindings shadowed in descendant graphs.
    old_output->setUniqueName(graph->getNextUniqueName(), false);
    new_output->setUniqueName(original_name, false);
    const auto direct_uses = old_output->uses();
    for (const auto &use : direct_uses)
      use.user->replaceInput(use.offset, new_output);
    node->destroy();
    return replacement;
  }
};

class MeanVarianceNormalization_28_29 final : public MeanVarianceNormalizationAdapter {
public:
  MeanVarianceNormalization_28_29() : MeanVarianceNormalizationAdapter(OpSetID(28), OpSetID(29)) {}

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    const auto input_type = get_input_type(node);
    const bool has_axes = node->hasAttribute(kaxes);
    const bool linked_axes = has_axes && node->kindOf(kaxes) == AttributeKind::ref;
    const std::string axes_ref = linked_axes ? node->ref(kaxes).ref_attr_name() : "";
    const std::vector<int64_t> axes =
        !has_axes || linked_axes ? std::vector<int64_t>{0, 2, 3} : node->is(kaxes);
    const auto compute_type =
        input_type == TensorProto_DataType_FLOAT16 || input_type == TensorProto_DataType_BFLOAT16
            ? TensorProto_DataType_FLOAT
            : input_type;
    Node *axes_constant = make_axes_constant(graph, node, axes, axes_ref, linked_axes);
    Value *input = cast_to(graph, node, node->inputs()[0], compute_type);
    Value *epsilon =
        cast_to(graph, node, make_float_constant(graph, node, 1e-9f)->output(), compute_type);
    Node *mean = make_reduce_mean(graph, node, input, axes_constant->output(), compute_type);
    Node *mean_squared =
        make_binary(graph, node, kMul, mean->output(), mean->output(), compute_type);
    Node *input_squared = make_binary(graph, node, kMul, input, input, compute_type);
    Node *squared_mean = make_reduce_mean(graph, node, input_squared->output(),
                                          axes_constant->output(), compute_type);
    Node *variance = make_binary(graph, node, kSub, squared_mean->output(), mean_squared->output(),
                                 compute_type);
    Node *stddev = make_sqrt(graph, node, variance->output(), compute_type);
    Node *centered = make_binary(graph, node, kSub, input, mean->output(), compute_type);
    Node *denominator = make_binary(graph, node, kAdd, stddev->output(), epsilon, compute_type);
    Node *normalized =
        make_binary(graph, node, kDiv, centered->output(), denominator->output(), compute_type);
    return replace_node(node, cast_to(graph, node, normalized->output(), input_type)->node());
  }
};

class MeanVarianceNormalization_29_28 final : public MeanVarianceNormalizationAdapter {
public:
  MeanVarianceNormalization_29_28() : MeanVarianceNormalizationAdapter(OpSetID(29), OpSetID(28)) {}

  Node *adapt(std::shared_ptr<Graph> graph, Node *node) const override {
    const auto input_type = get_input_type(node);
    const auto compute_type =
        input_type == TensorProto_DataType_FLOAT16 || input_type == TensorProto_DataType_BFLOAT16
            ? TensorProto_DataType_FLOAT
            : input_type;
    const bool has_axes = node->hasAttribute(kaxes);
    const bool linked_axes = has_axes && node->kindOf(kaxes) == AttributeKind::ref;
    const std::string axes_ref = linked_axes ? node->ref(kaxes).ref_attr_name() : "";
    const std::vector<int64_t> axes =
        !has_axes || linked_axes ? std::vector<int64_t>{0, 2, 3} : node->is(kaxes);
    const bool linked_epsilon =
        node->hasAttribute(kepsilon) && node->kindOf(kepsilon) == AttributeKind::ref;
    const std::string epsilon_ref = linked_epsilon ? node->ref(kepsilon).ref_attr_name() : "";
    const float epsilon_value = node->hasAttribute(kepsilon) && !linked_epsilon
                                    ? static_cast<float>(node->f(kepsilon))
                                    : 1e-9f;
    Node *axes_constant = make_axes_constant(graph, node, axes, axes_ref, linked_axes);
    Value *input = cast_to(graph, node, node->inputs()[0], compute_type);
    Node *epsilon_constant = linked_epsilon
                                 ? make_float_reference_constant(graph, node, epsilon_ref)
                                 : make_float_constant(graph, node, epsilon_value);
    Value *epsilon = cast_to(graph, node, epsilon_constant->output(), compute_type);
    Node *mean = make_reduce_mean(graph, node, input, axes_constant->output(), compute_type);
    Node *centered = make_binary(graph, node, kSub, input, mean->output(), compute_type);
    Node *centered_squared =
        make_binary(graph, node, kMul, centered->output(), centered->output(), compute_type);
    Node *variance = make_reduce_mean(graph, node, centered_squared->output(),
                                      axes_constant->output(), compute_type);
    Node *stddev = make_sqrt(graph, node, variance->output(), compute_type);
    Node *denominator = make_binary(graph, node, kAdd, stddev->output(), epsilon, compute_type);
    Node *normalized =
        make_binary(graph, node, kDiv, centered->output(), denominator->output(), compute_type);
    return replace_node(node, cast_to(graph, node, normalized->output(), input_type)->node());
  }
};

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion

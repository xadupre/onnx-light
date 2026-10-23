// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/optional/shape_optional.h"

#include <algorithm>
#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::optional {

namespace {

// Returns a pointer to the first attribute named ``name`` on ``node``,
// or ``nullptr`` if no such attribute exists.
const AttributeProto *FindOptionalAttribute(const NodeProto &node, const char *name) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attr = node.attribute(i);
    if (attr.ref_name() == name) {
      return &attr;
    }
  }
  return nullptr;
}

TypeProto SequenceTypeFromDescriptor(const SymSequence &sequence) {
  TypeProto type;
  TypeProto::Tensor *tensor =
      type.mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  tensor->set_elem_type(static_cast<int>(TensorTypeToDataType(sequence.ElemDtype())));
  if (sequence.HasElemShapes() && !sequence.ElemShapes().empty() &&
      std::all_of(sequence.ElemShapes().begin(), sequence.ElemShapes().end(),
                  [&](const SymShape &shape) { return shape == sequence.ElemShapes().front(); })) {
    TensorShapeProto *shape = tensor->add_shape();
    for (const SymDim &dimension : sequence.ElemShapes().front().Dims()) {
      auto *dim = shape->add_dim();
      if (dimension.IsInt()) {
        dim->set_dim_value(dimension.AsInt());
      } else if (dimension.IsExpr()) {
        dim->set_dim_param(dimension.AsExpr());
      }
    }
  }
  return type;
}

} // namespace

void ComputeShapeOptional(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Optional", "ComputeShapeOptional");

  const int num_inputs = node.input_size();
  EXT_ENFORCE_INVALID(num_inputs <= 1,
                      "ComputeShapeOptional: op 'Optional' expects at most 1 input, got ",
                      std::to_string(num_inputs), ".");

  if (num_inputs == 1) {
    // Copy the descriptor of the wrapped input value. ``SymTensor`` does
    // not model optional types, so the wrapping itself is elided and the
    // output is described by the same dtype/shape as the input.
    const std::string input_name = node.input(0);
    EXT_ENFORCE_INVALID(!(input_name.empty()),
                        "ComputeShapeOptional: input name of op 'Optional' must not be empty.");
    if (ctx.HasSequence(input_name)) {
      const SymSequence sequence(ctx.GetSequence(input_name));
      if (ctx.HasType(input_name) || sequence.HasElemDtype()) {
        const TypeProto input_type = ctx.HasType(input_name) ? ctx.GetType(input_name)
                                                             : SequenceTypeFromDescriptor(sequence);
        TypeProto optional_type;
        optional_type.mutable_optional_type()->ref_elem_type() = input_type;
        ctx.SetSequenceWithType(node.output(0), SymSequence(sequence), optional_type);
      } else {
        ctx.SetSequence(node.output(0), SymSequence(sequence));
      }
    } else if (!ctx.Has(input_name) && ctx.HasType(input_name)) {
      TypeProto type;
      type.mutable_optional_type()->ref_elem_type() = ctx.GetType(input_name);
      ctx.SetType(node.output(0), type);
    } else {
      const SymTensor &in = ctx.Get(input_name);
      ctx.Set(node.output(0), SymTensor(nullptr, in.Dtype(), SymShape(in.Shape().Dims())));
    }
    return;
  }

  // No input: the output element type must come from the ``type`` attribute.
  const AttributeProto *type_attr = FindOptionalAttribute(node, "type");
  EXT_ENFORCE_INVALID(type_attr != nullptr && type_attr->has_tp(),
                      "ComputeShapeOptional: op 'Optional' with no input must carry a 'type' "
                      "TypeProto attribute describing the wrapped element type.");
  const TypeProto &tp = type_attr->tp();
  // The ``type`` attribute may either be a bare tensor type (older
  // models / shorthand) or an optional-of-tensor type (the typical
  // shape). Sequence-of-tensor elements are also supported.
  const TypeProto *elem_tp = &tp;
  if (tp.has_optional_type()) {
    elem_tp = &tp.optional_type().elem_type();
  }
  EXT_ENFORCE_INVALID(
      elem_tp->has_tensor_type() ||
          (elem_tp->has_sequence_type() && elem_tp->sequence_type().elem_type().has_tensor_type()),
      "ComputeShapeOptional: the 'type' attribute must describe a tensor or sequence of tensors.");
  TypeProto type;
  type.mutable_optional_type()->ref_elem_type() = *elem_tp;
  ctx.SetType(node.output(0), type);
}

void ComputeShapeOptionalGetElement(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "OptionalGetElement", "ComputeShapeOptionalGetElement");

  EXT_ENFORCE_INVALID(
      node.input_size() == 1,
      "ComputeShapeOptionalGetElement: op 'OptionalGetElement' expects exactly 1 input, got ",
      std::to_string(node.input_size()), ".");

  const std::string input_name = node.input(0);
  EXT_ENFORCE_INVALID(
      !input_name.empty(),
      "ComputeShapeOptionalGetElement: input name of op 'OptionalGetElement' must not be empty.");

  // ``SymTensor`` does not model optional types, so the wrapping (if
  // any) is elided and the output descriptor mirrors the input
  // descriptor. The input may be either a tensor or a sequence (the
  // latter is supported by the schema since opset 15 for
  // optional<sequence> and since opset 18 for bare sequences).
  if (ctx.HasSequence(input_name)) {
    const SymSequence &seq = ctx.GetSequence(input_name);
    SymSequence output_sequence =
        seq.HasElemShapes() ? SymSequence(seq.ElemDtype(), std::vector<SymShape>(seq.ElemShapes()))
                            : SymSequence(seq.ElemDtype(), SymDim(seq.Length()));
    if (ctx.HasType(input_name)) {
      const TypeProto &input_type = ctx.GetType(input_name);
      const TypeProto &output_type =
          input_type.has_optional_type() ? input_type.optional_type().elem_type() : input_type;
      ctx.SetSequenceWithType(node.output(0), std::move(output_sequence), output_type);
    } else {
      ctx.SetSequence(node.output(0), std::move(output_sequence));
    }
    return;
  }

  if (!ctx.Has(input_name) && ctx.HasType(input_name)) {
    const TypeProto &type = ctx.GetType(input_name);
    ctx.SetType(node.output(0), type.has_optional_type() ? type.optional_type().elem_type() : type);
    return;
  }
  const SymTensor &in = ctx.Get(input_name);
  ctx.Set(node.output(0), SymTensor(nullptr, in.Dtype(), SymShape(in.Shape().Dims())));
}

void ComputeShapeOptionalHasElement(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "OptionalHasElement", "ComputeShapeOptionalHasElement");

  EXT_ENFORCE_INVALID(
      node.input_size() <= 1,
      "ComputeShapeOptionalHasElement: op 'OptionalHasElement' expects at most 1 input, got ",
      std::to_string(node.input_size()), ".");

  // The output is always a scalar boolean tensor regardless of the
  // input. The input itself is not consulted: its presence/absence is a
  // runtime property and the output shape does not depend on the input
  // dtype or shape.
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kBool, SymShape{}));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::optional

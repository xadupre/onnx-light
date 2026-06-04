// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/optional/shape_optional.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace optional {

namespace {

// Reads a TypeProto's tensor element type and shape into an
// ``OptimTensor`` descriptor. The TypeProto must wrap a tensor type
// (sequence / map / sparse types are rejected since ``OptimTensor``
// does not model them).
OptimTensor OptimTensorFromTensorTypeProto(const TypeProto &tp) {
  EXT_ENFORCE_INVALID(tp.has_tensor_type(),
                      "ComputeShapeOptional: the 'type' attribute must wrap a tensor type; "
                      "sequence, map and sparse element types are not supported.");
  const TypeProto::Tensor &tt = tp.tensor_type();
  const TensorType dtype = DataTypeToTensorType(tt.elem_type());
  OptimShape shape;
  if (tt.has_shape()) {
    const TensorShapeProto &sp = tt.shape();
    for (int i = 0; i < sp.dim().size(); ++i) {
      const TensorShapeProto::Dimension &d = sp.dim()[i];
      if (d.has_dim_value()) {
        shape.PushBack(OptimDim(static_cast<int64_t>(d.dim_value())));
      } else if (d.has_dim_param()) {
        shape.PushBack(OptimDim(std::string(d.dim_param().data(), d.dim_param().size())));
      } else {
        // Unknown dim: use a fresh symbolic placeholder.
        shape.PushBack(OptimDim(std::string("?")));
      }
    }
  }
  return OptimTensor(nullptr, dtype, std::move(shape));
}

// Returns a pointer to the first attribute named ``name`` on ``node``,
// or ``nullptr`` if no such attribute exists.
const AttributeProto *FindAttribute(const NodeProto &node, const char *name) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attr = node.attribute(i);
    if (attr.ref_name() == name) {
      return &attr;
    }
  }
  return nullptr;
}

} // namespace

void ComputeShapeOptional(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Optional", "ComputeShapeOptional");

  const int num_inputs = node.input_size();
  EXT_ENFORCE_INVALID(num_inputs <= 1,
                      "ComputeShapeOptional: op 'Optional' expects at most 1 input, got " +
                          std::to_string(num_inputs) + ".");

  if (num_inputs == 1) {
    // Copy the descriptor of the wrapped input value. ``OptimTensor`` does
    // not model optional types, so the wrapping itself is elided and the
    // output is described by the same dtype/shape as the input.
    const std::string input_name = node.input(0).as_string();
    if (input_name.empty()) {
      throw std::invalid_argument(
          "ComputeShapeOptional: input name of op 'Optional' must not be empty.");
    }
    const OptimTensor &in = ctx.Get(input_name);
    ctx.Set(node.output(0), OptimTensor(nullptr, in.Dtype(), OptimShape(in.Shape().Dims())));
    return;
  }

  // No input: the output element type must come from the ``type`` attribute.
  const AttributeProto *type_attr = FindAttribute(node, "type");
  EXT_ENFORCE_INVALID(type_attr != nullptr && type_attr->has_tp(),
                      "ComputeShapeOptional: op 'Optional' with no input must carry a 'type' "
                      "TypeProto attribute describing the wrapped element type.");
  const TypeProto &tp = type_attr->tp();
  // The ``type`` attribute may either be a bare tensor type (older
  // models / shorthand) or an optional-of-tensor type (the typical
  // shape). Both are accepted; sequence-of-tensor and sparse types
  // are rejected.
  const TypeProto *elem_tp = &tp;
  if (tp.has_optional_type()) {
    elem_tp = &tp.optional_type().elem_type();
  } else
    EXT_ENFORCE_INVALID(
        !tp.has_sequence_type() && !tp.has_sparse_tensor_type() && !tp.has_map_type(),
        "ComputeShapeOptional: the 'type' attribute must wrap a tensor or an "
        "optional-of-tensor type; sequence, map and sparse types are not supported.");
  ctx.Set(node.output(0), OptimTensorFromTensorTypeProto(*elem_tp));
}

void ComputeShapeOptionalGetElement(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "OptionalGetElement", "ComputeShapeOptionalGetElement");

  EXT_ENFORCE_INVALID(
      node.input_size() == 1,
      "ComputeShapeOptionalGetElement: op 'OptionalGetElement' expects exactly 1 input, got " +
          std::to_string(node.input_size()) + ".");

  const std::string input_name = node.input(0).as_string();
  EXT_ENFORCE_INVALID(
      !input_name.empty(),
      "ComputeShapeOptionalGetElement: input name of op 'OptionalGetElement' must not be empty.");

  // ``OptimTensor`` does not model optional types, so the wrapping (if
  // any) is elided and the output descriptor mirrors the input
  // descriptor. The input may be either a tensor or a sequence (the
  // latter is supported by the schema since opset 15 for
  // optional<sequence> and since opset 18 for bare sequences).
  if (ctx.HasSequence(input_name)) {
    const OptimSequence &seq = ctx.GetSequence(input_name);
    if (seq.HasElemShapes()) {
      ctx.SetSequence(node.output(0),
                      OptimSequence(seq.ElemDtype(), std::vector<OptimShape>(seq.ElemShapes())));
    } else {
      ctx.SetSequence(node.output(0), OptimSequence(seq.ElemDtype(), OptimDim(seq.Length())));
    }
    return;
  }

  const OptimTensor &in = ctx.Get(input_name);
  ctx.Set(node.output(0), OptimTensor(nullptr, in.Dtype(), OptimShape(in.Shape().Dims())));
}

void ComputeShapeOptionalHasElement(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "OptionalHasElement", "ComputeShapeOptionalHasElement");

  EXT_ENFORCE_INVALID(
      node.input_size() <= 1,
      "ComputeShapeOptionalHasElement: op 'OptionalHasElement' expects at most 1 input, got " +
          std::to_string(node.input_size()) + ".");

  // The output is always a scalar boolean tensor regardless of the
  // input. The input itself is not consulted: its presence/absence is a
  // runtime property and the output shape does not depend on the input
  // dtype or shape.
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kBool, OptimShape{}));
}

} // namespace optional
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE

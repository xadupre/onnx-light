// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

void ComputeShapeConstantOfShape(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ConstantOfShape", "ComputeShapeConstantOfShape");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeConstantOfShape: ConstantOfShape requires one input (shape).");

  // Output element type: from the ``value`` attribute when present,
  // otherwise float32 (per the schema default).
  TensorType dtype = TensorType::kFloat;
  const AttributeProto *value = FindAttribute(node, "value");
  if (value != nullptr) {
    EXT_ENFORCE_INVALID(
        value->has_t(),
        "ComputeShapeConstantOfShape: attribute 'value' must carry a tensor value.");
    dtype = DataTypeToTensorType(value->t().data_type());
  }

  const OptimTensor &shape_input = ctx.Get(node.input(0).as_string());

  OptimShape out_shape;
  if (shape_input.HasValueAsShape()) {
    out_shape = shape_input.ValueAsShape();
  } else if (shape_input.Shape().Rank() == 1 && shape_input.Shape()[0].IsInt()) {
    // Fall back to producing the right rank with symbolic dims when the
    // input value has not been data-propagated.
    const int64_t rank = shape_input.Shape()[0].AsInt();
    for (int64_t i = 0; i < rank; ++i) {
      out_shape.PushBack(OptimDim("ConstantOfShape_dim" + std::to_string(i)));
    }
  } else if (shape_input.Shape().Rank() == 1) {
    // Single, symbolic dim: rank is unknown, fall back to one symbolic dim.
    out_shape.PushBack(OptimDim("ConstantOfShape_dim0"));
  }
  // ``input`` could legitimately be the empty 1-D tensor, which makes
  // the output a scalar (no dims). Leaving ``out_shape`` empty handles
  // that case.

  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE

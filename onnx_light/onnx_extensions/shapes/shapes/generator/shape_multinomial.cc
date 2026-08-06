// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/generator/shape_generator.h"

#include <cstdint>
#include <string>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator {

void ComputeShapeMultinomial(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Multinomial", "ComputeShapeMultinomial");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeMultinomial: Multinomial requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));

  // Output element type: from the ``dtype`` attribute when present
  // (must be INT32 or INT64), otherwise defaults to INT32 per the schema.
  TensorType out_dtype = TensorType::kInt32;
  const AttributeProto *dtype_attr = FindAttribute(node, "dtype");
  if (dtype_attr != nullptr) {
    const int64_t dtype_value = dtype_attr->i();
    out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype_value));
    EXT_ENFORCE_INVALID(!(out_dtype != TensorType::kInt32 && out_dtype != TensorType::kInt64),
                        "ComputeShapeMultinomial: attribute 'dtype' must be INT32 or "
                        "INT64, got ",
                        dtype_value, ".");
  }

  const int64_t sample_size = GetAttributeOr<int64_t>(node, "sample_size", 1);
  EXT_ENFORCE_INVALID(sample_size >= 0,
                      "ComputeShapeMultinomial: attribute 'sample_size' must be non-negative, "
                      "got ",
                      std::to_string(sample_size), ".");

  // Output shape is [batch_size, sample_size]. ``batch_size`` is taken
  // from the input shape's first dim when the input has a known rank;
  // otherwise it is left as a default (zero-valued) symbolic dim. When the
  // input rank is statically known it must be 2, per the schema.
  SymShape out_shape;
  const SymShape &input_shape = input.Shape();
  if (input_shape.Rank() != 0) {
    EXT_ENFORCE_INVALID(input_shape.Rank() == 2,
                        "ComputeShapeMultinomial: input must be rank 2, got rank ",
                        std::to_string(input_shape.Rank()), ".");
    out_shape.PushBack(input_shape[0]);
  } else {
    out_shape.PushBack(SymDim());
  }
  out_shape.PushBack(SymDim(sample_size));

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator

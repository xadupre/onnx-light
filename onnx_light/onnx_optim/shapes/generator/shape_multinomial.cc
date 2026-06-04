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

void ComputeShapeMultinomial(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Multinomial", "ComputeShapeMultinomial");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeMultinomial: Multinomial requires one input.");

  const OptimTensor &input = ctx.Get(node.input(0).as_string());

  // Output element type: from the ``dtype`` attribute when present
  // (must be INT32 or INT64), otherwise defaults to INT32 per the schema.
  TensorType out_dtype = TensorType::kInt32;
  const AttributeProto *dtype_attr = FindAttribute(node, "dtype");
  if (dtype_attr != nullptr) {
    const int64_t dtype_value = dtype_attr->i();
    out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype_value));
    if (out_dtype != TensorType::kInt32 && out_dtype != TensorType::kInt64) {
      throw std::invalid_argument("ComputeShapeMultinomial: attribute 'dtype' must be INT32 or "
                                  "INT64, got " +
                                  std::to_string(dtype_value) + ".");
    }
  }

  const int64_t sample_size = GetAttributeOr<int64_t>(node, "sample_size", 1);
  EXT_ENFORCE_INVALID(sample_size >= 0,
                      "ComputeShapeMultinomial: attribute 'sample_size' must be non-negative, "
                      "got " +
                          std::to_string(sample_size) + ".");

  // Output shape is [batch_size, sample_size]. ``batch_size`` is taken
  // from the input shape's first dim when the input has a known rank;
  // otherwise it is left as a default (zero-valued) symbolic dim. When the
  // input rank is statically known it must be 2, per the schema.
  OptimShape out_shape;
  const OptimShape &input_shape = input.Shape();
  if (input_shape.Rank() != 0) {
    EXT_ENFORCE_INVALID(input_shape.Rank() == 2,
                        "ComputeShapeMultinomial: input must be rank 2, got rank " +
                            std::to_string(input_shape.Rank()) + ".");
    out_shape.PushBack(input_shape[0]);
  } else {
    out_shape.PushBack(OptimDim());
  }
  out_shape.PushBack(OptimDim(sample_size));

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE

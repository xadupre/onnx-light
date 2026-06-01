// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

void ComputeShapeHammingWindow(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "HammingWindow", "ComputeShapeHammingWindow");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeHammingWindow: HammingWindow requires one input (size).");

  // Output element type: from the ``output_datatype`` attribute when
  // present, otherwise float32 (per the schema default).
  TensorType dtype = TensorType::kFloat;
  const int64_t output_datatype =
      GetAttributeOr<int64_t>(node, "output_datatype", static_cast<int64_t>(TensorProto::FLOAT));
  dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(output_datatype));

  const OptimTensor &size_input = ctx.Get(node.input(0).as_string());

  OptimShape out_shape;
  if (size_input.HasValueAsShape() && size_input.ValueAsShape().Rank() == 1 &&
      size_input.ValueAsShape()[0].IsInt()) {
    // ``size`` is a known constant integer scalar: produce a concrete
    // 1-D output dim.
    out_shape.PushBack(size_input.ValueAsShape()[0]);
  } else {
    // Unknown ``size``: produce a single symbolic dim.
    out_shape.PushBack(OptimDim("HammingWindow_dim0"));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/generator/shape_generator.h"

#include <cstdint>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator {

void ComputeShapeBlackmanWindow(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "BlackmanWindow", "ComputeShapeBlackmanWindow");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeBlackmanWindow: BlackmanWindow requires one input (size).");

  // Output element type: from the ``output_datatype`` attribute when
  // present, otherwise float32 (per the schema default).
  TensorType dtype = TensorType::kFloat;
  const int64_t output_datatype =
      GetAttributeOr<int64_t>(node, "output_datatype", static_cast<int64_t>(TensorProto::FLOAT));
  dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(output_datatype));

  const SymTensor &size_input = ctx.Get(node.input(0));

  SymShape out_shape;
  if (size_input.HasValueAsShape() && size_input.ValueAsShape().Rank() == 1 &&
      size_input.ValueAsShape()[0].IsInt()) {
    // ``size`` is a known constant integer scalar: produce a concrete
    // 1-D output dim.
    out_shape.PushBack(size_input.ValueAsShape()[0]);
  } else {
    // Unknown ``size``: produce a single symbolic dim.
    out_shape.PushBack(SymDim("BlackmanWindow_dim0"));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator

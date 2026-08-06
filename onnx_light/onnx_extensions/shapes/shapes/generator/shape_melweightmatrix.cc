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

void ComputeShapeMelWeightMatrix(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "MelWeightMatrix", "ComputeShapeMelWeightMatrix");
  EXT_ENFORCE_INVALID(node.input_size() >= 2,
                      "ComputeShapeMelWeightMatrix: MelWeightMatrix requires at least "
                      "num_mel_bins and dft_length inputs.");

  // Output element type: from the ``output_datatype`` attribute when
  // present, otherwise float32 (per the schema default).
  const int64_t output_datatype =
      GetAttributeOr<int64_t>(node, "output_datatype", static_cast<int64_t>(TensorProto::FLOAT));
  const TensorType dtype =
      DataTypeToTensorType(static_cast<TensorProto::DataType>(output_datatype));

  // First output dim: floor(dft_length / 2) + 1 when ``dft_length`` is a
  // known constant scalar; symbolic otherwise.
  const SymTensor &num_mel_bins_input = ctx.Get(node.input(0));
  const SymTensor &dft_length_input = ctx.Get(node.input(1));

  SymShape out_shape;
  if (dft_length_input.HasValueAsShape() && dft_length_input.ValueAsShape().Rank() == 1 &&
      dft_length_input.ValueAsShape()[0].IsInt()) {
    const int64_t dft_length_value = dft_length_input.ValueAsShape()[0].AsInt();
    out_shape.PushBack(SymDim(dft_length_value / 2 + 1));
  } else {
    out_shape.PushBack(SymDim("MelWeightMatrix_dim0"));
  }

  if (num_mel_bins_input.HasValueAsShape() && num_mel_bins_input.ValueAsShape().Rank() == 1 &&
      num_mel_bins_input.ValueAsShape()[0].IsInt()) {
    out_shape.PushBack(num_mel_bins_input.ValueAsShape()[0]);
  } else {
    out_shape.PushBack(SymDim("MelWeightMatrix_dim1"));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator

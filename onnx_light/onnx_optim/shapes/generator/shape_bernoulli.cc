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

void ComputeShapeBernoulli(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Bernoulli", "ComputeShapeBernoulli");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeBernoulli: Bernoulli requires one input.");

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  OptimShape out_shape = input.Shape();

  // Output element type: from the ``dtype`` attribute when present,
  // otherwise the input's dtype (per the schema).
  TensorType out_dtype = input.Dtype();
  const AttributeProto *dtype_attr = FindAttribute(node, "dtype");
  if (dtype_attr != nullptr) {
    const int64_t dtype_value = dtype_attr->i();
    out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype_value));
    if (out_dtype == TensorType::kUndefined) {
      throw std::invalid_argument("ComputeShapeBernoulli: attribute 'dtype' has unsupported "
                                  "value " +
                                  std::to_string(dtype_value) + ".");
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE

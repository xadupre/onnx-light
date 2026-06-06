// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor Scatter::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                           const Attributes &attrs) const {
  Tensor out("", data.data_type, data.shape, data.data);
  (*this)(data, indices, updates, attrs, out);
  return out;
}

void Scatter::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                         const Attributes &attrs, Tensor &output) const {
  // ``Scatter`` (opset 9, deprecated since opset 11) is semantically
  // equivalent to ``ScatterElements`` with ``reduction="none"``; delegate to
  // that kernel to avoid duplicating the reference implementation.
  const ScatterElements se{this->ctx_};
  ScatterElements::Attributes se_attrs;
  se_attrs.axis = attrs.axis;
  se_attrs.reduction = "none";
  se(data, indices, updates, se_attrs, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

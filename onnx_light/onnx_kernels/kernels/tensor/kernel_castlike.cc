// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor CastLike::operator()(const Tensor &x, const Tensor &target_type) const {
  // ``CastLike`` is defined by the spec as ``Cast`` with the ``to`` attribute
  // taken from the second input's element type. The values of ``target_type``
  // are ignored — only its ``data_type`` is observed.
  const Cast cast_kernel{ctx_};
  return cast_kernel(x, target_type.data_type);
}

void CastLike::operator()(const Tensor &x, const Tensor &target_type, Tensor &output) const {
  const Cast cast_kernel{ctx_};
  cast_kernel(x, target_type.data_type, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

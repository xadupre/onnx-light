// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor If::operator()(const Tensor &cond, const Tensor &then_value,
                      const Tensor &else_value) const {
  if (cond.data_type != TensorProto::DataType::BOOL) {
    throw std::invalid_argument("kernel::If: 'cond' must be a BOOL tensor.");
  }
  if (cond.element_count() != 1) {
    throw std::invalid_argument("kernel::If: 'cond' must contain a single element.");
  }
  if (then_value.data_type != else_value.data_type) {
    throw std::invalid_argument(
        "kernel::If: 'then_value' and 'else_value' must have the same data type.");
  }
  if (then_value.shape != else_value.shape) {
    throw std::invalid_argument(
        "kernel::If: 'then_value' and 'else_value' must have the same shape.");
  }

  const bool taken = cond.data[0] != 0;
  const Tensor &src = taken ? then_value : else_value;
  return Tensor("", src.data_type, src.shape, src.data);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

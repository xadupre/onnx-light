// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"

#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor If::operator()(const Tensor &cond, const Tensor &then_value,
                      const Tensor &else_value) const {
  // Allocate an output matching either branch's type/shape (both must agree;
  // the in-place overload enforces this); the in-place overload writes into
  // ``out.data`` below.
  Tensor out("", then_value.data_type, then_value.shape,
             std::vector<uint8_t>(then_value.data.size()));
  (*this)(cond, then_value, else_value, &out);
  return out;
}

void If::operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value,
                    Tensor *output) const {
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
  if (output == nullptr) {
    throw std::invalid_argument("kernel::If requires a non-null preallocated output tensor.");
  }
  if (output->data_type != then_value.data_type) {
    throw std::invalid_argument(
        "kernel::If preallocated output must have the same data type as the branches.");
  }
  if (output->shape != then_value.shape) {
    throw std::invalid_argument(
        "kernel::If preallocated output shape must match the branch shape.");
  }
  if (output->data.size() != then_value.data.size()) {
    throw std::invalid_argument(
        "kernel::If preallocated output buffer has unexpected size in bytes.");
  }

  const bool taken = cond.data[0] != 0;
  const Tensor &src = taken ? then_value : else_value;
  if (!src.data.empty()) {
    std::memcpy(output->data.data(), src.data.data(), src.data.size());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

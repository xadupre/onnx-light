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
  (*this)(cond, then_value, else_value, out);
  return out;
}

void If::operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value,
                    Tensor &output) const {
  EXT_ENFORCE_INVALID(cond.data_type == TensorProto::DataType::BOOL,
                      "kernel::If: 'cond' must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(cond.element_count() == 1,
                      "kernel::If: 'cond' must contain a single element.");
  EXT_ENFORCE_INVALID(then_value.data_type == else_value.data_type,
                      "kernel::If: 'then_value' and 'else_value' must have the same data type.");
  EXT_ENFORCE_INVALID(then_value.shape == else_value.shape,
                      "kernel::If: 'then_value' and 'else_value' must have the same shape.");
  EXT_ENFORCE_INVALID(
      output.data_type == then_value.data_type,
      "kernel::If preallocated output must have the same data type as the branches.");
  EXT_ENFORCE_INVALID(output.shape == then_value.shape,
                      "kernel::If preallocated output shape must match the branch shape.");
  EXT_ENFORCE_INVALID(output.data.size() == then_value.data.size(),
                      "kernel::If preallocated output buffer has unexpected size in bytes.");

  const bool taken = cond.data[0] != 0;
  const Tensor &src = taken ? then_value : else_value;
  if (!src.data.empty()) {
    std::memcpy(output.data.data(), src.data.data(), src.data.size());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

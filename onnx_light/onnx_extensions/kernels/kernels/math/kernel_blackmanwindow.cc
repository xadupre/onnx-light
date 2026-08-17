// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor BlackmanWindow::operator()(const Tensor &size, bool periodic, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(size.data_type == DataType::INT32,
                      "kernel::BlackmanWindow expects an INT32 size tensor.");
  EXT_ENFORCE_INVALID(size.element_count() == 1 && size.shape.empty(),
                      "kernel::BlackmanWindow expects a scalar size tensor.");
  const int32_t n = size.AsInt32()[0];
  EXT_ENFORCE_INVALID(n >= 0, "kernel::BlackmanWindow size must be non-negative.");
  const size_t y_n_bytes = static_cast<size_t>(n) * sizeof(float);
  Tensor y = rt ? rt->MakeOutputTensor(0, DataType::FLOAT, {n}, y_n_bytes)
                : MakeOutputTensor(DataType::FLOAT, {n}, y_n_bytes, nullptr);
  (*this)(size, periodic, y);
  return y;
}

void BlackmanWindow::operator()(const Tensor &size, bool periodic, Tensor &output) const {
  EXT_ENFORCE_INVALID(size.data_type == DataType::INT32,
                      "kernel::BlackmanWindow expects an INT32 size tensor.");
  EXT_ENFORCE_INVALID(size.element_count() == 1 && size.shape.empty(),
                      "kernel::BlackmanWindow expects a scalar size tensor.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::BlackmanWindow preallocated output must be a FLOAT tensor.");

  const int32_t n = size.AsInt32()[0];
  EXT_ENFORCE_INVALID(n >= 0, "kernel::BlackmanWindow size must be non-negative.");
  EXT_ENFORCE_INVALID(periodic || n > 1,
                      "kernel::BlackmanWindow symmetric variant requires size > 1.");
  EXT_ENFORCE_INVALID(output.shape.size() == 1 && output.shape[0] == n,
                      "kernel::BlackmanWindow preallocated output shape must be {size}.");
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  EXT_ENFORCE_INVALID(
      output.size_bytes() == expected_bytes,
      "kernel::BlackmanWindow preallocated output buffer has unexpected size in bytes.");

  constexpr double kPi = 3.14159265358979323846;
  constexpr double a0 = 0.42;
  constexpr double a1 = -0.5;
  constexpr double a2 = 0.08;
  const double divisor = periodic ? static_cast<double>(n) : static_cast<double>(n - 1);

  float *py = output.AsFloat();
  for (int32_t i = 0; i < n; ++i) {
    const double k = static_cast<double>(i) / divisor;
    py[static_cast<size_t>(i)] =
        static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k) + a2 * std::cos(4.0 * kPi * k));
  }
}

void BlackmanWindow::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int64_t output_datatype =
      GetAttributeIntOrDefault(node, "output_datatype", static_cast<int64_t>(DataType::FLOAT));
  EXT_ENFORCE_INVALID(!(output_datatype != static_cast<int64_t>(DataType::FLOAT)),
                      "RunNode: op 'BlackmanWindow' only supports output_datatype=FLOAT.");
  const bool periodic = GetAttributeIntOrDefault(node, "periodic", 1) != 0;
  const Tensor &size = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(size, periodic, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

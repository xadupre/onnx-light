// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Element size in bytes for the four dtypes supported by ``Cast``. Returns
// 0 for unsupported dtypes so the caller can report an explicit error.
size_t SupportedCastElementSize(int32_t dtype) {
  switch (static_cast<TensorProto::DataType>(dtype)) {
  case TensorProto::DataType::FLOAT:
    return sizeof(float);
  case TensorProto::DataType::DOUBLE:
    return sizeof(double);
  case TensorProto::DataType::INT32:
    return sizeof(int32_t);
  case TensorProto::DataType::INT64:
    return sizeof(int64_t);
  default:
    return 0;
  }
}

// Returns one element of ``x`` (at flat index ``i``) widened to ``double``.
// Used as the canonical intermediate value because every supported
// dtype is exactly representable as ``double`` for the value ranges
// exercised by the backend test cases.
double LoadAsDouble(const Tensor &x, int64_t i) {
  switch (static_cast<TensorProto::DataType>(x.data_type)) {
  case TensorProto::DataType::FLOAT:
    return static_cast<double>(x.AsFloat()[i]);
  case TensorProto::DataType::DOUBLE:
    return x.AsDouble()[i];
  case TensorProto::DataType::INT32:
    return static_cast<double>(x.AsInt32()[i]);
  case TensorProto::DataType::INT64:
    return static_cast<double>(x.AsInt64()[i]);
  default:
    // SupportedCastElementSize would have rejected this dtype earlier.
    throw std::invalid_argument("kernel::Cast: unsupported input dtype.");
  }
}

// Writes the value ``v`` (already cast to the destination C++ type) into
// the output buffer at flat index ``i``.
void StoreFromDouble(Tensor &output, int64_t i, double v) {
  switch (static_cast<TensorProto::DataType>(output.data_type)) {
  case TensorProto::DataType::FLOAT:
    output.AsFloat()[i] = static_cast<float>(v);
    return;
  case TensorProto::DataType::DOUBLE:
    output.AsDouble()[i] = v;
    return;
  case TensorProto::DataType::INT32:
    output.AsInt32()[i] = static_cast<int32_t>(v);
    return;
  case TensorProto::DataType::INT64:
    output.AsInt64()[i] = static_cast<int64_t>(v);
    return;
  default:
    throw std::invalid_argument("kernel::Cast: unsupported output dtype.");
  }
}

} // namespace

Tensor Cast::operator()(const Tensor &x, int32_t to) const {
  const size_t out_elem = SupportedCastElementSize(to);
  EXT_ENFORCE_INVALID(out_elem != 0, "kernel::Cast: unsupported 'to' dtype " + std::to_string(to) +
                                         " (supported: FLOAT, DOUBLE, INT32, INT64).");
  Tensor out("", to, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * out_elem));
  (*this)(x, to, out);
  return out;
}

void Cast::operator()(const Tensor &x, int32_t to, Tensor &output) const {
  const size_t in_elem = SupportedCastElementSize(x.data_type);
  EXT_ENFORCE_INVALID(in_elem != 0, "kernel::Cast: unsupported input dtype " +
                                        std::to_string(x.data_type) +
                                        " (supported: FLOAT, DOUBLE, INT32, INT64).");
  const size_t out_elem = SupportedCastElementSize(to);
  EXT_ENFORCE_INVALID(out_elem != 0, "kernel::Cast: unsupported 'to' dtype " + std::to_string(to) +
                                         " (supported: FLOAT, DOUBLE, INT32, INT64).");
  EXT_ENFORCE_INVALID(output.data_type == to,
                      "kernel::Cast preallocated output dtype must match 'to'.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Cast preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * out_elem;
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::Cast preallocated output buffer has unexpected size in bytes.");

  // Fast path: identity cast — bitwise copy when input and output dtypes
  // match. This matches the ONNX semantics for ``to == input dtype`` and
  // avoids a needless double round-trip.
  if (x.data_type == to) {
    if (!x.data.empty()) {
      std::memcpy(output.data.data(), x.data.data(), x.data.size());
    }
    return;
  }

  for (int64_t i = 0; i < n; ++i) {
    StoreFromDouble(output, i, LoadAsDouble(x, i));
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

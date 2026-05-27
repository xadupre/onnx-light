// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/text/include_text_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Validates that both inputs are STRING tensors of equal shape or that one
// is a scalar (single element) — the same broadcasting flavor used by the
// existing element-wise binary helpers (And/Or/Xor). Returns the broadcasted
// shape, total element count, and the individual input element counts.
struct StringBroadcast {
  std::vector<int64_t> shape;
  int64_t element_count = 0;
  int64_t nx = 0;
  int64_t ny = 0;
};

StringBroadcast CheckStringConcatInputs(const Tensor &x, const Tensor &y) {
  EXT_ENFORCE_INVALID(!(x.data_type != static_cast<int32_t>(TensorProto::DataType::STRING) ||
                        y.data_type != static_cast<int32_t>(TensorProto::DataType::STRING)),
                      "kernel::StringConcat only supports STRING tensors.");
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  EXT_ENFORCE_INVALID(
      !(!(nx == ny || nx == 1 || ny == 1)),
      "kernel::StringConcat only supports equal-shape tensors or scalar broadcasting.");
  EXT_ENFORCE_INVALID(
      static_cast<int64_t>(x.string_data.size()) == nx,
      "kernel::StringConcat input ``x`` string_data size does not match its shape.");
  EXT_ENFORCE_INVALID(
      static_cast<int64_t>(y.string_data.size()) == ny,
      "kernel::StringConcat input ``y`` string_data size does not match its shape.");
  StringBroadcast bi;
  bi.nx = nx;
  bi.ny = ny;
  bi.element_count = nx >= ny ? nx : ny;
  bi.shape = nx >= ny ? x.shape : y.shape;
  return bi;
}

} // namespace

Tensor StringConcat::operator()(const Tensor &x, const Tensor &y) const {
  const StringBroadcast bi = CheckStringConcatInputs(x, y);
  Tensor out = Tensor::MakeString("", bi.shape,
                                  std::vector<std::string>(static_cast<size_t>(bi.element_count)));
  (*this)(x, y, out);
  return out;
}

void StringConcat::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  const StringBroadcast bi = CheckStringConcatInputs(x, y);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorProto::DataType::STRING),
                      "kernel::StringConcat preallocated output must be a STRING tensor.");
  EXT_ENFORCE_INVALID(output.shape == bi.shape,
                      "kernel::StringConcat preallocated output shape must match the "
                      "broadcasted input shape.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.string_data.size()) == bi.element_count,
                      "kernel::StringConcat preallocated output string_data has unexpected size.");
  for (int64_t i = 0; i < bi.element_count; ++i) {
    const std::string &a = x.string_data[bi.nx == 1 ? 0 : static_cast<size_t>(i)];
    const std::string &b = y.string_data[bi.ny == 1 ? 0 : static_cast<size_t>(i)];
    output.string_data[static_cast<size_t>(i)] = a + b;
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

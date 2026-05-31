// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

template <typename T> void ValidateInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "kernel::Scaler input data_type does not match the requested element T.");
}

void ValidateAttrs(const std::vector<float> &offset, const std::vector<float> &scale,
                   int64_t last_dim) {
  EXT_ENFORCE_INVALID(offset.size() == scale.size(),
                      "kernel::Scaler requires 'offset' and 'scale' to have the same length.");
  EXT_ENFORCE_INVALID(!offset.empty(),
                      "kernel::Scaler requires 'offset' and 'scale' to be non-empty.");
  EXT_ENFORCE_INVALID(offset.size() == 1u || static_cast<int64_t>(offset.size()) == last_dim,
                      "kernel::Scaler requires 'offset'/'scale' length to be 1 or to match the "
                      "size of the last dimension of the input.");
}

template <typename T>
void ApplyScaler(const Tensor &x, const std::vector<float> &offset, const std::vector<float> &scale,
                 float *out) {
  const T *px = x.As<T>();
  const int64_t n = x.element_count();
  const int64_t stride = static_cast<int64_t>(offset.size());
  if (stride == 1) {
    const float o = offset[0];
    const float s = scale[0];
    for (int64_t i = 0; i < n; ++i) {
      out[i] = (static_cast<float>(px[i]) - o) * s;
    }
  } else {
    for (int64_t i = 0; i < n; ++i) {
      const int64_t k = i % stride;
      out[i] = (static_cast<float>(px[i]) - offset[k]) * scale[k];
    }
  }
}

int64_t LastDim(const std::vector<int64_t> &shape) { return shape.empty() ? 1 : shape.back(); }

} // namespace

template <typename T>
Tensor Scaler::operator()(const Tensor &x, const std::vector<float> &offset,
                          const std::vector<float> &scale) const {
  ValidateInput<T>(x);
  ValidateAttrs(offset, scale, LastDim(x.shape));
  const int64_t n = x.element_count();
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(float));
  Tensor out("", TensorElementType<float>::value, x.shape, std::move(bytes));
  ApplyScaler<T>(x, offset, scale, reinterpret_cast<float *>(out.data.data()));
  return out;
}

template <typename T>
void Scaler::operator()(const Tensor &x, const std::vector<float> &offset,
                        const std::vector<float> &scale, Tensor &output) const {
  ValidateInput<T>(x);
  ValidateAttrs(offset, scale, LastDim(x.shape));
  EXT_ENFORCE_INVALID(output.data_type == TensorElementType<float>::value,
                      "kernel::Scaler preallocated output dtype must be float.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Scaler preallocated output shape must match the input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(x.element_count()) * sizeof(float),
                      "kernel::Scaler preallocated output buffer is incorrectly sized.");
  ApplyScaler<T>(x, offset, scale, output.As<float>());
}

// Explicit instantiations for the supported element types.
#define ONNX_LIGHT_INSTANTIATE_SCALER(T)                                                           \
  template Tensor Scaler::operator()<T>(const Tensor &, const std::vector<float> &,                \
                                        const std::vector<float> &) const;                         \
  template void Scaler::operator()<T>(const Tensor &, const std::vector<float> &,                  \
                                      const std::vector<float> &, Tensor &) const

ONNX_LIGHT_INSTANTIATE_SCALER(float);
ONNX_LIGHT_INSTANTIATE_SCALER(double);
ONNX_LIGHT_INSTANTIATE_SCALER(int64_t);
ONNX_LIGHT_INSTANTIATE_SCALER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_SCALER

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

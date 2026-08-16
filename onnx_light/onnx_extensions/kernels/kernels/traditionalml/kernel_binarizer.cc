// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

template <typename T> void ApplyThreshold(const Tensor &x, T threshold, T *out) {
  const T *px = x.As<T>();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) {
    out[i] = px[i] > threshold ? static_cast<T>(1) : static_cast<T>(0);
  }
}

template <typename T> void ValidateInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "kernel::Binarizer input data_type does not match the requested element T.");
}

} // namespace

template <typename T>
Tensor Binarizer::operator()(const Tensor &x, T threshold, RuntimeContext *rt) const {
  ValidateInput<T>(x);
  const int64_t n = x.element_count();
  Tensor out = MakeOutputTensor(TensorElementType<T>::value, x.shape,
                                static_cast<size_t>(n) * sizeof(T), ctx_.allocator);
  ApplyThreshold<T>(x, threshold, reinterpret_cast<T *>(out.mutable_bytes()));
  return out;
}

template <typename T>
void Binarizer::operator()(const Tensor &x, T threshold, Tensor &output) const {
  ValidateInput<T>(x);
  EXT_ENFORCE_INVALID(output.data_type == TensorElementType<T>::value,
                      "kernel::Binarizer preallocated output dtype must match the input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Binarizer preallocated output shape must match the input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(x.element_count()) * sizeof(T),
                      "kernel::Binarizer preallocated output buffer is incorrectly sized.");
  ApplyThreshold<T>(x, threshold, output.As<T>());
}

// Explicit instantiations for the supported element types.
#define ONNX_LIGHT_INSTANTIATE_BINARIZER(T)                                                        \
  template Tensor Binarizer::operator()(const Tensor &, T, RuntimeContext *) const;                \
  template void Binarizer::operator()(const Tensor &, T, Tensor &) const

ONNX_LIGHT_INSTANTIATE_BINARIZER(float);
ONNX_LIGHT_INSTANTIATE_BINARIZER(double);
ONNX_LIGHT_INSTANTIATE_BINARIZER(int64_t);
ONNX_LIGHT_INSTANTIATE_BINARIZER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_BINARIZER

void Binarizer::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const float threshold = GetAttributeFloatOrDefault(node, "threshold", 0.0f);
  onnx_kernels::kernel::Binarizer binarizer(rt.kernel_ctx());
  Tensor y = DispatchSVMByDataType(x, "Binarizer", [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    (void)tag;
    return binarizer.template operator()<T>(x, static_cast<T>(threshold));
  });
  SetOutput(node, 0, std::move(y), rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

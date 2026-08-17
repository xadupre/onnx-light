// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kClipName = "kernel::Clip";

template <typename T> T ReadScalar(const Tensor &t) {
  EXT_ENFORCE_INVALID(t.element_count() == 1, kClipName, ": min/max must be 0-D (scalar) tensors.");
  return *reinterpret_cast<const T *>(t.bytes());
}

template <typename T>
void ClipInPlace(const Tensor &x, const Tensor *min, const Tensor *max, Tensor &output) {
  const T lo = min ? ReadScalar<T>(*min) : std::numeric_limits<T>::lowest();
  const T hi = max ? ReadScalar<T>(*max) : std::numeric_limits<T>::max();
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  // Matches ONNX semantics: y = Min(max, Max(input, min)). When ``lo > hi``
  // every element is clamped to ``hi``.
  for (int64_t i = 0; i < n; ++i) {
    T v = px[i];
    if (v < lo)
      v = lo;
    if (v > hi)
      v = hi;
    py[i] = v;
  }
}

void ValidateBounds(const Tensor &x, const Tensor *min, const Tensor *max) {
  if (min != nullptr) {
    EXT_ENFORCE_INVALID(min->data_type == x.data_type, kClipName,
                        ": min dtype must match input dtype.");
    EXT_ENFORCE_INVALID(min->element_count() == 1, kClipName,
                        ": min must be a 0-D (scalar) tensor.");
  }
  if (max != nullptr) {
    EXT_ENFORCE_INVALID(max->data_type == x.data_type, kClipName,
                        ": max dtype must match input dtype.");
    EXT_ENFORCE_INVALID(max->element_count() == 1, kClipName,
                        ": max must be a 0-D (scalar) tensor.");
  }
}

void ValidateOutput(const Tensor &x, const Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type, kClipName,
                      ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kClipName, ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(), kClipName,
                      ": output buffer size mismatch.");
}

void Dispatch(const Tensor &x, const Tensor *min, const Tensor *max, Tensor &output) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ClipInPlace<float>(x, min, max, output);
    return;
  case DataType::DOUBLE:
    ClipInPlace<double>(x, min, max, output);
    return;
  case DataType::INT8:
    ClipInPlace<int8_t>(x, min, max, output);
    return;
  case DataType::INT16:
    ClipInPlace<int16_t>(x, min, max, output);
    return;
  case DataType::INT32:
    ClipInPlace<int32_t>(x, min, max, output);
    return;
  case DataType::INT64:
    ClipInPlace<int64_t>(x, min, max, output);
    return;
  case DataType::UINT8:
    ClipInPlace<uint8_t>(x, min, max, output);
    return;
  case DataType::UINT16:
    ClipInPlace<uint16_t>(x, min, max, output);
    return;
  case DataType::UINT32:
    ClipInPlace<uint32_t>(x, min, max, output);
    return;
  case DataType::UINT64:
    ClipInPlace<uint64_t>(x, min, max, output);
    return;
  default:
    EXT_THROW_INVALID(kClipName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE and (U)INT8/16/32/64 tensors.");
  }
}

} // namespace

Tensor Clip::operator()(const Tensor &x, const Tensor *min, const Tensor *max,
                        RuntimeContext *rt) const {
  ValidateBounds(x, min, max);
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, x.shape, out_n_bytes, nullptr);
  Dispatch(x, min, max, out);
  return out;
}

void Clip::operator()(const Tensor &x, const Tensor *min, const Tensor *max, Tensor &output) const {
  ValidateBounds(x, min, max);
  ValidateOutput(x, output);
  Dispatch(x, min, max, output);
}

void Clip::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor *min = GetOptionalInput(node, 1, rt.tensors());
  const Tensor *max = GetOptionalInput(node, 2, rt.tensors());
  onnx_kernels::kernel::Clip k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, min, max, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

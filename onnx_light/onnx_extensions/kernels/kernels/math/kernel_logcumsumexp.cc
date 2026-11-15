// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {
namespace {

int64_t ValidateInput(const Tensor &x, int64_t axis) {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::LogCumSumExp: input rank must be >= 1.");
  EXT_ENFORCE_INVALID(axis >= -rank && axis < rank, "kernel::LogCumSumExp: axis is out of range.");
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT || x.data_type == DataType::DOUBLE ||
                          x.data_type == DataType::FLOAT16 || x.data_type == DataType::BFLOAT16,
                      "kernel::LogCumSumExp: unsupported data type.");
  // Validate dimensions before taking products, including shapes with zero dimensions.
  int64_t count = 1;
  for (const int64_t dim : x.shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "kernel::LogCumSumExp: negative dimension.");
  }
  if (std::find(x.shape.begin(), x.shape.end(), int64_t{0}) != x.shape.end()) {
    count = 0;
  } else {
    for (const int64_t dim : x.shape) {
      EXT_ENFORCE_INVALID(dim <= std::numeric_limits<int64_t>::max() / count,
                          "kernel::LogCumSumExp: element count overflow.");
      count *= dim;
    }
  }
  const size_t element_size = ElementSize(x.data_type);
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(count) <=
                          std::numeric_limits<size_t>::max() / element_size,
                      "kernel::LogCumSumExp: byte count overflow.");
  EXT_ENFORCE_INVALID(x.size_bytes() == static_cast<size_t>(count) * element_size,
                      "kernel::LogCumSumExp: input buffer size mismatch.");
  return axis < 0 ? axis + rank : axis;
}

template <typename T> T LogAddExp(T a, T b) {
  if (std::isnan(a) || std::isnan(b))
    return std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();
  if (a == inf || b == inf)
    return inf;
  if (a == -inf)
    return b;
  if (b == -inf)
    return a;
  return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

template <typename T, typename Read, typename Write>
void Compute(const Tensor &x, int64_t axis, Read read, Write write) {
  if (x.size_bytes() == 0)
    return;
  int64_t outer = 1, inner = 1;
  const int64_t dim = x.shape[static_cast<size_t>(axis)];
  for (int64_t d = 0; d < axis; ++d)
    outer *= x.shape[static_cast<size_t>(d)];
  for (size_t d = static_cast<size_t>(axis) + 1; d < x.shape.size(); ++d)
    inner *= x.shape[d];
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      T acc = -std::numeric_limits<T>::infinity();
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t offset = (o * dim + k) * inner + i;
        // Keep the accumulator in the compute type, not the output storage type.
        acc = LogAddExp(acc, read(offset));
        write(offset, acc);
      }
    }
  }
}

} // namespace

Tensor LogCumSumExp::operator()(const Tensor &x, int64_t axis, RuntimeContext *rt) const {
  ValidateInput(x, axis);
  Tensor output = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, x.size_bytes())
                     : MakeOutputTensor(x.data_type, x.shape, x.size_bytes(), nullptr);
  (*this)(x, axis, output);
  return output;
}

void LogCumSumExp::operator()(const Tensor &x, int64_t axis, Tensor &output) const {
  const int64_t resolved_axis = ValidateInput(x, axis);
  EXT_ENFORCE_INVALID(output.data_type == x.data_type && output.shape == x.shape &&
                          output.size_bytes() == x.size_bytes(),
                      "kernel::LogCumSumExp: output dtype, shape and buffer must match input.");
  switch (x.data_type) {
  case DataType::DOUBLE: {
    const double *src = x.AsDouble();
    double *dst = reinterpret_cast<double *>(output.mutable_bytes());
    Compute<double>(
        x, resolved_axis, [src](int64_t i) { return src[i]; },
        [dst](int64_t i, double value) { dst[i] = value; });
    break;
  }
  case DataType::FLOAT: {
    const float *src = x.AsFloat();
    float *dst = reinterpret_cast<float *>(output.mutable_bytes());
    Compute<float>(
        x, resolved_axis, [src](int64_t i) { return src[i]; },
        [dst](int64_t i, float value) { dst[i] = value; });
    break;
  }
  default: {
    const uint16_t *src = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *dst = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    const auto decode = x.data_type == DataType::FLOAT16 ? core::runtime::Float16BitsToFloat
                                                         : core::runtime::Bfloat16BitsToFloat;
    const auto encode = x.data_type == DataType::FLOAT16 ? core::runtime::FloatToFloat16Bits
                                                         : core::runtime::FloatToBfloat16Bits;
    Compute<float>(
        x, resolved_axis, [src, decode](int64_t i) { return decode(src[i]); },
        [dst, encode](int64_t i, float value) { dst[i] = encode(value); });
    break;
  }
  }
}

void LogCumSumExp::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);
  SetOutput(node, 0, (*this)(GetInput(node, 0, rt), axis, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

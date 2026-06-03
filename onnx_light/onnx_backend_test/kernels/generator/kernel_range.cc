// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

template <typename T> T ReadScalar(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.element_count() == 1, "kernel::Range: '", name,
                      "' must be a scalar (single-element) tensor.");
  EXT_ENFORCE_INVALID(t.data.size() == sizeof(T), "kernel::Range: '", name,
                      "' has unexpected byte size.");
  T value;
  std::memcpy(&value, t.data.data(), sizeof(T));
  return value;
}

template <typename T>
Tensor ComputeRange(const Tensor &start, const Tensor &limit, const Tensor &delta, int32_t dtype) {
  const T s = ReadScalar<T>(start, "start");
  const T l = ReadScalar<T>(limit, "limit");
  const T d = ReadScalar<T>(delta, "delta");
  EXT_ENFORCE_INVALID(d != T(0), "kernel::Range: 'delta' must be non-zero.");

  // number_of_elements = max(ceil((limit - start) / delta), 0)
  // Compute in double to handle both integer and float types uniformly,
  // matching the upstream schema's shape-inference formula.
  int64_t n = static_cast<int64_t>(
      std::ceil((static_cast<double>(l) - static_cast<double>(s)) / static_cast<double>(d)));
  n = std::max<int64_t>(n, 0);

  std::vector<uint8_t> out_data(static_cast<std::size_t>(n) * sizeof(T));
  T *out_ptr = reinterpret_cast<T *>(out_data.data());
  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = static_cast<T>(s + static_cast<T>(i * d));
  }
  return Tensor("", dtype, {n}, std::move(out_data));
}

} // namespace

Tensor Range::operator()(const Tensor &start, const Tensor &limit, const Tensor &delta) const {
  EXT_ENFORCE_INVALID(start.data_type == limit.data_type && start.data_type == delta.data_type,
                      "kernel::Range: 'start', 'limit' and 'delta' must share the same dtype.");
  switch (static_cast<DataType>(start.data_type)) {
  case DataType::FLOAT:
    return ComputeRange<float>(start, limit, delta, start.data_type);
  case DataType::DOUBLE:
    return ComputeRange<double>(start, limit, delta, start.data_type);
  case DataType::INT16:
    return ComputeRange<int16_t>(start, limit, delta, start.data_type);
  case DataType::INT32:
    return ComputeRange<int32_t>(start, limit, delta, start.data_type);
  case DataType::INT64:
    return ComputeRange<int64_t>(start, limit, delta, start.data_type);
  default:
    throw std::invalid_argument("kernel::Range: unsupported input dtype.");
  }
}

void Range::operator()(const Tensor &start, const Tensor &limit, const Tensor &delta,
                       Tensor &output) const {
  Tensor produced = (*this)(start, limit, delta);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      "kernel::Range preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      "kernel::Range preallocated output shape must match the produced tensor "
                      "shape.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      "kernel::Range preallocated output buffer has unexpected size in bytes.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

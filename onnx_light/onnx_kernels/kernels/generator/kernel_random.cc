// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_kernels/random.h"
#include "onnx_kernels/runtime_context.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Validates the requested output shape: dims must be non-negative.
// Returns the element count.
int64_t CheckShape(const std::vector<int64_t> &shape, const char *op_name) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "kernel::", op_name, ": shape must not contain negative dims.");
    count *= dim;
  }
  return count;
}

// Resolves the output dtype. ``requested`` is the value of the ``dtype``
// attribute (0 means "absent"). ``default_dtype`` is used when no value
// is requested. Throws on unsupported dtype.
int32_t ResolveOutputDtype(int32_t requested, int32_t default_dtype, const char *op_name) {
  const int32_t out_dtype = (requested != 0) ? requested : default_dtype;
  switch (static_cast<DataType>(out_dtype)) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
    return out_dtype;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::", op_name, ": unsupported output dtype ",
                        std::to_string(out_dtype), "; only FLOAT and DOUBLE are supported.");
  }
  return out_dtype;
}

// Converts the FLOAT-typed ``seed`` attribute (passed here as int64) to
// the ``std::optional<uint64_t>`` accepted by :cpp:func:`Rand` /
// :cpp:func:`Randn`. ``kNoSeed`` (-1) means "no seed", which maps to
// ``std::nullopt``.
std::optional<uint64_t> NormalizeSeed(int64_t seed) {
  if (seed == -1) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(seed);
}

// Writes ``lowT + samples[i] * span`` into ``dst[i]`` for each sample.
template <typename T>
void FillUniform(T *dst, const std::vector<T> &samples, double low, double high) {
  const T span = static_cast<T>(high - low);
  const T lowT = static_cast<T>(low);
  for (size_t i = 0; i < samples.size(); ++i) {
    dst[i] = lowT + samples[i] * span;
  }
}

// Writes ``meanT + samples[i] * scaleT`` into ``dst[i]`` for each sample.
template <typename T>
void FillNormal(T *dst, const std::vector<T> &samples, double mean, double scale) {
  const T meanT = static_cast<T>(mean);
  const T scaleT = static_cast<T>(scale);
  for (size_t i = 0; i < samples.size(); ++i) {
    dst[i] = meanT + samples[i] * scaleT;
  }
}

template <typename T>
Tensor MakeUniformTensor(const std::vector<int64_t> &shape, double low, double high,
                         std::optional<uint64_t> seed, int32_t dtype) {
  const std::vector<T> samples = Rand<T>(shape, seed);
  std::vector<uint8_t> bytes(samples.size() * sizeof(T));
  FillUniform<T>(reinterpret_cast<T *>(bytes.data()), samples, low, high);
  return Tensor("", dtype, shape, std::move(bytes));
}

template <typename T>
Tensor MakeNormalTensor(const std::vector<int64_t> &shape, double mean, double scale,
                        std::optional<uint64_t> seed, int32_t dtype) {
  const std::vector<T> samples = Randn<T>(shape, seed);
  std::vector<uint8_t> bytes(samples.size() * sizeof(T));
  FillNormal<T>(reinterpret_cast<T *>(bytes.data()), samples, mean, scale);
  return Tensor("", dtype, shape, std::move(bytes));
}

Tensor MakeUniform(const std::vector<int64_t> &shape, double low, double high,
                   std::optional<uint64_t> seed, int32_t dtype, const char *op_name) {
  CheckShape(shape, op_name);
  if (static_cast<DataType>(dtype) == DataType::DOUBLE) {
    return MakeUniformTensor<double>(shape, low, high, seed, dtype);
  }
  return MakeUniformTensor<float>(shape, low, high, seed, dtype);
}

Tensor MakeNormal(const std::vector<int64_t> &shape, double mean, double scale,
                  std::optional<uint64_t> seed, int32_t dtype, const char *op_name) {
  CheckShape(shape, op_name);
  if (static_cast<DataType>(dtype) == DataType::DOUBLE) {
    return MakeNormalTensor<double>(shape, mean, scale, seed, dtype);
  }
  return MakeNormalTensor<float>(shape, mean, scale, seed, dtype);
}

// Writes uniform samples directly into the pre-allocated ``output`` buffer.
void WriteUniformInto(Tensor &output, const std::vector<int64_t> &shape, double low, double high,
                      std::optional<uint64_t> seed, int32_t out_dtype, const char *op_name) {
  EXT_ENFORCE_INVALID(output.data_type == out_dtype, "kernel::", op_name,
                      " preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == shape, "kernel::", op_name,
                      " preallocated output shape must match the requested shape.");
  if (static_cast<DataType>(out_dtype) == DataType::DOUBLE) {
    const std::vector<double> samples = Rand<double>(shape, seed);
    FillUniform<double>(reinterpret_cast<double *>(output.mutable_bytes()), samples, low, high);
  } else {
    const std::vector<float> samples = Rand<float>(shape, seed);
    FillUniform<float>(reinterpret_cast<float *>(output.mutable_bytes()), samples, low, high);
  }
}

// Writes normal samples directly into the pre-allocated ``output`` buffer.
void WriteNormalInto(Tensor &output, const std::vector<int64_t> &shape, double mean, double scale,
                     std::optional<uint64_t> seed, int32_t out_dtype, const char *op_name) {
  EXT_ENFORCE_INVALID(output.data_type == out_dtype, "kernel::", op_name,
                      " preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == shape, "kernel::", op_name,
                      " preallocated output shape must match the requested shape.");
  if (static_cast<DataType>(out_dtype) == DataType::DOUBLE) {
    const std::vector<double> samples = Randn<double>(shape, seed);
    FillNormal<double>(reinterpret_cast<double *>(output.mutable_bytes()), samples, mean, scale);
  } else {
    const std::vector<float> samples = Randn<float>(shape, seed);
    FillNormal<float>(reinterpret_cast<float *>(output.mutable_bytes()), samples, mean, scale);
  }
}

} // namespace

Tensor RandomNormal::operator()(const std::vector<int64_t> &shape, double mean, double scale,
                                int64_t seed, int32_t dtype, RuntimeContext *rt) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomNormal");
  return MakeNormal(shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormal");
}

void RandomNormal::operator()(const std::vector<int64_t> &shape, double mean, double scale,
                              int64_t seed, int32_t dtype, Tensor &output) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomNormal");
  CheckShape(shape, "RandomNormal");
  WriteNormalInto(output, shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormal");
}

Tensor RandomUniform::operator()(const std::vector<int64_t> &shape, double low, double high,
                                 int64_t seed, int32_t dtype, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniform: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomUniform");
  return MakeUniform(shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniform");
}

void RandomUniform::operator()(const std::vector<int64_t> &shape, double low, double high,
                               int64_t seed, int32_t dtype, Tensor &output) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniform: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomUniform");
  CheckShape(shape, "RandomUniform");
  WriteUniformInto(output, shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniform");
}

Tensor RandomNormalLike::operator()(const Tensor &input, double mean, double scale, int64_t seed,
                                    int32_t dtype, RuntimeContext *rt) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomNormalLike");
  return MakeNormal(input.shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormalLike");
}

void RandomNormalLike::operator()(const Tensor &input, double mean, double scale, int64_t seed,
                                  int32_t dtype, Tensor &output) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomNormalLike");
  CheckShape(input.shape, "RandomNormalLike");
  WriteNormalInto(output, input.shape, mean, scale, NormalizeSeed(seed), out_dtype,
                  "RandomNormalLike");
}

Tensor RandomUniformLike::operator()(const Tensor &input, double low, double high, int64_t seed,
                                     int32_t dtype, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniformLike: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomUniformLike");
  return MakeUniform(input.shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniformLike");
}

void RandomUniformLike::operator()(const Tensor &input, double low, double high, int64_t seed,
                                   int32_t dtype, Tensor &output) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniformLike: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomUniformLike");
  CheckShape(input.shape, "RandomUniformLike");
  WriteUniformInto(output, input.shape, low, high, NormalizeSeed(seed), out_dtype,
                   "RandomUniformLike");
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

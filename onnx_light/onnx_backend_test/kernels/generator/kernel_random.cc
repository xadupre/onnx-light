// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_backend_test/random.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Validates the requested output shape: dims must be non-negative.
// Returns the element count.
int64_t CheckShape(const std::vector<int64_t> &shape, const char *op_name) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0, std::string("kernel::") + op_name +
                                      ": shape must not contain negative dims.");
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
    EXT_ENFORCE_INVALID(false, std::string("kernel::") + op_name + ": unsupported output dtype " +
                                   std::to_string(out_dtype) +
                                   "; only FLOAT and DOUBLE are supported.");
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

template <typename T>
Tensor MakeUniformTensor(const std::vector<int64_t> &shape, double low, double high,
                         std::optional<uint64_t> seed, int32_t dtype) {
  const std::vector<T> samples = Rand<T>(shape, seed);
  const T span = static_cast<T>(high - low);
  const T lowT = static_cast<T>(low);
  std::vector<uint8_t> bytes(samples.size() * sizeof(T));
  for (size_t i = 0; i < samples.size(); ++i) {
    T v = lowT + samples[i] * span;
    std::memcpy(bytes.data() + i * sizeof(T), &v, sizeof(T));
  }
  return Tensor("", dtype, shape, std::move(bytes));
}

template <typename T>
Tensor MakeNormalTensor(const std::vector<int64_t> &shape, double mean, double scale,
                        std::optional<uint64_t> seed, int32_t dtype) {
  const std::vector<T> samples = Randn<T>(shape, seed);
  const T meanT = static_cast<T>(mean);
  const T scaleT = static_cast<T>(scale);
  std::vector<uint8_t> bytes(samples.size() * sizeof(T));
  for (size_t i = 0; i < samples.size(); ++i) {
    T v = meanT + samples[i] * scaleT;
    std::memcpy(bytes.data() + i * sizeof(T), &v, sizeof(T));
  }
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

void CopyIntoOutput(const Tensor &produced, Tensor &output, const char *op_name) {
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      std::string("kernel::") + op_name +
                          " preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      std::string("kernel::") + op_name +
                          " preallocated output shape must match the produced tensor shape.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      std::string("kernel::") + op_name +
                          " preallocated output buffer has unexpected size in bytes.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace

Tensor RandomNormal::operator()(const std::vector<int64_t> &shape, double mean, double scale,
                                int64_t seed, int32_t dtype) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomNormal");
  return MakeNormal(shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormal");
}

void RandomNormal::operator()(const std::vector<int64_t> &shape, double mean, double scale,
                              int64_t seed, int32_t dtype, Tensor &output) const {
  Tensor produced = (*this)(shape, mean, scale, seed, dtype);
  CopyIntoOutput(produced, output, "RandomNormal");
}

Tensor RandomUniform::operator()(const std::vector<int64_t> &shape, double low, double high,
                                 int64_t seed, int32_t dtype) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniform: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomUniform");
  return MakeUniform(shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniform");
}

void RandomUniform::operator()(const std::vector<int64_t> &shape, double low, double high,
                               int64_t seed, int32_t dtype, Tensor &output) const {
  Tensor produced = (*this)(shape, low, high, seed, dtype);
  CopyIntoOutput(produced, output, "RandomUniform");
}

Tensor RandomNormalLike::operator()(const Tensor &input, double mean, double scale, int64_t seed,
                                    int32_t dtype) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomNormalLike");
  return MakeNormal(input.shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormalLike");
}

void RandomNormalLike::operator()(const Tensor &input, double mean, double scale, int64_t seed,
                                  int32_t dtype, Tensor &output) const {
  Tensor produced = (*this)(input, mean, scale, seed, dtype);
  CopyIntoOutput(produced, output, "RandomNormalLike");
}

Tensor RandomUniformLike::operator()(const Tensor &input, double low, double high, int64_t seed,
                                     int32_t dtype) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniformLike: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomUniformLike");
  return MakeUniform(input.shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniformLike");
}

void RandomUniformLike::operator()(const Tensor &input, double low, double high, int64_t seed,
                                   int32_t dtype, Tensor &output) const {
  Tensor produced = (*this)(input, low, high, seed, dtype);
  CopyIntoOutput(produced, output, "RandomUniformLike");
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

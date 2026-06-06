// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Default seed used when the operator's ``seed`` attribute is absent.
// Picking a fixed value keeps the kernel deterministic so that backend
// test cases can store stable expected outputs while still mirroring the
// schema's non-deterministic contract.
constexpr uint32_t kDefaultMultinomialSeed = 0u;

// Decodes a single IEEE-754 binary16 value to ``double`` (same logic as
// the Bernoulli kernel; kept local here to avoid coupling the two
// translation units).
double DecodeHalf(uint16_t h) {
  const uint32_t sign = (h >> 15) & 0x1u;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      uint32_t m = mant;
      int32_t e = -1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      f = (sign << 31) | (static_cast<uint32_t>(e + 127 + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    f = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    f = (sign << 31) | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
  }
  float fv;
  std::memcpy(&fv, &f, sizeof(float));
  return static_cast<double>(fv);
}

// Reads the per-row log-probabilities of ``input`` into a row-major
// ``std::vector<double>`` of size ``batch_size * class_size``.
std::vector<double> ReadLogits(const Tensor &input, int64_t batch_size, int64_t class_size) {
  const int64_t n = batch_size * class_size;
  std::vector<double> logits(static_cast<std::size_t>(n));
  switch (static_cast<DataType>(input.data_type)) {
  case DataType::FLOAT: {
    const float *src = input.AsFloat();
    for (int64_t i = 0; i < n; ++i) {
      logits[static_cast<std::size_t>(i)] = static_cast<double>(src[i]);
    }
    break;
  }
  case DataType::DOUBLE: {
    const double *src = input.AsDouble();
    for (int64_t i = 0; i < n; ++i) {
      logits[static_cast<std::size_t>(i)] = src[i];
    }
    break;
  }
  case DataType::FLOAT16: {
    const uint16_t *src = reinterpret_cast<const uint16_t *>(input.data.data());
    for (int64_t i = 0; i < n; ++i) {
      logits[static_cast<std::size_t>(i)] = DecodeHalf(src[i]);
    }
    break;
  }
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Multinomial: unsupported input dtype " +
                                   std::to_string(input.data_type) + ".");
  }
  return logits;
}

// Returns the byte stride of the supported output dtypes.
std::size_t OutputElementSize(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::INT32:
    return sizeof(int32_t);
  case DataType::INT64:
    return sizeof(int64_t);
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Multinomial: unsupported output dtype " +
                                   std::to_string(dtype) + "; only INT32 and INT64 are supported.");
  }
  return 0;
}

// Stores ``sample`` (a class index) at index ``i`` of ``out`` using the
// stride for ``dtype``.
void StoreSample(int32_t dtype, std::vector<uint8_t> &out, int64_t i, int64_t sample) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::INT32: {
    const int32_t v = static_cast<int32_t>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(int32_t), &v, sizeof(int32_t));
    break;
  }
  case DataType::INT64: {
    const int64_t v = sample;
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(int64_t), &v, sizeof(int64_t));
    break;
  }
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Multinomial: unsupported output dtype " +
                                   std::to_string(dtype) + ".");
  }
}

} // namespace

Tensor Multinomial::operator()(const Tensor &input, int64_t sample_size, int64_t seed,
                               int32_t dtype) const {
  EXT_ENFORCE_INVALID(input.shape.size() == 2,
                      "kernel::Multinomial: input must be a 2-D tensor of shape "
                      "[batch_size, class_size].");
  EXT_ENFORCE_INVALID(sample_size >= 0,
                      "kernel::Multinomial: sample_size must be non-negative, got " +
                          std::to_string(sample_size) + ".");

  const int64_t batch_size = input.shape[0];
  const int64_t class_size = input.shape[1];
  EXT_ENFORCE_INVALID(batch_size >= 0 && class_size >= 0,
                      "kernel::Multinomial: input shape must be non-negative.");
  EXT_ENFORCE_INVALID(batch_size == 0 || class_size > 0,
                      "kernel::Multinomial: class_size must be > 0 for a non-empty batch.");

  const std::vector<double> logits = ReadLogits(input, batch_size, class_size);

  const int32_t out_dtype = (dtype == 0) ? static_cast<int32_t>(DataType::INT32) : dtype;
  const std::size_t es = OutputElementSize(out_dtype);
  const int64_t n_out = batch_size * sample_size;
  std::vector<uint8_t> out_data(static_cast<std::size_t>(n_out) * es, 0);

  const uint32_t engine_seed =
      (seed == kNoSeed) ? kDefaultMultinomialSeed : static_cast<uint32_t>(seed);
  std::mt19937 engine(engine_seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  // Per-row CDF over normalized class probabilities (softmax of the
  // unnormalized log-probabilities). Reused across rows.
  std::vector<double> cdf(static_cast<std::size_t>(class_size));

  for (int64_t b = 0; b < batch_size; ++b) {
    const double *row = logits.data() + static_cast<std::size_t>(b * class_size);

    // Stable softmax: subtract the row max before exponentiating so very
    // large logits do not overflow.
    double max_logit = row[0];
    for (int64_t c = 1; c < class_size; ++c) {
      if (row[c] > max_logit) {
        max_logit = row[c];
      }
    }
    double sum = 0.0;
    for (int64_t c = 0; c < class_size; ++c) {
      const double p = std::exp(row[c] - max_logit);
      sum += p;
      cdf[static_cast<std::size_t>(c)] = sum;
    }
    EXT_ENFORCE_INVALID(sum > 0.0, "kernel::Multinomial: row " + std::to_string(b) +
                                       " produced an all-zero probability distribution.");
    // Normalize the CDF to end at exactly 1.0.
    for (int64_t c = 0; c < class_size; ++c) {
      cdf[static_cast<std::size_t>(c)] /= sum;
    }

    for (int64_t s = 0; s < sample_size; ++s) {
      const double u = uniform(engine);
      // Inverse-CDF: find the smallest index whose CDF >= u.
      const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
      int64_t idx = static_cast<int64_t>(it - cdf.begin());
      if (idx >= class_size) {
        idx = class_size - 1;
      }
      StoreSample(out_dtype, out_data, b * sample_size + s, idx);
    }
  }

  return Tensor("", out_dtype, {batch_size, sample_size}, std::move(out_data));
}

void Multinomial::operator()(const Tensor &input, int64_t sample_size, int64_t seed, int32_t dtype,
                             Tensor &output) const {
  Tensor produced = (*this)(input, sample_size, seed, dtype);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      "kernel::Multinomial preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      "kernel::Multinomial preallocated output shape must match the produced "
                      "tensor shape.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      "kernel::Multinomial preallocated output buffer has unexpected size in "
                      "bytes.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

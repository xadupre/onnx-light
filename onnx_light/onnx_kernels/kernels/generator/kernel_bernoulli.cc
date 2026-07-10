// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include "onnx_kernels/runtime_context.h"
#include <bit>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Default seed used when the operator's ``seed`` attribute is absent.
// Picking a fixed value keeps the kernel deterministic so that backend
// test cases can store stable expected outputs while still mirroring the
// schema's non-deterministic contract.
constexpr uint32_t kDefaultBernoulliSeed = 0u;

// Decodes a single IEEE-754 binary16 value to ``double``.
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
  return static_cast<double>(std::bit_cast<float>(f));
}

// Number of bytes per element for the supported output dtypes.
std::size_t OutputElementSize(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
    return sizeof(float);
  case DataType::DOUBLE:
    return sizeof(double);
  case DataType::FLOAT16:
    return 2;
  case DataType::INT8:
  case DataType::UINT8:
  case DataType::BOOL:
    return 1;
  case DataType::INT16:
  case DataType::UINT16:
    return 2;
  case DataType::INT32:
  case DataType::UINT32:
    return 4;
  case DataType::INT64:
  case DataType::UINT64:
    return 8;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Bernoulli: unsupported output dtype ",
                        std::to_string(dtype), ".");
  }
  return 0;
}

// Stores ``sample`` (either 0 or 1) at the byte position pointed to by ``out``
// using the encoding for ``dtype``.  The caller is responsible for advancing
// ``out`` by the element size between successive calls.
void StoreSample(int32_t dtype, uint8_t *out, int32_t sample) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
    *reinterpret_cast<float *>(out) = static_cast<float>(sample);
    break;
  case DataType::DOUBLE:
    *reinterpret_cast<double *>(out) = static_cast<double>(sample);
    break;
  case DataType::FLOAT16:
    // 0.0 -> 0x0000, 1.0 -> 0x3C00 in IEEE-754 binary16.
    *reinterpret_cast<uint16_t *>(out) =
        sample == 0 ? static_cast<uint16_t>(0x0000) : static_cast<uint16_t>(0x3C00);
    break;
  case DataType::INT8:
    *reinterpret_cast<int8_t *>(out) = static_cast<int8_t>(sample);
    break;
  case DataType::UINT8:
  case DataType::BOOL:
    *out = static_cast<uint8_t>(sample);
    break;
  case DataType::INT16:
    *reinterpret_cast<int16_t *>(out) = static_cast<int16_t>(sample);
    break;
  case DataType::UINT16:
    *reinterpret_cast<uint16_t *>(out) = static_cast<uint16_t>(sample);
    break;
  case DataType::INT32:
    *reinterpret_cast<int32_t *>(out) = sample;
    break;
  case DataType::UINT32:
    *reinterpret_cast<uint32_t *>(out) = static_cast<uint32_t>(sample);
    break;
  case DataType::INT64:
    *reinterpret_cast<int64_t *>(out) = static_cast<int64_t>(sample);
    break;
  case DataType::UINT64:
    *reinterpret_cast<uint64_t *>(out) = static_cast<uint64_t>(sample);
    break;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Bernoulli: unsupported output dtype ",
                        std::to_string(dtype), ".");
  }
}

} // namespace

Tensor Bernoulli::operator()(const Tensor &input, int64_t seed, int32_t dtype,
                             RuntimeContext *rt) const {
  const int32_t out_dtype = (dtype == 0) ? input.data_type : dtype;
  const std::size_t es = OutputElementSize(out_dtype);
  const int64_t n = input.element_count();
  EXT_ENFORCE_INVALID(n >= 0, "kernel::Bernoulli: input element count must be non-negative.");
  const std::size_t out_n_bytes = static_cast<std::size_t>(n) * es;

  Tensor out =
      MakeOutputTensor(out_dtype, input.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(input, seed, dtype, out);
  return out;
}

void Bernoulli::operator()(const Tensor &input, int64_t seed, int32_t dtype, Tensor &output) const {
  const int32_t out_dtype = (dtype == 0) ? input.data_type : dtype;
  EXT_ENFORCE_INVALID(output.data_type == out_dtype,
                      "kernel::Bernoulli preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == input.shape,
                      "kernel::Bernoulli preallocated output shape must match the input "
                      "tensor shape.");

  const int64_t n = input.element_count();
  EXT_ENFORCE_INVALID(n >= 0, "kernel::Bernoulli: input element count must be non-negative.");
  const uint32_t engine_seed =
      (seed == kNoSeed) ? kDefaultBernoulliSeed : static_cast<uint32_t>(seed);
  std::mt19937 engine(engine_seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  const std::size_t es = OutputElementSize(out_dtype);
  uint8_t *out_data = output.mutable_bytes();

  auto draw_samples = [&](auto read_probability) {
    for (int64_t i = 0; i < n; ++i) {
      const double p = read_probability(i);
      EXT_ENFORCE_INVALID(p >= 0.0 && p <= 1.0,
                          "kernel::Bernoulli: input values must lie in [0, 1].");
      const double u = uniform(engine);
      const int32_t sample = (u < p) ? 1 : 0;
      StoreSample(out_dtype, out_data + static_cast<std::size_t>(i) * es, sample);
    }
  };

  switch (static_cast<DataType>(input.data_type)) {
  case DataType::FLOAT: {
    const float *src = input.AsFloat();
    draw_samples([&](int64_t i) { return static_cast<double>(src[i]); });
    break;
  }
  case DataType::DOUBLE: {
    const double *src = input.AsDouble();
    draw_samples([&](int64_t i) { return src[i]; });
    break;
  }
  case DataType::FLOAT16: {
    const uint16_t *src = reinterpret_cast<const uint16_t *>(input.bytes());
    draw_samples([&](int64_t i) { return DecodeHalf(src[i]); });
    break;
  }
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Bernoulli: unsupported input dtype ",
                        std::to_string(input.data_type), ".");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

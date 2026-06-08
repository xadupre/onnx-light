// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

template <typename T> T ReadScalar(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.element_count() == 1, "kernel::Range: '", name,
                      "' must be a scalar (single-element) tensor.");
  EXT_ENFORCE_INVALID(t.size_bytes() == sizeof(T), "kernel::Range: '", name,
                      "' has unexpected byte size.");
  T value;
  std::memcpy(&value, t.bytes(), sizeof(T));
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

// Reads a scalar tensor stored as the raw IEEE-754 binary16 ``float16``
// bit pattern and returns its value as a ``float``. The conversion mirrors
// the upstream IEEE-754 round-half-to-even mapping for finite, infinite,
// and NaN inputs.
float ReadFloat16Scalar(const Tensor &t, const char *name) {
  const uint16_t h = ReadScalar<uint16_t>(t, name);
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h & 0x7c00u) >> 10;
  const uint32_t mant = h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize.
      uint32_t m = mant;
      int e = -1;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x0400u) == 0);
      bits = sign | ((127u - 15u - static_cast<uint32_t>(e)) << 23) | ((m & 0x03ffu) << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Reads a scalar tensor stored as the raw ``bfloat16`` bit pattern
// (the upper 16 bits of an IEEE-754 binary32 ``float``) and returns its
// value as a ``float``.
float ReadBfloat16Scalar(const Tensor &t, const char *name) {
  const uint16_t b = ReadScalar<uint16_t>(t, name);
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// IEEE-754 binary16 encoder (round-to-nearest-even). Duplicated locally
// to keep this kernel self-contained.
uint16_t FloatToFloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t e32 = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t m32 = u & 0x007fffffu;
  if (e32 == 0xff) {
    return static_cast<uint16_t>(sign | 0x7c00u | (m32 != 0 ? 0x0200u : 0u));
  }
  const int32_t e = e32 - 127 + 15;
  if (e >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t m = (m32 | 0x00800000u) >> static_cast<uint32_t>(1 - e);
    const uint32_t round_bit = (m >> 12) & 1u;
    const uint32_t sticky = m & 0x00000fffu;
    uint16_t h = static_cast<uint16_t>(sign | (m >> 13));
    if (round_bit && (sticky != 0 || (h & 1))) {
      h = static_cast<uint16_t>(h + 1);
    }
    return h;
  }
  const uint32_t low = m32 & 0x1fffu;
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m32 >> 13));
  if (low > 0x1000u || (low == 0x1000u && (h & 1u))) {
    h = static_cast<uint16_t>(h + 1);
  }
  return h;
}

// Round-to-nearest-even ``float`` -> ``bfloat16`` encoder. Matches the
// behaviour of the upstream ``onnx.helper`` bfloat16 helpers.
uint16_t FloatToBfloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // NaN: preserve a quiet NaN (non-zero mantissa) in the upper 16 bits.
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
    return static_cast<uint16_t>((u >> 16) | 0x0040u);
  }
  // Round-to-nearest-even on the lower 16 bits.
  const uint32_t rounding_bias = 0x00007fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

// Computes a Range output whose element type is float16 or bfloat16 by
// accumulating in 32-bit float (the v27 ``stash_type`` semantics). The
// encoder/decoder pair (``read``/``encode``) selects between the two
// half-precision layouts.
Tensor ComputeRangeHalf(const Tensor &start, const Tensor &limit, const Tensor &delta,
                        int32_t dtype, float (*read)(const Tensor &, const char *),
                        uint16_t (*encode)(float)) {
  const float s = read(start, "start");
  const float l = read(limit, "limit");
  const float d = read(delta, "delta");
  EXT_ENFORCE_INVALID(d != 0.0f, "kernel::Range: 'delta' must be non-zero.");

  int64_t n = static_cast<int64_t>(
      std::ceil((static_cast<double>(l) - static_cast<double>(s)) / static_cast<double>(d)));
  n = std::max<int64_t>(n, 0);

  std::vector<uint8_t> out_data(static_cast<std::size_t>(n) * sizeof(uint16_t));
  uint16_t *out_ptr = reinterpret_cast<uint16_t *>(out_data.data());
  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = encode(s + static_cast<float>(i) * d);
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
  case DataType::FLOAT16:
    return ComputeRangeHalf(start, limit, delta, start.data_type, &ReadFloat16Scalar,
                            &FloatToFloat16Bits);
  case DataType::BFLOAT16:
    return ComputeRangeHalf(start, limit, delta, start.data_type, &ReadBfloat16Scalar,
                            &FloatToBfloat16Bits);
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
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

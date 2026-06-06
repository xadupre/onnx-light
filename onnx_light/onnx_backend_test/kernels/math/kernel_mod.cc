// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kModName = "kernel::Mod";

// ``fmod == 0``: Python/NumPy-style integer modulo whose sign follows the
// divisor (matches ``numpy.mod`` and the upstream
// ``test_mod_mixed_sign_int*`` / ``test_mod_uint*`` / ``test_mod_broadcast``
// reference cases). For unsigned types this collapses to ``a % b`` because
// the result is always non-negative.
template <typename T> T PythonMod(T a, T b) {
  static_assert(std::is_integral<T>::value, "PythonMod requires an integral type.");
  T r = static_cast<T>(a % b);
  if constexpr (std::is_signed<T>::value) {
    if (r != 0 && ((r < 0) != (b < 0))) {
      r = static_cast<T>(r + b);
    }
  }
  return r;
}

// ``fmod == 1`` on integers: C ``%`` truncated modulo whose sign follows the
// dividend (matches ``numpy.fmod`` and the upstream ``test_mod_int64_fmod``
// reference case).
template <typename T> T TruncMod(T a, T b) {
  static_assert(std::is_integral<T>::value, "TruncMod requires an integral type.");
  return static_cast<T>(a % b);
}

// ``fmod == 1`` on floats: C ``std::fmod``. Matches the upstream
// ``test_mod_mixed_sign_float{32,64}`` reference cases. No explicit
// divide-by-zero check is required here: ``std::fmod`` follows IEEE 754
// and returns NaN for ``b == 0``, mirroring NumPy's ``np.fmod``. The
// integer ``PythonMod``/``TruncMod`` overloads above also do not check
// for zero divisors, matching the convention established by
// :ref:`kernel::Div` (the upstream ONNX backend tests guarantee non-zero
// divisors).
template <typename T> T FloatFmod(T a, T b) {
  static_assert(std::is_floating_point<T>::value, "FloatFmod requires a floating-point type.");
  return std::fmod(a, b);
}

template <typename T>
Tensor ModAllocInt(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                   int64_t fmod) {
  if (fmod == 0) {
    return detail::BinaryElementwiseAlloc<T, T>(kModName, dtype_name, dtype, x, y,
                                                [](T a, T b) -> T { return PythonMod<T>(a, b); });
  }
  return detail::BinaryElementwiseAlloc<T, T>(kModName, dtype_name, dtype, x, y,
                                              [](T a, T b) -> T { return TruncMod<T>(a, b); });
}

template <typename T>
void ModInPlaceInt(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                   int64_t fmod, Tensor &output) {
  if (fmod == 0) {
    detail::BinaryElementwise<T, T>(kModName, dtype_name, dtype, x, y, output,
                                    [](T a, T b) -> T { return PythonMod<T>(a, b); });
    return;
  }
  detail::BinaryElementwise<T, T>(kModName, dtype_name, dtype, x, y, output,
                                  [](T a, T b) -> T { return TruncMod<T>(a, b); });
}

template <typename T>
Tensor ModAllocFloat(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAlloc<T, T>(kModName, dtype_name, dtype, x, y,
                                              [](T a, T b) -> T { return FloatFmod<T>(a, b); });
}

template <typename T>
void ModInPlaceFloat(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                     Tensor &output) {
  detail::BinaryElementwise<T, T>(kModName, dtype_name, dtype, x, y, output,
                                  [](T a, T b) -> T { return FloatFmod<T>(a, b); });
}

// IEEE-754 binary16 helpers for the FLOAT16 dispatch path. Mirrors the
// helpers used by ``cases_attention`` / ``cases_dequantizelinear``;
// duplicated here to keep the kernel self-contained and avoid depending
// on the case files. ``np.fmod`` on float16 inputs yields the same bit
// pattern as round-tripping through float32 fmod, so this conversion
// path matches the upstream ``test_mod_mixed_sign_float16`` reference.
uint16_t FloatToFloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t e32 = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t m32 = u & 0x007fffffu;
  if (e32 == 0xff) {
    // Inf / NaN: preserve sign; collapse the mantissa to a quiet-NaN
    // marker when it was non-zero.
    return static_cast<uint16_t>(sign | 0x7c00u | (m32 != 0 ? 0x0200u : 0u));
  }
  const int32_t e = e32 - 127 + 15;
  if (e >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u); // overflow -> +/-inf
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<uint16_t>(sign); // too small -> +/-0
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
    h = static_cast<uint16_t>(h + 1); // mantissa carry naturally bumps exponent
  }
  return h;
}

float Float16BitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 0x1u;
  const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1fu;
  const uint32_t mant = static_cast<uint32_t>(h) & 0x3ffu;
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
      m &= 0x3ffu;
      f = (sign << 31) | (static_cast<uint32_t>(e + 127 + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
  }
  float fv;
  std::memcpy(&fv, &f, sizeof(float));
  return fv;
}

Tensor ModAllocFloat16(const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAlloc<uint16_t, uint16_t>(
      kModName, "FLOAT16", DataType::FLOAT16, x, y, [](uint16_t a, uint16_t b) -> uint16_t {
        return FloatToFloat16Bits(std::fmod(Float16BitsToFloat(a), Float16BitsToFloat(b)));
      });
}

void ModInPlaceFloat16(const Tensor &x, const Tensor &y, Tensor &output) {
  detail::BinaryElementwise<uint16_t, uint16_t>(
      kModName, "FLOAT16", DataType::FLOAT16, x, y, output, [](uint16_t a, uint16_t b) -> uint16_t {
        return FloatToFloat16Bits(std::fmod(Float16BitsToFloat(a), Float16BitsToFloat(b)));
      });
}

constexpr const char *kSupportedModTypesMsg =
    " only supports FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32 and "
    "UINT64 inputs.";

constexpr const char *kFmodRequiredForFloatMsg =
    " requires attribute ``fmod`` set to 1 for floating-point inputs.";
} // namespace

Tensor Mod::operator()(const Tensor &x, const Tensor &y, int64_t fmod) const {
  switch (x.data_type) {
  case DataType::FLOAT16:
    if (fmod != 1) {
      throw std::invalid_argument(std::string(kModName) + kFmodRequiredForFloatMsg);
    }
    return ModAllocFloat16(x, y);
  case DataType::FLOAT:
    if (fmod != 1) {
      throw std::invalid_argument(std::string(kModName) + kFmodRequiredForFloatMsg);
    }
    return ModAllocFloat<float>("FLOAT", DataType::FLOAT, x, y);
  case DataType::DOUBLE:
    if (fmod != 1) {
      throw std::invalid_argument(std::string(kModName) + kFmodRequiredForFloatMsg);
    }
    return ModAllocFloat<double>("DOUBLE", DataType::DOUBLE, x, y);
  case DataType::INT8:
    return ModAllocInt<int8_t>("INT8", DataType::INT8, x, y, fmod);
  case DataType::INT16:
    return ModAllocInt<int16_t>("INT16", DataType::INT16, x, y, fmod);
  case DataType::INT32:
    return ModAllocInt<int32_t>("INT32", DataType::INT32, x, y, fmod);
  case DataType::INT64:
    return ModAllocInt<int64_t>("INT64", DataType::INT64, x, y, fmod);
  case DataType::UINT8:
    return ModAllocInt<uint8_t>("UINT8", DataType::UINT8, x, y, fmod);
  case DataType::UINT16:
    return ModAllocInt<uint16_t>("UINT16", DataType::UINT16, x, y, fmod);
  case DataType::UINT32:
    return ModAllocInt<uint32_t>("UINT32", DataType::UINT32, x, y, fmod);
  case DataType::UINT64:
    return ModAllocInt<uint64_t>("UINT64", DataType::UINT64, x, y, fmod);
  default:
    throw std::invalid_argument(std::string(kModName) + kSupportedModTypesMsg);
  }
}

void Mod::operator()(const Tensor &x, const Tensor &y, int64_t fmod, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT16:
    if (fmod != 1) {
      throw std::invalid_argument(std::string(kModName) + kFmodRequiredForFloatMsg);
    }
    return ModInPlaceFloat16(x, y, output);
  case DataType::FLOAT:
    if (fmod != 1) {
      throw std::invalid_argument(std::string(kModName) + kFmodRequiredForFloatMsg);
    }
    return ModInPlaceFloat<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::DOUBLE:
    if (fmod != 1) {
      throw std::invalid_argument(std::string(kModName) + kFmodRequiredForFloatMsg);
    }
    return ModInPlaceFloat<double>("DOUBLE", DataType::DOUBLE, x, y, output);
  case DataType::INT8:
    return ModInPlaceInt<int8_t>("INT8", DataType::INT8, x, y, fmod, output);
  case DataType::INT16:
    return ModInPlaceInt<int16_t>("INT16", DataType::INT16, x, y, fmod, output);
  case DataType::INT32:
    return ModInPlaceInt<int32_t>("INT32", DataType::INT32, x, y, fmod, output);
  case DataType::INT64:
    return ModInPlaceInt<int64_t>("INT64", DataType::INT64, x, y, fmod, output);
  case DataType::UINT8:
    return ModInPlaceInt<uint8_t>("UINT8", DataType::UINT8, x, y, fmod, output);
  case DataType::UINT16:
    return ModInPlaceInt<uint16_t>("UINT16", DataType::UINT16, x, y, fmod, output);
  case DataType::UINT32:
    return ModInPlaceInt<uint32_t>("UINT32", DataType::UINT32, x, y, fmod, output);
  case DataType::UINT64:
    return ModInPlaceInt<uint64_t>("UINT64", DataType::UINT64, x, y, fmod, output);
  default:
    throw std::invalid_argument(std::string(kModName) + kSupportedModTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

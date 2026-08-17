// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kModName = "kernel::Mod";
constexpr std::array<int32_t, 12> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),   static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16),
    static_cast<int32_t>(DataType::INT8),    static_cast<int32_t>(DataType::INT16),
    static_cast<int32_t>(DataType::INT32),   static_cast<int32_t>(DataType::INT64),
    static_cast<int32_t>(DataType::UINT8),   static_cast<int32_t>(DataType::UINT16),
    static_cast<int32_t>(DataType::UINT32),  static_cast<int32_t>(DataType::UINT64),
};

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
                   int64_t fmod, int64_t grain, RawBufferAllocator *allocator = nullptr) {
  if (fmod == 0) {
    return detail::BinaryElementwiseAlloc<T, T>(
        kModName, dtype_name, dtype, x, y, [](T a, T b) -> T { return PythonMod<T>(a, b); },
        allocator, grain);
  }
  return detail::BinaryElementwiseAlloc<T, T>(
      kModName, dtype_name, dtype, x, y, [](T a, T b) -> T { return TruncMod<T>(a, b); }, allocator,
      grain);
}

template <typename T>
void ModInPlaceInt(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                   int64_t fmod, Tensor &output, int64_t grain) {
  if (fmod == 0) {
    detail::BinaryElementwise<T, T>(
        kModName, dtype_name, dtype, x, y, output, [](T a, T b) -> T { return PythonMod<T>(a, b); },
        grain);
    return;
  }
  detail::BinaryElementwise<T, T>(
      kModName, dtype_name, dtype, x, y, output, [](T a, T b) -> T { return TruncMod<T>(a, b); },
      grain);
}

template <typename T>
Tensor ModAllocFloat(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                     int64_t grain, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<T, T>(
      kModName, dtype_name, dtype, x, y, [](T a, T b) -> T { return FloatFmod<T>(a, b); },
      allocator, grain);
}

template <typename T>
void ModInPlaceFloat(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                     Tensor &output, int64_t grain) {
  detail::BinaryElementwise<T, T>(
      kModName, dtype_name, dtype, x, y, output, [](T a, T b) -> T { return FloatFmod<T>(a, b); },
      grain);
}

// IEEE-754 binary16 helpers for the FLOAT16 dispatch path are provided by
// ``onnx_core/runtime/kernels/cast_helper.h`` (``FloatToFloat16Bits`` /
// ``Float16BitsToFloat``). ``np.fmod`` on float16 inputs yields the same
// bit pattern as round-tripping through float32 fmod, so this conversion
// path matches the upstream ``test_mod_mixed_sign_float16`` reference.

Tensor ModAllocFloat16(const Tensor &x, const Tensor &y, int64_t grain,
                       RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<uint16_t, uint16_t>(
      kModName, "FLOAT16", DataType::FLOAT16, x, y,
      [](uint16_t a, uint16_t b) -> uint16_t {
        return FloatToFloat16Bits(std::fmod(Float16BitsToFloat(a), Float16BitsToFloat(b)));
      },
      allocator, grain);
}

void ModInPlaceFloat16(const Tensor &x, const Tensor &y, Tensor &output, int64_t grain) {
  detail::BinaryElementwise<uint16_t, uint16_t>(
      kModName, "FLOAT16", DataType::FLOAT16, x, y, output,
      [](uint16_t a, uint16_t b) -> uint16_t {
        return FloatToFloat16Bits(std::fmod(Float16BitsToFloat(a), Float16BitsToFloat(b)));
      },
      grain);
}

Tensor ModAllocBfloat16(const Tensor &x, const Tensor &y, int64_t grain,
                        RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<uint16_t, uint16_t>(
      kModName, "BFLOAT16", DataType::BFLOAT16, x, y,
      [](uint16_t a, uint16_t b) -> uint16_t {
        return FloatToBfloat16Bits(std::fmod(Bfloat16BitsToFloat(a), Bfloat16BitsToFloat(b)));
      },
      allocator, grain);
}

void ModInPlaceBfloat16(const Tensor &x, const Tensor &y, Tensor &output, int64_t grain) {
  detail::BinaryElementwise<uint16_t, uint16_t>(
      kModName, "BFLOAT16", DataType::BFLOAT16, x, y, output,
      [](uint16_t a, uint16_t b) -> uint16_t {
        return FloatToBfloat16Bits(std::fmod(Bfloat16BitsToFloat(a), Bfloat16BitsToFloat(b)));
      },
      grain);
}

constexpr const char *kSupportedModTypesMsg =
    " only supports FLOAT16, BFLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, "
    "UINT32 and UINT64 inputs.";

constexpr const char *kFmodRequiredForFloatMsg =
    " requires attribute ``fmod`` set to 1 for floating-point inputs.";
} // namespace

Mod::Mod(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Mod", kSupportedElementTypes, kParallelForGrainSize) {}

void Mod::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Mod", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor Mod::operator()(const Tensor &x, const Tensor &y, int64_t fmod, RuntimeContext *rt) const {
  if (rt != nullptr) {
    const Shape out_shape = detail::BroadcastShape("kernel::Mod", x.shape, y.shape);
    const int64_t out_count = out_shape.product();
    Tensor output = rt->MakeOutputTensor(0, x.data_type, out_shape,
                                         static_cast<size_t>(out_count) * ElementSize(x.data_type));
    (*this)(x, y, fmod, output);
    return output;
  }
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (x.data_type) {
  case DataType::FLOAT16:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModAllocFloat16(x, y, grain, nullptr);
  case DataType::BFLOAT16:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModAllocBfloat16(x, y, grain, nullptr);
  case DataType::FLOAT:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModAllocFloat<float>("FLOAT", DataType::FLOAT, x, y, grain, nullptr);
  case DataType::DOUBLE:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModAllocFloat<double>("DOUBLE", DataType::DOUBLE, x, y, grain, nullptr);
  case DataType::INT8:
    return ModAllocInt<int8_t>("INT8", DataType::INT8, x, y, fmod, grain, nullptr);
  case DataType::INT16:
    return ModAllocInt<int16_t>("INT16", DataType::INT16, x, y, fmod, grain, nullptr);
  case DataType::INT32:
    return ModAllocInt<int32_t>("INT32", DataType::INT32, x, y, fmod, grain, nullptr);
  case DataType::INT64:
    return ModAllocInt<int64_t>("INT64", DataType::INT64, x, y, fmod, grain, nullptr);
  case DataType::UINT8:
    return ModAllocInt<uint8_t>("UINT8", DataType::UINT8, x, y, fmod, grain, nullptr);
  case DataType::UINT16:
    return ModAllocInt<uint16_t>("UINT16", DataType::UINT16, x, y, fmod, grain, nullptr);
  case DataType::UINT32:
    return ModAllocInt<uint32_t>("UINT32", DataType::UINT32, x, y, fmod, grain, nullptr);
  case DataType::UINT64:
    return ModAllocInt<uint64_t>("UINT64", DataType::UINT64, x, y, fmod, grain, nullptr);
  default:
    EXT_THROW_INVALID(kModName, ": unsupported data type ", x.data_type, kSupportedModTypesMsg);
  }
}

void Mod::operator()(const Tensor &x, const Tensor &y, int64_t fmod, Tensor &output) const {
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (x.data_type) {
  case DataType::FLOAT16:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModInPlaceFloat16(x, y, output, grain);
  case DataType::BFLOAT16:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModInPlaceBfloat16(x, y, output, grain);
  case DataType::FLOAT:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModInPlaceFloat<float>("FLOAT", DataType::FLOAT, x, y, output, grain);
  case DataType::DOUBLE:
    EXT_ENFORCE_INVALID(fmod == 1, kModName, ": unsupported data type ", x.data_type,
                        kFmodRequiredForFloatMsg);
    return ModInPlaceFloat<double>("DOUBLE", DataType::DOUBLE, x, y, output, grain);
  case DataType::INT8:
    return ModInPlaceInt<int8_t>("INT8", DataType::INT8, x, y, fmod, output, grain);
  case DataType::INT16:
    return ModInPlaceInt<int16_t>("INT16", DataType::INT16, x, y, fmod, output, grain);
  case DataType::INT32:
    return ModInPlaceInt<int32_t>("INT32", DataType::INT32, x, y, fmod, output, grain);
  case DataType::INT64:
    return ModInPlaceInt<int64_t>("INT64", DataType::INT64, x, y, fmod, output, grain);
  case DataType::UINT8:
    return ModInPlaceInt<uint8_t>("UINT8", DataType::UINT8, x, y, fmod, output, grain);
  case DataType::UINT16:
    return ModInPlaceInt<uint16_t>("UINT16", DataType::UINT16, x, y, fmod, output, grain);
  case DataType::UINT32:
    return ModInPlaceInt<uint32_t>("UINT32", DataType::UINT32, x, y, fmod, output, grain);
  case DataType::UINT64:
    return ModInPlaceInt<uint64_t>("UINT64", DataType::UINT64, x, y, fmod, output, grain);
  default:
    EXT_THROW_INVALID(kModName, ": unsupported data type ", x.data_type, kSupportedModTypesMsg);
  }
}

void Mod::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  const int64_t fmod = GetAttributeIntOrDefault(node, "fmod", 0);
  onnx_kernels::kernel::Mod k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, y, fmod, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

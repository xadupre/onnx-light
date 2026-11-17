// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::detail {

// Two limbs retain exact integer sums on platforms without a native 128-bit integer.
struct ScatterIntegerSum {
  uint64_t low = 0;
  uint64_t high = 0;

  template <typename T> void Add(T value) {
    const uint64_t bits = static_cast<uint64_t>(value);
    const uint64_t old = low;
    low += bits;
    high += low < old;
    if constexpr (std::is_signed_v<T>) {
      high += value < 0 ? UINT64_MAX : 0;
    }
  }

  template <typename T> T Mean(uint64_t count) const {
    uint64_t lo = low;
    uint64_t hi = high;
    const bool negative = std::is_signed_v<T> && (hi >> 63) != 0;
    if (negative) {
      lo = ~lo + 1;
      hi = ~hi + (lo == 0);
    }
    uint64_t quotient = 0;
    uint64_t remainder = 0;
    for (int bit = 127; bit >= 0; --bit) {
      const uint64_t next = bit >= 64 ? (hi >> (bit - 64)) & 1 : (lo >> bit) & 1;
      const bool carry = (remainder >> 63) != 0;
      remainder = (remainder << 1) | next;
      if (carry || remainder >= count) {
        remainder -= count;
        if (bit < 64) {
          quotient |= uint64_t{1} << bit;
        }
      }
    }
    if (negative) {
      quotient += remainder != 0; // Integer means round toward negative infinity.
      if (quotient == uint64_t{1} << 63) {
        return std::numeric_limits<T>::min();
      }
      return static_cast<T>(-static_cast<int64_t>(quotient));
    }
    return static_cast<T>(quotient);
  }
};

// Exact multiples of the smallest double subnormal, with enough headroom for
// INT64_MAX contributions. Used only when the ordinary compensated sum overflows.
struct ScatterFloatSum {
  std::array<uint64_t, 35> limbs{};

  void Add(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const unsigned exponent = static_cast<unsigned>((bits >> 52) & 0x7ff);
    const uint64_t significand =
        (bits & UINT64_C(0xfffffffffffff)) | (exponent != 0 ? uint64_t{1} << 52 : 0);
    const unsigned shift = exponent == 0 ? 0 : exponent - 1;
    const size_t first = shift / 64;
    const unsigned offset = shift % 64;
    const bool negative = (bits >> 63) != 0;
    uint64_t carry = 0;
    for (size_t i = first; i < limbs.size(); ++i) {
      uint64_t term = 0;
      if (i == first) {
        term = significand << offset;
      } else if (i == first + 1 && offset != 0) {
        term = significand >> (64 - offset);
      }
      const uint64_t old = limbs[i];
      if (negative) {
        const uint64_t next = old - term;
        limbs[i] = next - carry;
        carry = (old < term) || (next < carry);
      } else {
        const uint64_t next = old + term;
        limbs[i] = next + carry;
        carry = (next < old) || (limbs[i] < next);
      }
    }
  }

  double Mean(uint64_t count) const {
    auto quotient = limbs;
    const bool negative = (quotient.back() >> 63) != 0;
    if (negative) {
      uint64_t carry = 1;
      for (uint64_t &limb : quotient) {
        limb = ~limb + carry;
        carry = carry != 0 && limb == 0;
      }
    }
    uint64_t remainder = 0;
    for (size_t i = quotient.size(); i-- > 0;) {
      const uint64_t value = quotient[i];
      quotient[i] = 0;
      for (int bit = 63; bit >= 0; --bit) {
        const bool carry = (remainder >> 63) != 0;
        remainder = (remainder << 1) | ((value >> bit) & 1);
        if (carry || remainder >= count) {
          remainder -= count;
          quotient[i] |= uint64_t{1} << bit;
        }
      }
    }
    int highest = static_cast<int>(quotient.size() * 64) - 1;
    const auto bit_at = [&](int bit) {
      return (quotient[static_cast<size_t>(bit) / 64] >> (bit % 64)) & 1;
    };
    while (highest >= 0 && bit_at(highest) == 0) {
      --highest;
    }
    const int shift = std::max(0, highest - 52);
    uint64_t significand = 0;
    for (int bit = highest; bit >= shift; --bit) {
      significand = (significand << 1) | bit_at(bit);
    }
    bool round_up;
    if (shift == 0) {
      round_up =
          remainder > count - remainder || (remainder == count - remainder && (significand & 1));
    } else {
      bool sticky = remainder != 0;
      for (int bit = 0; bit < shift - 1; ++bit) {
        sticky |= bit_at(bit) != 0;
      }
      round_up = bit_at(shift - 1) && (sticky || (significand & 1));
    }
    significand += round_up;
    const double result = std::scalbn(static_cast<double>(significand), shift - 1074);
    return negative ? -result : result;
  }
};

template <typename T, typename Read, typename Write>
void ScatterMeanTyped(const Tensor &data, const Tensor &updates, Tensor &output,
                      const std::vector<int64_t> &offsets, int64_t slice_elements, Read read,
                      Write write) {
  if (updates.element_count() == 0) {
    return;
  }
  const size_t size = static_cast<size_t>(data.element_count());
  std::vector<uint64_t> counts(size, 1);
  const auto visit = [&](auto apply) {
    for (size_t t = 0; t < offsets.size(); ++t) {
      for (int64_t j = 0; j < slice_elements; ++j) {
        apply(static_cast<size_t>(offsets[t] + j),
              t * static_cast<size_t>(slice_elements) + static_cast<size_t>(j));
      }
    }
  };
  visit([&](size_t dst, size_t) { ++counts[dst]; });
  if constexpr (std::is_integral_v<T>) {
    std::vector<ScatterIntegerSum> sums(size);
    for (size_t i = 0; i < size; ++i) {
      if (counts[i] > 1) {
        sums[i].Add(read(data, i));
      }
    }
    visit([&](size_t dst, size_t src) { sums[dst].Add(read(updates, src)); });
    for (size_t i = 0; i < size; ++i) {
      if (counts[i] > 1) {
        write(output, i, sums[i].template Mean<T>(counts[i]));
      }
    }
  } else {
    std::vector<double> scales(size, 0), sums(size, 0), corrections(size, 0);
    std::vector<uint8_t> special(size, 0);
    const auto observe = [&](size_t i, T value) {
      if (std::isnan(value)) {
        special[i] |= 1;
      } else if (std::isinf(value)) {
        special[i] |= std::signbit(value) ? 2 : 4;
      } else {
        scales[i] = std::max(scales[i], std::abs(static_cast<double>(value)));
      }
    };
    for (size_t i = 0; i < size; ++i) {
      if (counts[i] > 1) {
        observe(i, read(data, i));
      }
    }
    visit([&](size_t dst, size_t src) { observe(dst, read(updates, src)); });
    const auto add = [&](size_t i, double value) {
      if (special[i] != 0) {
        return;
      }
      const double next = sums[i] + value;
      corrections[i] += std::abs(sums[i]) >= std::abs(value) ? (sums[i] - next) + value
                                                             : (value - next) + sums[i];
      sums[i] = next;
    };
    for (size_t i = 0; i < size; ++i) {
      if (counts[i] > 1) {
        add(i, read(data, i));
      }
    }
    visit([&](size_t dst, size_t src) { add(dst, read(updates, src)); });
    std::vector<std::unique_ptr<ScatterFloatSum>> repairs(size);
    for (size_t i = 0; i < size; ++i) {
      if (counts[i] > 1 && special[i] == 0 && !std::isfinite(sums[i] + corrections[i])) {
        repairs[i] = std::make_unique<ScatterFloatSum>();
        repairs[i]->Add(read(data, i));
      }
    }
    visit([&](size_t dst, size_t src) {
      if (repairs[dst]) {
        repairs[dst]->Add(read(updates, src));
      }
    });
    for (size_t i = 0; i < size; ++i) {
      if (counts[i] == 1) {
        continue; // Retains untouched NaN payloads and signed zeros bit-for-bit.
      }
      double value;
      if ((special[i] & 1) || (special[i] & 6) == 6) {
        value = std::numeric_limits<double>::quiet_NaN();
      } else if (special[i] != 0) {
        value = special[i] & 2 ? -std::numeric_limits<double>::infinity()
                               : std::numeric_limits<double>::infinity();
      } else {
        value = repairs[i] ? repairs[i]->Mean(counts[i])
                           : (sums[i] + corrections[i]) / static_cast<double>(counts[i]);
        value = std::clamp(value, -scales[i], scales[i]);
      }
      write(output, i, static_cast<T>(value));
    }
  }
}

template <typename T>
void ScatterMeanNative(const Tensor &data, const Tensor &updates, Tensor &output,
                       const std::vector<int64_t> &offsets, int64_t slice_elements) {
  ScatterMeanTyped<T>(
      data, updates, output, offsets, slice_elements,
      [](const Tensor &tensor, size_t i) { return tensor.As<T>()[i]; },
      [](Tensor &tensor, size_t i, T value) { tensor.As<T>()[i] = value; });
}

inline void ScatterMean(const Tensor &data, const Tensor &updates, Tensor &output,
                        const std::vector<int64_t> &offsets, int64_t slice_elements) {
  switch (static_cast<DataType>(data.data_type)) {
#define SCATTER_MEAN_TYPE(dtype, type)                                                             \
  case DataType::dtype:                                                                            \
    ScatterMeanNative<type>(data, updates, output, offsets, slice_elements);                       \
    return
    SCATTER_MEAN_TYPE(INT8, int8_t);
    SCATTER_MEAN_TYPE(UINT8, uint8_t);
    SCATTER_MEAN_TYPE(INT16, int16_t);
    SCATTER_MEAN_TYPE(UINT16, uint16_t);
    SCATTER_MEAN_TYPE(INT32, int32_t);
    SCATTER_MEAN_TYPE(UINT32, uint32_t);
    SCATTER_MEAN_TYPE(INT64, int64_t);
    SCATTER_MEAN_TYPE(UINT64, uint64_t);
    SCATTER_MEAN_TYPE(FLOAT, float);
    SCATTER_MEAN_TYPE(DOUBLE, double);
#undef SCATTER_MEAN_TYPE
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    const auto decode =
        data.data_type == DataType::FLOAT16 ? Float16BitsToFloat : Bfloat16BitsToFloat;
    const auto encode =
        data.data_type == DataType::FLOAT16 ? FloatToFloat16Bits : FloatToBfloat16Bits;
    ScatterMeanTyped<float>(
        data, updates, output, offsets, slice_elements,
        [decode](const Tensor &tensor, size_t i) {
          return decode(reinterpret_cast<const uint16_t *>(tensor.bytes())[i]);
        },
        [encode](Tensor &tensor, size_t i, float value) {
          reinterpret_cast<uint16_t *>(tensor.mutable_bytes())[i] = encode(value);
        });
    return;
  }
  default:
    EXT_THROW_INVALID("Scatter mean requires a supported real numeric dtype.");
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::detail

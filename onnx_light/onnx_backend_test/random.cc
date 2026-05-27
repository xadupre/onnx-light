// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/random.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

constexpr double kInvTwoPow53 = 1.0 / static_cast<double>(static_cast<uint64_t>(1) << 53);

int64_t ShapeToCount(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return 1;
  }
  int64_t count = 1;
  for (int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "shape cannot contain negative dimensions.");
    count *= dim;
  }
  return count;
}

inline double UniformFromState(uint64_t value) {
  return static_cast<double>(value >> 11) * kInvTwoPow53;
}

} // namespace

std::pair<uint64_t, uint64_t> NextUint64(uint64_t state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t mixed = state;
  mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
  mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
  mixed = mixed ^ (mixed >> 31);
  return {state, mixed};
}

template <typename T>
std::vector<T> Rand(const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
  static_assert(std::is_floating_point_v<T>, "Rand<T> requires a floating-point element type.");
  const int64_t count = ShapeToCount(shape);
  std::vector<T> values(static_cast<size_t>(count));
  uint64_t state = seed.value_or(kDefaultSeed);
  for (int64_t i = 0; i < count; ++i) {
    auto [next_state, value] = NextUint64(state);
    state = next_state;
    values[static_cast<size_t>(i)] = static_cast<T>(UniformFromState(value));
  }
  return values;
}

template std::vector<double> Rand<double>(const std::vector<int64_t> &shape,
                                          std::optional<uint64_t> seed);
template std::vector<float> Rand<float>(const std::vector<int64_t> &shape,
                                        std::optional<uint64_t> seed);

std::vector<int64_t> RandInt(int64_t low, int64_t high, const std::vector<int64_t> &shape,
                             std::optional<uint64_t> seed) {
  EXT_ENFORCE_INVALID(high > low, "high must be greater than low.");
  const int64_t count = ShapeToCount(shape);
  const uint64_t span = static_cast<uint64_t>(high - low);
  // Mirror the Python reference:
  //   limit = 2**64 - (2**64 % span)
  // Using unsigned 64-bit arithmetic: (0 - span) % span == 2**64 % span.
  const uint64_t modulo = (static_cast<uint64_t>(0) - span) % span;
  const uint64_t limit = static_cast<uint64_t>(0) - modulo;
  std::vector<int64_t> values(static_cast<size_t>(count));
  uint64_t state = seed.value_or(kDefaultSeed);
  for (int64_t i = 0; i < count; ++i) {
    uint64_t candidate;
    while (true) {
      auto [next_state, value] = NextUint64(state);
      state = next_state;
      // ``limit`` equals 0 when span is a power of two that divides 2**64; in
      // that case every candidate is acceptable.
      if (limit == 0 || value < limit) {
        candidate = value % span;
        break;
      }
    }
    values[static_cast<size_t>(i)] = static_cast<int64_t>(candidate) + low;
  }
  return values;
}

template <typename T>
std::vector<T> Randn(const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
  static_assert(std::is_floating_point_v<T>, "Randn<T> requires a floating-point element type.");
  const int64_t count = ShapeToCount(shape);
  std::vector<T> values(static_cast<size_t>(count));
  uint64_t state = seed.value_or(kDefaultSeed);
  for (int64_t i = 0; i < count; ++i) {
    double sample = 0.0;
    for (int j = 0; j < 12; ++j) {
      auto [next_state, value] = NextUint64(state);
      state = next_state;
      sample += UniformFromState(value);
    }
    values[static_cast<size_t>(i)] = static_cast<T>(sample - 6.0);
  }
  return values;
}

template std::vector<double> Randn<double>(const std::vector<int64_t> &shape,
                                           std::optional<uint64_t> seed);
template std::vector<float> Randn<float>(const std::vector<int64_t> &shape,
                                         std::optional<uint64_t> seed);

Tensor RandBool(const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
  const std::vector<double> values = Randn<double>(shape, seed);
  std::vector<uint8_t> bytes(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bytes[i] = values[i] > 0.0 ? 1 : 0;
  }
  return Tensor("", static_cast<int32_t>(TensorProto::DataType::BOOL), shape, std::move(bytes));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

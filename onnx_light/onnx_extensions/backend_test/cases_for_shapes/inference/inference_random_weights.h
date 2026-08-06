// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

/// Builds a deterministic pseudo-random FLOAT weight vector of ``count``
/// elements in ``[-0.05, 0.05]`` (a typical initialization range for small
/// language-model weights). A Numerical-Recipes LCG seeded by ``seed`` keeps
/// the data reproducible across runs and platforms without depending on a
/// global RNG.
inline std::vector<float> RandomWeights(size_t count, uint32_t seed) {
  std::vector<float> values(count);
  uint32_t s = seed;
  for (size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s & 0x00ffffffu) / static_cast<float>(0x01000000u);
    values[i] = -0.05f + 0.1f * u;
  }
  return values;
}

/// Builds a deterministic pseudo-random FP16 weight vector of ``count``
/// elements. Each element is a normal FP16 value with exponent 0x34 (i.e.
/// in the range [0.25, 0.5)) and a random 10-bit mantissa. A
/// Numerical-Recipes LCG seeded by ``seed`` keeps the data reproducible
/// across runs and platforms. The values are returned as raw ``uint16_t``
/// bit-patterns as required by ``AddInitializer<uint16_t>``.
inline std::vector<uint16_t> RandomWeightsF16(size_t count, uint32_t seed) {
  std::vector<uint16_t> values(count);
  uint32_t s = seed;
  for (size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    values[i] = static_cast<uint16_t>(0x3400u | (s & 0x03FFu));
  }
  return values;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/parallel_for.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::statistics {

// Population variance is M2 / count. Empty reductions have zero statistics.
template <typename T> struct VarianceAccumulator {
  int64_t count = 0;
  T mean = 0;
  T m2 = 0;

  void Add(T value) {
    ++count;
    const T delta = value - mean;
    mean += delta / static_cast<T>(count);
    m2 += delta * (value - mean);
  }

  void Merge(const VarianceAccumulator &other) {
    if (other.count == 0) {
      return;
    }
    if (count == 0) {
      *this = other;
      return;
    }
    const T delta = other.mean - mean;
    const int64_t combined = count + other.count;
    m2 += other.m2 +
          delta * delta *
              (static_cast<T>(count) * static_cast<T>(other.count) / static_cast<T>(combined));
    mean += delta * (static_cast<T>(other.count) / static_cast<T>(combined));
    count = combined;
  }

  T Variance() const { return count == 0 ? T{0} : m2 / static_cast<T>(count); }
};

// Chunk boundaries and the left-to-right Chan merge are independent of worker
// scheduling and of the parallel crossover policy. Scratch is fixed-size.
template <typename T, typename Get>
VarianceAccumulator<T> Accumulate(int64_t count, Get get, int64_t minimum_elements) {
  constexpr int64_t kChunkCount = 8;
  const int64_t chunks = count >= core::runtime::kParallelForGrainSize ? kChunkCount : 1;
  std::array<VarianceAccumulator<T>, kChunkCount> partials{};
  if (chunks == 1) {
    for (int64_t i = 0; i < count; ++i) {
      partials[0].Add(static_cast<T>(get(i)));
    }
    return partials[0];
  }
  const int64_t chunk_size = count / chunks + (count % chunks != 0);
  core::runtime::ParallelFor(
      chunks,
      std::max<int64_t>(1, minimum_elements / chunk_size + (minimum_elements % chunk_size != 0)),
      [&](int64_t begin, int64_t end) {
        for (int64_t chunk = begin; chunk < end; ++chunk) {
          auto &part = partials[static_cast<size_t>(chunk)];
          const int64_t start = chunk * chunk_size;
          const int64_t stop = start + std::min<int64_t>(chunk_size, count - start);
          for (int64_t i = start; i < stop; ++i) {
            part.Add(static_cast<T>(get(i)));
          }
        }
      },
      "Variance");
  VarianceAccumulator<T> result;
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    result.Merge(partials[static_cast<size_t>(chunk)]);
  }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::statistics

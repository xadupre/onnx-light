// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/tensor_statistics.h"
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

bool ParseDimension(std::string_view text, int64_t &value) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && value > 0;
}

int64_t PeakRssBytes() {
#if defined(__unix__) || defined(__APPLE__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
#if defined(__APPLE__)
  return usage.ru_maxrss;
#else
  return usage.ru_maxrss * 1024;
#endif
#else
  return -1;
#endif
}

} // namespace

int main(int argc, char **argv) {
  int64_t rows = 150000, cols = 1024;
  if ((argc != 1 && argc != 3 && argc != 4) ||
      (argc >= 3 && (!ParseDimension(argv[1], rows) || !ParseDimension(argv[2], cols))) ||
      rows > std::numeric_limits<int64_t>::max() / cols ||
      static_cast<uint64_t>(rows * cols) > std::vector<float>().max_size() ||
      static_cast<uint64_t>(rows * cols) >
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float)) {
    std::cerr << "Usage: " << argv[0] << " [rows cols [raw-float-file]]\n";
    return 1;
  }
  const auto start = std::chrono::steady_clock::now();
  std::vector<float> values(static_cast<size_t>(rows * cols));
  if (argc == 4) {
    std::ifstream input(argv[3], std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float))) ||
        input.peek() != std::char_traits<char>::eof()) {
      std::cerr << "Expected a raw FLOAT file matching rows*cols elements\n";
      return 1;
    }
  } else {
    // A deterministic permutation avoids benchmarking already sorted or constant input.
    uint32_t state = 42;
    for (float &value : values) {
      state = state * 1664525u + 1013904223u;
      value = static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
    }
  }
  const auto tensor = Tensor::Borrow("X", TensorProto::FLOAT, {rows, cols},
                                     reinterpret_cast<const uint8_t *>(values.data()),
                                     values.size() * sizeof(float));
  const auto loaded = std::chrono::steady_clock::now();
  const int64_t materialization_rss = PeakRssBytes();
  auto phase_start = loaded;
  const char *phase_names[] = {"decode",    "summary",   "mean",      "distinct",
                               "histogram", "rectangle", "quantiles", "complete"};
  const auto stats = ComputeTensorStatistics(
      tensor, 0, {0.25, 0.5, 0.75}, true, {32, 32}, kTensorStatisticsMaxOutputBytes,
      [&](TensorStatisticsPhase phase, int64_t completed, int64_t total) {
        const auto now = std::chrono::steady_clock::now();
        if (completed == 0)
          phase_start = now;
        if (completed == total && phase != TensorStatisticsPhase::kComplete)
          std::cout << phase_names[static_cast<size_t>(phase)]
                    << "_seconds=" << std::chrono::duration<double>(now - phase_start).count()
                    << '\n';
        return true;
      });
  const auto finished = std::chrono::steady_clock::now();
  std::cout << "shape=[" << rows << ',' << cols << "]\n"
            << "materialization_seconds=" << std::chrono::duration<double>(loaded - start).count()
            << "\nstatistics_seconds=" << std::chrono::duration<double>(finished - loaded).count()
            << "\nmaterialization_peak_rss_bytes=" << materialization_rss
            << "\ntotal_peak_rss_bytes=" << PeakRssBytes() << "\nmedian=" << *stats.median
            << "\nquantiles=";
  for (double value : stats.quantiles)
    std::cout << value << ' ';
  std::cout << "\nrectangle_cells=" << stats.rectangle_map.size() << '\n';
}

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/prepared_tensor_cache.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

PreparedTensorMetadata Metadata() {
  PreparedTensorMetadata metadata;
  metadata.source_digest = "reduced-fixture-v1";
  metadata.architecture = "portable";
  metadata.runtime = "onnx-light";
  metadata.runtime_version = "1";
  metadata.kernel_layout = "benchmark-pack-v1";
  metadata.format_version = "1";
  return metadata;
}

double ElapsedMilliseconds(std::chrono::steady_clock::time_point begin,
                           std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

int main(int argc, char **argv) {
  const size_t fixture_bytes =
      argc > 1 ? static_cast<size_t>(std::strtoull(argv[1], nullptr, 10)) : 4U * 1024U * 1024U;
  const double minimum_improvement = argc > 2 ? std::atof(argv[2]) : 20.0;
  if (fixture_bytes == 0 || minimum_improvement < 0) {
    std::cerr << "fixture bytes must be positive and minimum improvement non-negative\n";
    return 1;
  }

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "onnx-light-prepared-cache-benchmark";
  const std::filesystem::path source_path = directory / "portable.bin";
  const std::filesystem::path cache_path = directory / "prepared.bin";
  std::filesystem::create_directories(directory);
  std::filesystem::remove(cache_path);
  {
    std::ofstream output(source_path, std::ios::binary | std::ios::trunc);
    for (size_t i = 0; i < fixture_bytes; ++i) {
      output.put(static_cast<char>(i * 131U + 17U));
    }
  }

  const auto load_source = [&]() {
    std::ifstream input(source_path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                                std::istreambuf_iterator<char>());
  };
  const auto prepack = [](const std::vector<uint8_t> &source) {
    std::vector<uint8_t> prepared = source;
    for (int pass = 0; pass < 8; ++pass) {
      for (size_t i = 0; i < prepared.size(); ++i) {
        prepared[i] = static_cast<uint8_t>((prepared[i] << 1) | (prepared[i] >> 7));
      }
    }
    return prepared;
  };

  std::vector<uint8_t> cold_value;
  PreparedTensorCache cache;
  const auto cold_begin = std::chrono::steady_clock::now();
  cache.LoadOrPrepare(cache_path, Metadata(), load_source, prepack,
                      [&](const std::vector<uint8_t> &value) { cold_value = value; });
  const auto cold_end = std::chrono::steady_clock::now();
  cache.WaitForBackgroundWrites();

  std::vector<uint8_t> warm_value;
  const auto warm_begin = std::chrono::steady_clock::now();
  const PreparedTensorLoadResult warm =
      cache.LoadOrPrepare(cache_path, Metadata(), load_source, prepack,
                          [&](const std::vector<uint8_t> &value) { warm_value = value; });
  const auto warm_end = std::chrono::steady_clock::now();

  const double cold_ms = ElapsedMilliseconds(cold_begin, cold_end);
  const double warm_ms = ElapsedMilliseconds(warm_begin, warm_end);
  const double improvement = 100.0 * (cold_ms - warm_ms) / cold_ms;
  std::cout << "portable_first_token_ms=" << cold_ms << '\n'
            << "prepared_first_token_ms=" << warm_ms << '\n'
            << "improvement_percent=" << improvement << '\n'
            << "minimum_improvement_percent=" << minimum_improvement << '\n';

  std::filesystem::remove_all(directory);
  return warm.cache_hit && warm_value == cold_value && improvement >= minimum_improvement ? 0 : 2;
}

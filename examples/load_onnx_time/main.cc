/**
 * main.cc – Standalone example: measure ONNX loading time with the standard
 * onnx C++ library (protobuf-based).
 *
 * Usage:
 *   ./load_onnx_time <model.onnx> [iterations] [num_threads]
 *
 * Note: num_threads is accepted for interface compatibility with the common
 * benchmark CLI but is not used because the standard onnx protobuf library
 * loads models sequentially.
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx/onnx_pb.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool ParsePositiveInt(const char *text, int &value) {
  const std::string arg(text);
  if (arg.empty()) {
    return false;
  }

  int parsed = 0;
  const char *begin = arg.data();
  const char *end = begin + arg.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end || parsed <= 0) {
    return false;
  }

  value = parsed;
  return true;
}

double ToMilliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

double ComputePopulationStdDevMs(const std::vector<double> &timings_ms, double avg_ms) {
  double squared_diff_sum = 0.0;
  for (double timing_ms : timings_ms) {
    const double diff = timing_ms - avg_ms;
    squared_diff_sum += diff * diff;
  }
  return std::sqrt(squared_diff_sum / static_cast<double>(timings_ms.size()));
}

void PrintModelSummary(const onnx::ModelProto &model) {
  if (model.has_ir_version()) {
    std::cout << "  IR version       : " << model.ir_version() << "\n";
  }
  if (model.has_producer_name()) {
    std::cout << "  Producer name    : " << model.producer_name() << "\n";
  }
  if (model.has_producer_version()) {
    std::cout << "  Producer version : " << model.producer_version() << "\n";
  }
  if (model.has_graph()) {
    const onnx::GraphProto &graph = model.graph();
    std::cout << "  Graph name       : " << graph.name() << "\n";
    std::cout << "  Nodes            : " << graph.node_size() << "\n";
    std::cout << "  Inputs           : " << graph.input_size() << "\n";
    std::cout << "  Outputs          : " << graph.output_size() << "\n";
    std::cout << "  Initializers     : " << graph.initializer_size() << "\n";
  }
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [iterations] [num_threads]\n";
    return 1;
  }

  const std::string file_path = argv[1];
  int iterations = 10;
  int num_threads = 1;
  if (argc >= 3) {
    if (!ParsePositiveInt(argv[2], iterations)) {
      std::cerr << "Invalid iterations value: " << argv[2] << "\n";
      return 1;
    }
  }
  if (argc == 4) {
    if (!ParsePositiveInt(argv[3], num_threads)) {
      std::cerr << "Invalid num_threads value: " << argv[3] << "\n";
      return 1;
    }
  }

  std::vector<double> timings_ms;
  timings_ms.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    model.Clear();

    try {
      if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
      }
      const auto begin = std::chrono::steady_clock::now();
      std::ifstream input(file_path, std::ios::binary);
      onnx::ModelProto model;
      if (!model.ParseFromIstream(&input)) {
        throw std::runtime_error("Failed to parse ONNX model from: " + file_path);
      }
      const auto end = std::chrono::steady_clock::now();
      timings_ms.push_back(ToMilliseconds(end - begin));
    } catch (const std::exception &e) {
      std::cerr << "Error loading '" << file_path << "': " << e.what() << "\n";
      return 1;
    }
  }

  std::vector<double> sorted_timings = timings_ms;
  std::sort(sorted_timings.begin(), sorted_timings.end());
  const double total_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
  const double avg_ms = total_ms / static_cast<double>(timings_ms.size());
  const double std_ms = ComputePopulationStdDevMs(timings_ms, avg_ms);
  const std::size_t n = sorted_timings.size();
  const double median_ms = (n % 2 == 1) ? sorted_timings[n / 2]
                                        : (sorted_timings[n / 2 - 1] + sorted_timings[n / 2]) / 2.0;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Loaded: " << file_path << "\n";
  if (std::filesystem::exists(file_path)) {
    const double file_size_mb =
        static_cast<double>(std::filesystem::file_size(file_path)) / (1024.0 * 1024.0);
    std::cout << "  File size (MB)   : " << file_size_mb << "\n";
  }
  std::cout << "  Iterations       : " << iterations << "\n";
  std::cout << "  Num threads      : " << num_threads << "\n";
  std::cout << "  Total load (ms)  : " << total_ms << "\n";
  std::cout << "  Average load (ms): " << avg_ms << "\n";
  std::cout << "  Median load (ms) : " << median_ms << "\n";
  std::cout << "  Min load (ms)    : " << sorted_timings.front() << "\n";
  std::cout << "  Max load (ms)    : " << sorted_timings.back() << "\n";
  std::cout << "  Std load (ms)    : " << std_ms << "\n";
  PrintModelSummary(model);

  return 0;
}

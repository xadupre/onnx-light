/**
 * main.cc — Standalone example: load an ONNX file with the onnx_light
 * C++ API and report loading timing statistics.
 *
 * Usage:
 *   ./load_onnx_light_time <model.onnx> [iterations] [num_threads] [copy_mode]
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

bool ParsePositiveInt(const char *text, int &value) {
  const std::string_view arg(text);
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

constexpr double kBytesPerMb = 1024.0 * 1024.0;

bool ParseLoadMode(const char *text, bool &no_copy, bool &touch_raw_data_pages) {
  const std::string_view arg(text);
  if (arg == "default") {
    no_copy = false;
    touch_raw_data_pages = false;
    return true;
  }
  if (arg == "nocopy") {
    no_copy = true;
    touch_raw_data_pages = false;
    return true;
  }
  if (arg == "nocopy_touch") {
    no_copy = true;
    touch_raw_data_pages = true;
    return true;
  }
  return false;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 5) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [iterations] [num_threads] [copy_mode]\n"
              << "  copy_mode: default | nocopy | nocopy_touch\n";
    return 1;
  }

  const std::string file_path = argv[1];
  int iterations = 5;
  int num_threads = 1;
  bool no_copy = false;
  bool touch_raw_data_pages = false;
  if (argc >= 3) {
    if (!ParsePositiveInt(argv[2], iterations)) {
      std::cerr << "Invalid iterations value: " << argv[2] << "\n";
      return 1;
    }
  }
  if (argc >= 4) {
    if (!ParsePositiveInt(argv[3], num_threads)) {
      std::cerr << "Invalid num_threads value: " << argv[3] << "\n";
      return 1;
    }
  }
  if (argc == 5) {
    if (!ParseLoadMode(argv[4], no_copy, touch_raw_data_pages)) {
      std::cerr << "Invalid copy_mode value: " << argv[4]
                << " (expected default, nocopy, or nocopy_touch)\n";
      return 1;
    }
  }

  ONNX_LIGHT_NAMESPACE::ModelProto model;
  std::vector<double> timings_ms;
  timings_ms.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    ONNX_LIGHT_NAMESPACE::ModelProto parsed_model;

    try {
      ONNX_LIGHT_NAMESPACE::utils::FileStream stream(file_path);
      ONNX_LIGHT_NAMESPACE::ParseOptions opts;
      opts.parallel = num_threads > 1;
      opts.num_threads = num_threads;
      opts.no_copy = no_copy;
      opts._touch_raw_data_pages = touch_raw_data_pages;
      const auto begin = std::chrono::steady_clock::now();
      ONNX_LIGHT_NAMESPACE::ParseModelProtoFromStream(parsed_model, stream, opts);
      const auto end = std::chrono::steady_clock::now();
      timings_ms.push_back(ToMilliseconds(end - begin));
      if (i + 1 == iterations) {
        model = std::move(parsed_model);
      }
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
        static_cast<double>(std::filesystem::file_size(file_path)) / kBytesPerMb;
    std::cout << "  File size (MB)   : " << file_size_mb << "\n";
  }
  std::cout << "  Iterations       : " << iterations << "\n";
  std::cout << "  Num threads      : " << num_threads << "\n";
  std::cout << "  Copy mode        : " << (no_copy ? "nocopy" : "default") << "\n";
  std::cout << "  Touch pages      : " << (touch_raw_data_pages ? "true" : "false") << "\n";
  std::cout << "  Total load (ms)  : " << total_ms << "\n";
  std::cout << "  Average load (ms): " << avg_ms << "\n";
  std::cout << "  Median load (ms) : " << median_ms << "\n";
  std::cout << "  Min load (ms)    : " << sorted_timings.front() << "\n";
  std::cout << "  Max load (ms)    : " << sorted_timings.back() << "\n";
  std::cout << "  Std load (ms)    : " << std_ms << "\n";

  if (model.has_ir_version()) {
    std::cout << "  IR version       : " << model.ref_ir_version() << "\n";
  }
  if (model.has_producer_name()) {
    std::cout << "  Producer name    : " << model.ref_producer_name().as_string() << "\n";
  }
  if (model.has_producer_version()) {
    std::cout << "  Producer version : " << model.ref_producer_version().as_string() << "\n";
  }
  if (model.has_domain()) {
    std::cout << "  Domain           : " << model.ref_domain().as_string() << "\n";
  }
  if (model.has_model_version()) {
    std::cout << "  Model version    : " << model.ref_model_version() << "\n";
  }
  if (model.has_doc_string()) {
    std::cout << "  Doc string       : " << model.ref_doc_string().as_string() << "\n";
  }

  if (model.has_graph()) {
    const ONNX_LIGHT_NAMESPACE::GraphProto &graph = model.ref_graph();
    std::cout << "  Graph name       : " << graph.ref_name().as_string() << "\n";
    std::cout << "  Nodes            : " << graph.ref_node().size() << "\n";
    std::cout << "  Inputs           : " << graph.ref_input().size() << "\n";
    std::cout << "  Outputs          : " << graph.ref_output().size() << "\n";
    std::cout << "  Initializers     : " << graph.ref_initializer().size() << "\n";
  }

  return 0;
}

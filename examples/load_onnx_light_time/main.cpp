/**
 * main.cpp — Standalone example: load an ONNX file with the onnx_light
 * C++ API and report loading timing statistics.
 *
 * Usage:
 *   ./load_onnx_light_time <model.onnx> [iterations] [num_threads]
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <algorithm>
#include <charconv>
#include <chrono>
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

constexpr double kBytesPerMb = 1024.0 * 1024.0;

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [iterations] [num_threads]\n";
    return 1;
  }

  const std::string file_path = argv[1];
  int iterations = 5;
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

  ONNX_LIGHT_NAMESPACE::ModelProto model;
  std::vector<double> timings_ms;
  timings_ms.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    ONNX_LIGHT_NAMESPACE::ModelProto parsed_model;

    try {
      ONNX_LIGHT_NAMESPACE::utils::MmapStream stream(file_path);
      ONNX_LIGHT_NAMESPACE::ParseOptions opts;
      opts.parallel = num_threads > 1;
      opts.num_threads = num_threads;
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

  const auto [min_it, max_it] = std::minmax_element(timings_ms.begin(), timings_ms.end());
  const double total_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
  const double avg_ms = total_ms / static_cast<double>(timings_ms.size());

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Loaded: " << file_path << "\n";
  if (std::filesystem::exists(file_path)) {
    const double file_size_mb =
        static_cast<double>(std::filesystem::file_size(file_path)) / kBytesPerMb;
    std::cout << "  File size (MB)   : " << file_size_mb << "\n";
  }
  std::cout << "  Iterations       : " << iterations << "\n";
  std::cout << "  Num threads      : " << num_threads << "\n";
  std::cout << "  Total load (ms)  : " << total_ms << "\n";
  std::cout << "  Average load (ms): " << avg_ms << "\n";
  std::cout << "  Min load (ms)    : " << *min_it << "\n";
  std::cout << "  Max load (ms)    : " << *max_it << "\n";

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

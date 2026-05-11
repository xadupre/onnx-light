/**
 * main.cpp – Standalone example: measure ONNX saving time (with external data)
 * using the onnx_light C++ API.
 *
 * Usage:
 *   ./save_onnx_light_time <model.onnx> <output_dir> [iterations] [num_threads]
 *
 * The executable loads <model.onnx> once, then saves it repeatedly using
 * TwoFilesWriteStream (main proto + external weights file).  Wall-clock
 * times for each save are reported as:
 *
 *   Average save (ms): X.XXX
 *   Min save (ms)    : X.XXX
 *   Max save (ms)    : X.XXX
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
#include <system_error>
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

void PrintModelSummary(const ONNX_LIGHT_NAMESPACE::ModelProto &model) {
  if (model.has_ir_version()) {
    std::cout << "  IR version       : " << model.ref_ir_version() << "\n";
  }
  if (model.has_producer_name()) {
    std::cout << "  Producer name    : " << model.ref_producer_name().as_string() << "\n";
  }
  if (model.has_producer_version()) {
    std::cout << "  Producer version : " << model.ref_producer_version().as_string() << "\n";
  }
  if (model.has_graph()) {
    const ONNX_LIGHT_NAMESPACE::GraphProto &graph = model.ref_graph();
    std::cout << "  Graph name       : " << graph.ref_name().as_string() << "\n";
    std::cout << "  Nodes            : " << graph.ref_node().size() << "\n";
    std::cout << "  Inputs           : " << graph.ref_input().size() << "\n";
    std::cout << "  Outputs          : " << graph.ref_output().size() << "\n";
    std::cout << "  Initializers     : " << graph.ref_initializer().size() << "\n";
  }
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> <output_dir> [iterations] [num_threads]\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const std::string output_dir = argv[2];
  int iterations = 5;
  int num_threads = 1;
  if (argc >= 4) {
    if (!ParsePositiveInt(argv[3], iterations)) {
      std::cerr << "Invalid iterations value: " << argv[3] << "\n";
      return 1;
    }
  }
  if (argc == 5) {
    if (!ParsePositiveInt(argv[4], num_threads)) {
      std::cerr << "Invalid num_threads value: " << argv[4] << "\n";
      return 1;
    }
  }

  // Load the model once before benchmarking.
  ONNX_LIGHT_NAMESPACE::ModelProto model;
  try {
    ONNX_LIGHT_NAMESPACE::utils::FileStream stream(input_path);
    ONNX_LIGHT_NAMESPACE::ParseOptions parse_opts;
    ONNX_LIGHT_NAMESPACE::ParseModelProtoFromStream(model, stream, parse_opts);
  } catch (const std::exception &e) {
    std::cerr << "Error loading '" << input_path << "': " << e.what() << "\n";
    return 1;
  }

  // Ensure the output directory exists.
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  if (ec) {
    std::cerr << "Cannot create output directory '" << output_dir << "': " << ec.message() << "\n";
    return 1;
  }

  const std::string out_onnx = output_dir + "/out.onnx";
  const std::string out_data = output_dir + "/out.onnx.data";

  std::vector<double> timings_ms;
  timings_ms.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    try {
      ONNX_LIGHT_NAMESPACE::utils::TwoFilesWriteStream stream(out_onnx, out_data);
      ONNX_LIGHT_NAMESPACE::SerializeOptions opts;
      opts.parallel = num_threads > 1;
      opts.num_threads = num_threads;
      const auto begin = std::chrono::steady_clock::now();
      ONNX_LIGHT_NAMESPACE::SerializeModelProtoToStream(model, stream, opts);
      const auto end = std::chrono::steady_clock::now();
      timings_ms.push_back(ToMilliseconds(end - begin));
    } catch (const std::exception &e) {
      std::cerr << "Error saving to '" << out_onnx << "': " << e.what() << "\n";
      return 1;
    }
  }

  const auto [min_it, max_it] = std::minmax_element(timings_ms.begin(), timings_ms.end());
  const double total_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
  const double avg_ms = total_ms / static_cast<double>(timings_ms.size());

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Saved: " << out_onnx << "\n";
  if (std::filesystem::exists(input_path)) {
    const double file_size_mb =
        static_cast<double>(std::filesystem::file_size(input_path)) / (1024.0 * 1024.0);
    std::cout << "  Input size (MB)  : " << file_size_mb << "\n";
  }
  std::cout << "  Iterations       : " << iterations << "\n";
  std::cout << "  Num threads      : " << num_threads << "\n";
  std::cout << "  Total save (ms)  : " << total_ms << "\n";
  std::cout << "  Average save (ms): " << avg_ms << "\n";
  std::cout << "  Min save (ms)    : " << *min_it << "\n";
  std::cout << "  Max save (ms)    : " << *max_it << "\n";
  PrintModelSummary(model);

  return 0;
}

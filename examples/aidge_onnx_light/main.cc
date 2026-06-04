/**
 * main.cc -- Standalone example: load an ONNX model with the onnx-light
 * C++ API, re-serialize it through onnx-light, and (optionally) hand the
 * resulting file to the Eclipse Aidge ONNX importer to build an Aidge
 * computation graph.
 *
 * Usage:
 *   ./aidge_onnx_light <model.onnx> [output.onnx]
 *
 * If ``output.onnx`` is omitted, a temporary file is created next to
 * ``<model.onnx>`` with the ``.onnxlight.tmp`` suffix and removed when the
 * program exits.
 *
 * The Aidge integration is enabled at compile time through the
 * ``AIDGE_ONNX_LIGHT_HAS_AIDGE`` macro, which is defined automatically by the
 * accompanying CMake project when both ``aidge_core`` and ``aidge_onnx`` are
 * found. See CMakeLists.txt for build instructions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#ifdef AIDGE_ONNX_LIGHT_HAS_AIDGE
#include <aidge/graph/GraphView.hpp>
#include <aidge/onnx/ONNX.hpp>
#endif

namespace {

namespace olight = ONNX_LIGHT_NAMESPACE;

double ToMilliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

void PrintModelSummary(const olight::ModelProto &model) {
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
    const olight::GraphProto &graph = model.ref_graph();
    std::cout << "  Graph name       : " << graph.ref_name().as_string() << "\n";
    std::cout << "  Nodes            : " << graph.ref_node().size() << "\n";
    std::cout << "  Inputs           : " << graph.ref_input().size() << "\n";
    std::cout << "  Outputs          : " << graph.ref_output().size() << "\n";
    std::cout << "  Initializers     : " << graph.ref_initializer().size() << "\n";
  }
}

// RAII helper: removes the path at scope exit unless ``release()`` was called.
class ScopedFile {
public:
  explicit ScopedFile(std::filesystem::path path) : path_(std::move(path)) {}
  ~ScopedFile() {
    if (!released_ && !path_.empty()) {
      std::error_code ec;
      std::filesystem::remove(path_, ec);
    }
  }
  ScopedFile(const ScopedFile &) = delete;
  ScopedFile &operator=(const ScopedFile &) = delete;

  const std::filesystem::path &path() const { return path_; }
  void release() { released_ = true; }

private:
  std::filesystem::path path_;
  bool released_ = false;
};

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [output.onnx]\n";
    return 1;
  }

  const std::string input_path = argv[1];

  // Step 1: load the input ONNX model with onnx-light. We use the
  // memory-mapped stream so models larger than the protobuf 2 GB limit and
  // models with external data are both handled transparently.
  olight::ModelProto model;
  try {
    const auto begin = std::chrono::steady_clock::now();
    olight::utils::MmapFileStream stream(input_path);
    olight::ParseOptions opts;
    olight::ParseModelProtoFromStream(model, stream, opts);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "Loaded with onnx-light: " << input_path << "\n";
    std::cout << "  Load time (ms)   : " << ToMilliseconds(end - begin) << "\n";
    PrintModelSummary(model);
  } catch (const std::exception &e) {
    std::cerr << "Error loading '" << input_path << "' with onnx-light: " << e.what() << "\n";
    return 1;
  }

  // Step 2: decide where to write the re-serialized, protobuf-compatible
  // ModelProto. When the caller did not provide an explicit destination we
  // create a temporary file next to the input and remove it on exit.
  std::filesystem::path output_path;
  bool temporary_output = (argc < 3);
  if (temporary_output) {
    output_path = std::filesystem::path(input_path);
    output_path += ".onnxlight.tmp";
  } else {
    output_path = argv[2];
  }
  ScopedFile output_guard(temporary_output ? output_path : std::filesystem::path{});

  // Step 3: re-serialize the in-memory ModelProto with onnx-light. The
  // resulting file is byte-compatible with the standard ONNX protobuf
  // bindings (and therefore with Aidge's ONNX importer).
  try {
    const auto begin = std::chrono::steady_clock::now();
    olight::utils::FileWriteStream out_stream(output_path.string());
    olight::SerializeOptions ser_opts;
    olight::SerializeModelProtoToStream(model, out_stream, ser_opts);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "Re-serialized with onnx-light: " << output_path.string() << "\n";
    std::cout << "  Save time (ms)   : " << ToMilliseconds(end - begin) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error re-serializing model to '" << output_path.string()
              << "' with onnx-light: " << e.what() << "\n";
    return 1;
  }

#ifdef AIDGE_ONNX_LIGHT_HAS_AIDGE
  // Step 4: import the re-serialized file with Eclipse Aidge.
  try {
    const auto begin = std::chrono::steady_clock::now();
    std::shared_ptr<Aidge::GraphView> graph = Aidge::loadONNX(output_path.string());
    const auto end = std::chrono::steady_clock::now();
    if (!graph) {
      std::cerr << "Aidge::loadONNX returned a null GraphView for '" << output_path.string()
                << "'\n";
      return 1;
    }
    std::cout << "Imported with Aidge: " << output_path.string() << "\n";
    std::cout << "  Aidge load (ms)  : " << ToMilliseconds(end - begin) << "\n";
    std::cout << "  Aidge nodes      : " << graph->getNodes().size() << "\n";
    std::cout << "  Aidge inputs     : " << graph->inputNodes().size() << "\n";
    std::cout << "  Aidge outputs    : " << graph->outputNodes().size() << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error importing '" << output_path.string() << "' with Aidge: " << e.what()
              << "\n";
    return 1;
  }
#else
  std::cout << "Aidge integration disabled at build time "
               "(rebuild with the Eclipse Aidge CMake packages on CMAKE_PREFIX_PATH "
               "to enable it).\n";
#endif

  if (!temporary_output) {
    output_guard.release();
  }
  return 0;
}

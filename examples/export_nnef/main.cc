/**
 * main.cc -- Standalone example: export an ONNX model to the Khronos NNEF
 * v1.0 format using the onnx_light C++ API (lib_onnx_nnef).
 *
 * Usage:
 *   ./export_nnef <model.onnx> <out_dir> [graph_name]
 *
 * The executable loads the ONNX model with FileStream +
 * ParseModelProtoFromStream, converts it to a ``NNEFGraph`` via
 * ``nnef::ExportToNNEF``, prints the ``graph.nnef`` text to stdout and
 * writes the NNEF directory (``graph.nnef`` plus one ``<label>.dat`` file
 * per initializer) using ``nnef::SaveNNEF``.
 *
 * See CMakeLists.txt for build instructions.
 */

#include "nnef/exporter.h"
#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace onnx_light = ONNX_LIGHT_NAMESPACE;

void PrintModelSummary(const onnx_light::ModelProto &model) {
  if (model.has_ir_version()) {
    std::cout << "  IR version       : " << model.ref_ir_version() << "\n";
  }
  if (model.has_producer_name()) {
    std::cout << "  Producer name    : " << model.ref_producer_name().as_string() << "\n";
  }
  if (model.has_graph()) {
    const onnx_light::GraphProto &graph = model.ref_graph();
    std::cout << "  Graph name       : " << graph.ref_name().as_string() << "\n";
    std::cout << "  Nodes            : " << graph.ref_node().size() << "\n";
    std::cout << "  Inputs           : " << graph.ref_input().size() << "\n";
    std::cout << "  Outputs          : " << graph.ref_output().size() << "\n";
    std::cout << "  Initializers     : " << graph.ref_initializer().size() << "\n";
  }
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> <out_dir> [graph_name]\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string out_dir = argv[2];
  const std::string graph_name = (argc == 4) ? argv[3] : "";

  // 1. Load the ONNX model from disk.
  onnx_light::ModelProto model;
  try {
    onnx_light::utils::FileStream stream(model_path);
    onnx_light::ParseOptions parse_opts;
    onnx_light::ParseModelProtoFromStream(model, stream, parse_opts);
  } catch (const std::exception &e) {
    std::cerr << "Error loading '" << model_path << "': " << e.what() << "\n";
    return 1;
  }
  std::cout << "Loaded: " << model_path << "\n";
  PrintModelSummary(model);

  // 2. Convert to a NNEF graph and print its text representation.
  std::string nnef_text;
  try {
    nnef_text = onnx_light::nnef::ToNNEFText(model, graph_name);
  } catch (const onnx_light::nnef::NNEFExportError &e) {
    std::cerr << "NNEF export error for '" << model_path << "': " << e.what() << "\n";
    return 2;
  } catch (const std::exception &e) {
    std::cerr << "Error converting '" << model_path << "' to NNEF: " << e.what() << "\n";
    return 1;
  }

  std::cout << "\n--- graph.nnef ---\n" << nnef_text;
  if (!nnef_text.empty() && nnef_text.back() != '\n') {
    std::cout << "\n";
  }
  std::cout << "------------------\n";

  // 3. Write the NNEF directory (graph.nnef + one <label>.dat per initializer).
  std::string absolute_out_dir;
  try {
    absolute_out_dir = onnx_light::nnef::SaveNNEF(model, out_dir, graph_name, /*overwrite=*/true);
  } catch (const onnx_light::nnef::NNEFExportError &e) {
    std::cerr << "NNEF export error while writing '" << out_dir << "': " << e.what() << "\n";
    return 2;
  } catch (const std::exception &e) {
    std::cerr << "Error writing '" << out_dir << "': " << e.what() << "\n";
    return 1;
  }

  std::cout << "Wrote NNEF directory: " << absolute_out_dir << "\n";

  // 4. List the files produced for confirmation.
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(absolute_out_dir, ec)) {
    if (ec) {
      break;
    }
    std::cout << "  " << entry.path().filename().string() << "\n";
  }

  return 0;
}

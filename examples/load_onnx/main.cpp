/**
 * load_onnx.cpp – Standalone example: load an ONNX file with the onnx_light
 * C++ API and print a summary of the model to stdout.
 *
 * Usage:
 *   ./load_onnx <model.onnx>
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
    return 1;
  }

  const std::string file_path = argv[1];

  onnx::ModelProto model;
  {
    onnx::utils::FileStream stream(file_path);
    onnx::ParseOptions opts;
    onnx::ParseModelProtoFromStream(model, stream, opts);
  }

  std::cout << "Loaded: " << file_path << "\n";

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
    const onnx::GraphProto &graph = model.ref_graph();
    std::cout << "  Graph name       : " << graph.ref_name().as_string() << "\n";
    std::cout << "  Nodes            : " << graph.ref_node().size() << "\n";
    std::cout << "  Inputs           : " << graph.ref_input().size() << "\n";
    std::cout << "  Outputs          : " << graph.ref_output().size() << "\n";
    std::cout << "  Initializers     : " << graph.ref_initializer().size() << "\n";
  }

  return 0;
}

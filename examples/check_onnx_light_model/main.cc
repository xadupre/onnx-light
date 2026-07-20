/**
 * main.cc — Standalone example: validate an ONNX model with the onnx_light
 * checker API and optionally run onnx_shapes shape inference on it.
 *
 * Usage:
 *   ./check_onnx_light_model <model.onnx> [full_check] [infer_shapes]
 *
 * When ``infer_shapes`` is set to 1, the model is loaded into a ``ModelProto``
 * and passed to :cpp:func:`core::shapes::InferShapesModel`, which
 * mutates the graph in place so that its outputs and value_info entries carry
 * the inferred shapes. The example then prints how many value_info entries
 * the inferred graph contains.
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/common/file_utils.h"
#include "onnx_proto/onnx.h"
#include "onnx_shapes/dispatch_table.h"

#include <charconv>
#include <iostream>
#include <string_view>

namespace {

bool ParseZeroOrOne(const char *text, bool &value) {
  const std::string_view arg(text);
  if (arg.empty()) {
    return false;
  }

  int parsed = 0;
  const char *begin = arg.data();
  const char *end = begin + arg.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end || (parsed != 0 && parsed != 1)) {
    return false;
  }

  value = (parsed == 1);
  return true;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [full_check] [infer_shapes]\n";
    std::cerr << "  full_check:   0 (default) or 1\n";
    std::cerr << "  infer_shapes: 0 (default) or 1 — runs onnx_shapes shape inference\n";
    return 1;
  }

  bool full_check = false;
  if (argc >= 3 && !ParseZeroOrOne(argv[2], full_check)) {
    std::cerr << "Invalid full_check value: " << argv[2] << " (expected 0 or 1)\n";
    return 1;
  }

  bool infer_shapes = false;
  if (argc == 4 && !ParseZeroOrOne(argv[3], infer_shapes)) {
    std::cerr << "Invalid infer_shapes value: " << argv[3] << " (expected 0 or 1)\n";
    return 1;
  }

  try {
    ONNX_LIGHT_NAMESPACE::checker::check_model(argv[1], full_check);
    std::cout << "Model is valid: " << argv[1] << "\n";
    std::cout << "  full_check: " << (full_check ? "true" : "false") << "\n";

    if (infer_shapes) {
      ONNX_LIGHT_NAMESPACE::ModelProto model;
      ONNX_LIGHT_NAMESPACE::LoadProtoFromPath(argv[1], model);
      // `core::shapes::InferShapesModel` looks up shape functions in
      // `core::shapes::DispatchTable()`, which `onnx_core` leaves empty; the
      // built-in `onnx_shapes` shape functions must be registered explicitly
      // before it can resolve any operator.
      ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterShapeFunctions();
      ONNX_LIGHT_NAMESPACE::core::shapes::InferShapesModel(model);
      std::cout << "  shape inference: ok\n";
      std::cout << "    graph.value_info entries: " << model.graph().value_info_size() << "\n";
      std::cout << "    graph.output entries:     " << model.graph().output_size() << "\n";
    }
  } catch (const ONNX_LIGHT_NAMESPACE::checker::ValidationError &e) {
    std::cerr << "Validation error in '" << argv[1] << "':\n" << e.what() << "\n";
    return 2;
  } catch (const std::exception &e) {
    std::cerr << "Error while processing '" << argv[1] << "': " << e.what() << "\n";
    return 1;
  }

  return 0;
}

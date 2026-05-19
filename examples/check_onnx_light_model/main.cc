/**
 * main.cc — Standalone example: validate an ONNX model with the onnx_light
 * checker API.
 *
 * Usage:
 *   ./check_onnx_light_model <model.onnx> [full_check]
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx_lib/checker.h"

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
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [full_check]\n";
    std::cerr << "  full_check: 0 (default) or 1\n";
    return 1;
  }

  bool full_check = false;
  if (argc == 3 && !ParseZeroOrOne(argv[2], full_check)) {
    std::cerr << "Invalid full_check value: " << argv[2] << " (expected 0 or 1)\n";
    return 1;
  }

  try {
    ONNX_LIGHT_NAMESPACE::checker::check_model(argv[1], full_check);
    std::cout << "Model is valid: " << argv[1] << "\n";
    std::cout << "  full_check: " << (full_check ? "true" : "false") << "\n";
  } catch (const ONNX_LIGHT_NAMESPACE::checker::ValidationError &e) {
    std::cerr << "Validation error in '" << argv[1] << "':\n" << e.what() << "\n";
    return 2;
  } catch (const std::exception &e) {
    std::cerr << "Error while checking '" << argv[1] << "': " << e.what() << "\n";
    return 1;
  }

  return 0;
}

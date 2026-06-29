/**
 * main.cc -- Standalone example: print an ONNX proto as readable text for
 * debugging using only the onnx_light proto layer (lib_onnx_proto).
 *
 * Usage:
 *   ./print_proto_debug
 *
 * The executable creates a tiny NodeProto, formats it with
 * PrintToStringStream (writing to a stringstream), and writes the debug text to
 * stdout.
 */

#include "onnx.h"
#include "simple_string.h"

#include <iostream>
#include <sstream>

namespace onnx_light = ONNX_LIGHT_NAMESPACE;

int main() {
  onnx_light::NodeProto node;
  node.set_name("relu1");
  node.set_op_type("Relu");
  *node.add_input() = "X";
  *node.add_output() = "Y";
  node.set_doc_string("Simple ReLU activation");

  onnx_light::utils::PrintOptions options;
  std::stringstream ss;
  node.PrintToStringStream(ss, options);
  std::cout << ss.str() << "\n";
  return 0;
}

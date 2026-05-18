.. _l-cpp-print-proto-debug-example:

Standalone C++ example: print a proto for debugging
===================================================

When debugging model parsing or graph transformations, it can be helpful to
dump any ONNX proto as readable text. Each generated proto class exposes
``PrintToVectorString`` for this purpose.

.. code-block:: cpp

    #include "onnx.h"
    #include "simple_string.h"

    #include <iostream>
    #include <vector>

    int main() {
      onnx::NodeProto node;
      node.set_name("relu1");
      node.set_op_type("Relu");
      *node.add_input() = "X";
      *node.add_output() = "Y";
      node.set_doc_string("Simple ReLU activation");

      onnx::utils::PrintOptions options;
      std::vector<std::string> lines = node.PrintToVectorString(options);
      // join_string is declared in simple_string.h.
      std::cout << onnx::utils::join_string(lines, "\n") << "\n";
      return 0;
    }

Example output:

.. code-block:: text

    input: "X"
    output: "Y"
    name: "relu1"
    op_type: "Relu"
    doc_string: "Simple ReLU activation"

See also
--------

* ``onnx::ProtoDebugString`` (from ``proto_utils.h``) is a convenience helper
  that internally calls ``PrintToVectorString`` and returns a single
  ``std::string``.

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/kernels/text/include_text_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// StringConcat — z[i] = x[i] + y[i] element-wise with NumPy-style
// broadcasting (since opset 20 in the ai.onnx domain). Inputs and outputs
// are ``tensor(string)``; the reference kernel supports equal-shape and
// scalar broadcasting.
// ---------------------------------------------------------------------------
void RegisterStringConcatCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::StringConcat string_concat{kernel::KernelContext(opset)};

  // Equal-shape variant: element-wise concatenation of two 1-D string
  // tensors.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {3}, {"abc", "", "hello "});
    Tensor y = Tensor::FromStrings("", {3}, {"def", "xyz", "world"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] + y (scalar).
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {2, 2}, {"a", "b", "c", "d"});
    Tensor y = Tensor::FromStrings("", {}, {"!"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat_bcast", {opset}, "backend-test", registry);
  }

  // Zero-dimensional (scalar) variant: both inputs are 0-D string tensors.
  // Mirrors upstream onnx ``test_string_concat_zero_dimensional``.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {}, {"cat"});
    Tensor y = Tensor::FromStrings("", {}, {"s"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat_zero_dimensional", {opset}, "backend-test",
           registry);
  }

  // UTF-8 variant: element-wise concatenation of multi-byte UTF-8 strings.
  // Mirrors upstream onnx ``test_string_concat_utf8``.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {2}, {"\xe7\x9a\x84", "\xe4\xb8\xad"});
    Tensor y = Tensor::FromStrings("", {2}, {"\xe7\x9a\x84", "\xe4\xb8\xad"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat_utf8", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

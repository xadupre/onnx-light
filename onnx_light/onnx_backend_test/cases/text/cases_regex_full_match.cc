// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/text/include_text_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// RegexFullMatch — element-wise full regex match of a ``tensor(string)``
// input against the ``pattern`` attribute, producing a ``tensor(bool)``
// output with the same shape (since opset 20 in the ai.onnx domain).
// ---------------------------------------------------------------------------
void RegisterRegexFullMatchCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::KernelContext ctx{opset};
  const kernel::RegexFullMatch regex_full_match{ctx};

  // Basic variant: simple word pattern on a 1-D ``[C]`` input. Mirrors
  // upstream onnx ``test_regex_full_match_basic``.
  {
    NodeProto node;
    node.set_op_type("RegexFullMatch");
    node.add_input("x");
    node.add_output("y");
    const std::string pattern = "www\\.[\\w.-]+\\.\\bcom\\b";
    AddAttribute(node, "pattern", pattern);

    Tensor x =
        Tensor::FromStrings("", {3}, {"www.google.com", "www.facebook.com", "www.bbc.co.uk"});
    Tensor y = regex_full_match(x, pattern);

    Expect(node, {x}, {y}, "test_cc_regex_full_match_basic", {opset}, "backend-test", registry);
  }

  // Email-like pattern on a 2-D ``[2, 2]`` input exercising True/False
  // results in a non-flat shape.
  {
    NodeProto node;
    node.set_op_type("RegexFullMatch");
    node.add_input("x");
    node.add_output("y");
    const std::string pattern = "[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\\.[A-Za-z0-9-.]+";
    AddAttribute(node, "pattern", pattern);

    Tensor x =
        Tensor::FromStrings("", {2, 2}, {"account@gmail.com", "not_an_email", "x@y.z", "@nope"});
    Tensor y = regex_full_match(x, pattern);

    Expect(node, {x}, {y}, "test_cc_regex_full_match_email", {opset}, "backend-test", registry);
  }

  // Empty input tensor — the kernel must still produce a correctly
  // shaped empty BOOL output.
  {
    NodeProto node;
    node.set_op_type("RegexFullMatch");
    node.add_input("x");
    node.add_output("y");
    const std::string pattern = "abc";
    AddAttribute(node, "pattern", pattern);

    Tensor x = Tensor::FromStrings("", {0}, std::vector<std::string>{});
    Tensor y = regex_full_match(x, pattern);

    Expect(node, {x}, {y}, "test_cc_regex_full_match_empty", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

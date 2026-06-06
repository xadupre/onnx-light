// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/text/include_text_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterStringSplitCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::KernelContext ctx{opset};
  const kernel::StringSplit string_split{ctx};

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    AddAttribute(node, "delimiter", std::string("."));

    Tensor x = Tensor::FromStrings("", {2}, {"abc.com", "def.net"});
    auto [substrings, length] = string_split(x, ".");

    Expect(node, {x}, {std::move(substrings), std::move(length)}, "test_cc_string_split_basic",
           {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    AddAttribute(node, "maxsplit", static_cast<int64_t>(2));

    Tensor x = Tensor::FromStrings("", {2, 2},
                                   {"hello world", "def.net", "o n n x", "the quick brown fox"});
    auto [substrings, length] = string_split(x, "", 2);

    Expect(node, {x}, {std::move(substrings), std::move(length)}, "test_cc_string_split_maxsplit",
           {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    AddAttribute(node, "delimiter", std::string("-"));

    Tensor x = Tensor::FromStrings("", {2}, {"o-n-n--x-", "o-n----nx"});
    auto [substrings, length] = string_split(x, "-");

    Expect(node, {x}, {std::move(substrings), std::move(length)},
           "test_cc_string_split_consecutive_delimiters", {opset}, "backend-test", registry);
  }

  // Keep both variants: ONNX upstream exercises both an explicit empty-string
  // delimiter attribute and the absence of the delimiter attribute, and both
  // must resolve to whitespace splitting.
  for (const auto &[set_delimiter_attr, test_name] : std::vector<std::pair<bool, std::string>>{
           {true, "test_cc_string_split_empty_string_delimiter"},
           {false, "test_cc_string_split_no_delimiter"},
       }) {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    if (set_delimiter_attr) {
      AddAttribute(node, "delimiter", std::string());
    }

    Tensor x =
        Tensor::FromStrings("", {3}, {"hello world !", "  hello   world !", " hello world   ! "});
    auto [substrings, length] = string_split(x);

    Expect(node, {x}, {std::move(substrings), std::move(length)}, test_name, {opset},
           "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");

    Tensor x = Tensor::FromStrings("", {0}, std::vector<std::string>{});
    auto [substrings, length] = string_split(x);

    Expect(node, {x}, {std::move(substrings), std::move(length)},
           "test_cc_string_split_empty_tensor", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

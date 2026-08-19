// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/text/include_text_cases.h"
#include "onnx_extensions/kernels/kernels/text/include_text_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterStringSplitCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(20);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::StringSplit string_split{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    AddAttribute(node, "delimiter", std::string("."));

    constexpr int64_t count = 131072;
    constexpr int64_t substring_count = count * 2;
    Expect(registry, std::move(node), "test_cc_string_split_basic_benchmark", {opset}, {count},
           {substring_count, count}, [string_split]() -> IoData {
             std::vector<std::string> values(static_cast<size_t>(count));
             for (size_t i = 0; i < values.size(); ++i) {
               values[i] = (i % 2 == 0) ? "abc.com" : "def.net";
             }
             Tensor x = Tensor::FromStrings("", {count}, values);
             auto [substrings, length] = string_split(x, ".");
             return IoData{{std::move(x)}, {std::move(substrings), std::move(length)}};
           });
    return;
  }

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    AddAttribute(node, "delimiter", std::string("."));
    Expect(registry, std::move(node), "test_cc_string_split_basic", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromStrings("", {2}, {"abc.com", "def.net"});
      auto [substrings, length] = string_split(x, ".");

      return IoData{{std::move(x)}, {std::move(substrings), std::move(length)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    AddAttribute(node, "maxsplit", static_cast<int64_t>(2));
    Expect(registry, std::move(node), "test_cc_string_split_maxsplit", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromStrings("", {2, 2},
                                     {"hello world", "def.net", "o n n x", "the quick brown fox"});
      auto [substrings, length] = string_split(x, "", 2);

      return IoData{{std::move(x)}, {std::move(substrings), std::move(length)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    AddAttribute(node, "delimiter", std::string("-"));
    Expect(registry, std::move(node), "test_cc_string_split_consecutive_delimiters", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromStrings("", {2}, {"o-n-n--x-", "o-n----nx"});
             auto [substrings, length] = string_split(x, "-");

             return IoData{{std::move(x)}, {std::move(substrings), std::move(length)}};
           });
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

    Expect(registry, std::move(node), test_name, {opset}, [string_split]() -> IoData {
      Tensor x =
          Tensor::FromStrings("", {3}, {"hello world !", "  hello   world !", " hello world   ! "});
      auto [substrings, length] = string_split(x);
      return IoData{{std::move(x)}, {std::move(substrings), std::move(length)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("StringSplit");
    node.add_input("x");
    node.add_output("substrings");
    node.add_output("length");
    Expect(registry, std::move(node), "test_cc_string_split_empty_tensor", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromStrings("", {0}, std::vector<std::string>{});
             auto [substrings, length] = string_split(x);

             return IoData{{std::move(x)}, {std::move(substrings), std::move(length)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

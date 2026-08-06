// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/text/include_text_cases.h"
#include "onnx_extensions/kernels/kernels/text/include_text_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// RegexFullMatch — element-wise full regex match of a ``tensor(string)``
// input against the ``pattern`` attribute, producing a ``tensor(bool)``
// output with the same shape (since opset 20 in the ai.onnx domain).
// ---------------------------------------------------------------------------
void RegisterRegexFullMatchCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(20);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::RegexFullMatch regex_full_match{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("RegexFullMatch");
    node.add_input("x");
    node.add_output("y");
    const std::string pattern = "www\\.[\\w.-]+\\.\\bcom\\b";
    AddAttribute(node, "pattern", pattern);

    constexpr int64_t count = 262144;
    Expect(registry, std::move(node), "test_cc_regex_full_match_basic_benchmark", {opset}, {count},
           {count}, [regex_full_match, pattern, count]() -> IoData {
             std::vector<std::string> values(static_cast<size_t>(count));
             for (size_t i = 0; i < values.size(); ++i) {
               values[i] = (i % 3 == 0) ? "www.google.com"
                                        : ((i % 3 == 1) ? "www.facebook.com" : "www.bbc.co.uk");
             }
             Tensor x = Tensor::FromStrings("", {count}, values);
             Tensor y = regex_full_match(x, pattern);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Basic variant: simple word pattern on a 1-D ``[C]`` input. Mirrors
  // upstream onnx ``test_regex_full_match_basic``.
  {
    NodeProto node;
    node.set_op_type("RegexFullMatch");
    node.add_input("x");
    node.add_output("y");
    const std::string pattern = "www\\.[\\w.-]+\\.\\bcom\\b";
    AddAttribute(node, "pattern", pattern);
    Expect(registry, std::move(node), "test_cc_regex_full_match_basic", {opset}, [=]() -> IoData {
      Tensor x =
          Tensor::FromStrings("", {3}, {"www.google.com", "www.facebook.com", "www.bbc.co.uk"});
      Tensor y = regex_full_match(x, pattern);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_regex_full_match_email", {opset}, [=]() -> IoData {
      Tensor x =
          Tensor::FromStrings("", {2, 2}, {"account@gmail.com", "not_an_email", "x@y.z", "@nope"});
      Tensor y = regex_full_match(x, pattern);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Email-domain pattern mirroring upstream onnx
  // ``test_regex_full_match_email_domain``: anchored alternation of
  // ``yahoo``/``gmail`` ``.com`` domains on a 2-D ``[2, 2]`` input.
  {
    NodeProto node;
    node.set_op_type("RegexFullMatch");
    node.add_input("x");
    node.add_output("y");
    const std::string pattern = "(\\W|^)[\\w.\\-]{0,25}@(yahoo|gmail)\\.com(\\W|$)";
    AddAttribute(node, "pattern", pattern);
    Expect(registry, std::move(node), "test_cc_regex_full_match_email_domain", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromStrings(
                 "", {2, 2},
                 {"account@gmail.com", "account@hotmail.com", "not email", "account2@yahoo.com"});
             Tensor y = regex_full_match(x, pattern);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_regex_full_match_empty", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromStrings("", {0}, std::vector<std::string>{});
      Tensor y = regex_full_match(x, pattern);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

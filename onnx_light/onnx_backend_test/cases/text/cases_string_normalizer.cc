// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/kernels/text/include_text_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// StringNormalizer — removes elements matching the ``stopwords`` attribute
// from a ``[C]`` or ``[1, C]`` ``tensor(string)`` and applies the requested
// ``case_change_action`` (since opset 10 in the ai.onnx domain).
// ---------------------------------------------------------------------------
void RegisterStringNormalizerCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::StringNormalizer string_normalizer{ctx};

  using CaseChangeAction = kernel::StringNormalizer::CaseChangeAction;

  // Plain lowercase variant on a 1-D ``[C]`` input. No stopwords are
  // dropped; every element is lowercased in place.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("LOWER"));

    Tensor x = Tensor::FromStrings("", {3}, {"Hello", "World", "FOO"});
    Tensor y = string_normalizer(x, CaseChangeAction::kLower, /*is_case_sensitive=*/false, {});

    Expect(node, {x}, {y}, "test_cc_string_normalizer_lower", {opset}, "backend-test", registry);
  }

  // Stopwords + uppercasing on a 2-D ``[1, C]`` input. ``"a"`` is a
  // case-insensitive stopword and is therefore dropped.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("UPPER"));
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(0));
    AddAttribute(node, "stopwords", std::vector<std::string>{"a"});

    Tensor x = Tensor::FromStrings("", {1, 4}, {"A", "hello", "a", "world"});
    Tensor y = string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/false, {"a"});

    Expect(node, {x}, {y}, "test_cc_string_normalizer_upper", {opset}, "backend-test", registry);
  }

  // Case-sensitive stopwords variant: only the exact-case "the" is
  // dropped; "The" is kept.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));
    AddAttribute(node, "stopwords", std::vector<std::string>{"the"});

    Tensor x = Tensor::FromStrings("", {3}, {"The", "the", "cat"});
    Tensor y = string_normalizer(x, CaseChangeAction::kNone,
                                 /*is_case_sensitive=*/true, {"the"});

    Expect(node, {x}, {y}, "test_cc_string_normalizer_case_sensitive", {opset}, "backend-test",
           registry);
  }

  // All elements dropped → output is a single empty string with shape
  // matching the input rank (``[1]`` for ``[C]`` / ``[1, 1]`` for ``[1, C]``).
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "stopwords", std::vector<std::string>{"a", "b"});

    Tensor x = Tensor::FromStrings("", {2}, {"a", "b"});
    Tensor y =
        string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/false, {"a", "b"});

    Expect(node, {x}, {y}, "test_cc_string_normalizer_all_dropped", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

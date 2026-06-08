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

  // NOOP variant: no stopwords, no case change, case-sensitive comparison.
  // Mirrors upstream ``test_strnormalizer_nostopwords_nochangecase``.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));

    Tensor x = Tensor::FromStrings("", {2}, {"monday", "tuesday"});
    Tensor y = string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/true, {});

    Expect(node, {x}, {y}, "test_cc_string_normalizer_nostopwords_nochangecase", {opset},
           "backend-test", registry);
  }

  // Case-sensitive stopword drop combined with ``LOWER`` case change on a 1-D
  // ``[C]`` input. Mirrors upstream
  // ``test_strnormalizer_export_monday_casesensintive_lower``.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("LOWER"));
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));
    AddAttribute(node, "stopwords", std::vector<std::string>{"monday"});

    Tensor x = Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
    Tensor y =
        string_normalizer(x, CaseChangeAction::kLower, /*is_case_sensitive=*/true, {"monday"});

    Expect(node, {x}, {y}, "test_cc_string_normalizer_case_sensitive_lower", {opset},
           "backend-test", registry);
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

  // ---------------------------------------------------------------------------
  // Direct mirrors of the upstream ``test_strnormalizer_*`` node backend tests
  // (see ``onnx/backend/test/case/node/stringnormalizer.py``). The case names
  // embed the upstream stems so the substring-based name parity check in
  // ``unittests/onnxl_vs_onnx/test_backend_test_names_onnx_vs_onnxlight.py``
  // sees them as covered.
  // ---------------------------------------------------------------------------

  // Mirrors ``test_strnormalizer_nostopwords_nochangecase``: no stopwords
  // means the operator is a NOOP.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));

    Tensor x = Tensor::FromStrings("", {2}, {"monday", "tuesday"});
    Tensor y = string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/true, {});

    Expect(node, {x}, {y}, "test_cc_strnormalizer_nostopwords_nochangecase", {opset},
           "backend-test", registry);
  }

  // Mirrors ``test_strnormalizer_export_monday_casesensintive_nochangecase``:
  // case-sensitive stopword drop without case folding.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));
    AddAttribute(node, "stopwords", std::vector<std::string>{"monday"});

    Tensor x = Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
    Tensor y =
        string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/true, {"monday"});

    Expect(node, {x}, {y}, "test_cc_strnormalizer_export_monday_casesensintive_nochangecase",
           {opset}, "backend-test", registry);
  }

  // Mirrors ``test_strnormalizer_export_monday_casesensintive_lower``:
  // case-sensitive stopword drop combined with ``LOWER``.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("LOWER"));
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));
    AddAttribute(node, "stopwords", std::vector<std::string>{"monday"});

    Tensor x = Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
    Tensor y =
        string_normalizer(x, CaseChangeAction::kLower, /*is_case_sensitive=*/true, {"monday"});

    Expect(node, {x}, {y}, "test_cc_strnormalizer_export_monday_casesensintive_lower", {opset},
           "backend-test", registry);
  }

  // Mirrors ``test_strnormalizer_export_monday_casesensintive_upper``:
  // case-sensitive stopword drop combined with ``UPPER``.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("UPPER"));
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));
    AddAttribute(node, "stopwords", std::vector<std::string>{"monday"});

    Tensor x = Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
    Tensor y =
        string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/true, {"monday"});

    Expect(node, {x}, {y}, "test_cc_strnormalizer_export_monday_casesensintive_upper", {opset},
           "backend-test", registry);
  }

  // Mirrors ``test_strnormalizer_export_monday_empty_output``: every element
  // is dropped → output is a single empty string of shape ``[1]``.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("UPPER"));
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));
    AddAttribute(node, "stopwords", std::vector<std::string>{"monday"});

    Tensor x = Tensor::FromStrings("", {2}, {"monday", "monday"});
    Tensor y =
        string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/true, {"monday"});

    Expect(node, {x}, {y}, "test_cc_strnormalizer_export_monday_empty_output", {opset},
           "backend-test", registry);
  }

  // Mirrors ``test_strnormalizer_export_monday_insensintive_upper_twodim``:
  // case-insensitive stopword drop on a ``[1, 6]`` input + ``UPPER``.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("UPPER"));
    AddAttribute(node, "stopwords", std::vector<std::string>{"monday"});

    Tensor x = Tensor::FromStrings(
        "", {1, 6}, {"Monday", "tuesday", "wednesday", "Monday", "tuesday", "wednesday"});
    Tensor y =
        string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/false, {"monday"});

    Expect(node, {x}, {y}, "test_cc_strnormalizer_export_monday_insensintive_upper_twodim", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

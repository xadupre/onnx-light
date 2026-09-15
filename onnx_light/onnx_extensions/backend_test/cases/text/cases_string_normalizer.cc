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
// StringNormalizer — removes elements matching the ``stopwords`` attribute
// from a ``[C]`` or ``[1, C]`` ``tensor(string)`` and applies the requested
// ``case_change_action`` (since opset 10 in the ai.onnx domain).
// ---------------------------------------------------------------------------
void RegisterStringNormalizerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(10);

  using CaseChangeAction = onnx_kernels::kernel::StringNormalizer::CaseChangeAction;

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("LOWER"));

    constexpr int64_t count = 262144;
    Expect(registry, std::move(node), "test_cc_string_normalizer_lower_benchmark", {opset}, {count},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             std::vector<std::string> values(static_cast<size_t>(count));
             for (size_t i = 0; i < values.size(); ++i) {
               values[i] = (i % 3 == 0) ? "Hello" : ((i % 3 == 1) ? "World" : "FOO");
             }
             Tensor x = Tensor::FromStrings("", {count}, values);
             Tensor y = string_normalizer(x, CaseChangeAction::kLower,
                                          /*is_case_sensitive=*/false, {});
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Plain lowercase variant on a 1-D ``[C]`` input. No stopwords are
  // dropped; every element is lowercased in place.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "case_change_action", std::string("LOWER"));
    Expect(registry, std::move(node), "test_cc_string_normalizer_lower", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(10);

      const KernelContext string_normalizer_ctx{opset};
      const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

      Tensor x = Tensor::FromStrings("", {3}, {"Hello", "World", "FOO"});
      Tensor y = string_normalizer(x, CaseChangeAction::kLower, /*is_case_sensitive=*/false, {});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // NOOP variant: no stopwords, no case change, case-sensitive comparison.
  // Mirrors upstream ``test_strnormalizer_nostopwords_nochangecase``.
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "is_case_sensitive", static_cast<int64_t>(1));
    Expect(registry, std::move(node), "test_cc_string_normalizer_nostopwords_nochangecase", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x = Tensor::FromStrings("", {2}, {"monday", "tuesday"});
             Tensor y =
                 string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/true, {});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_string_normalizer_case_sensitive_lower", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x =
                 Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
             Tensor y = string_normalizer(x, CaseChangeAction::kLower, /*is_case_sensitive=*/true,
                                          {"monday"});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_string_normalizer_upper", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(10);

      const KernelContext string_normalizer_ctx{opset};
      const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

      Tensor x = Tensor::FromStrings("", {1, 4}, {"A", "hello", "a", "world"});
      Tensor y = string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/false, {"a"});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_string_normalizer_case_sensitive", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x = Tensor::FromStrings("", {3}, {"The", "the", "cat"});
             Tensor y = string_normalizer(x, CaseChangeAction::kNone,
                                          /*is_case_sensitive=*/true, {"the"});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // All elements dropped → output is a single empty string with shape
  // matching the input rank (``[1]`` for ``[C]`` / ``[1, 1]`` for ``[1, C]``).
  {
    NodeProto node;
    node.set_op_type("StringNormalizer");
    node.add_input("x");
    node.add_output("y");
    AddAttribute(node, "stopwords", std::vector<std::string>{"a", "b"});
    Expect(registry, std::move(node), "test_cc_string_normalizer_all_dropped", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x = Tensor::FromStrings("", {2}, {"a", "b"});
             Tensor y = string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/false,
                                          {"a", "b"});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_strnormalizer_nostopwords_nochangecase", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x = Tensor::FromStrings("", {2}, {"monday", "tuesday"});
             Tensor y =
                 string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/true, {});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(
        registry, std::move(node),
        "test_cc_strnormalizer_export_monday_casesensintive_nochangecase", {opset}, []() -> IoData {
          const OpsetId opset = DefaultOpset(10);

          const KernelContext string_normalizer_ctx{opset};
          const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

          Tensor x = Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
          Tensor y =
              string_normalizer(x, CaseChangeAction::kNone, /*is_case_sensitive=*/true, {"monday"});

          return IoData{{std::move(x)}, {std::move(y)}};
        });
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
    Expect(registry, std::move(node), "test_cc_strnormalizer_export_monday_casesensintive_lower",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x =
                 Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
             Tensor y = string_normalizer(x, CaseChangeAction::kLower, /*is_case_sensitive=*/true,
                                          {"monday"});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_strnormalizer_export_monday_casesensintive_upper",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x =
                 Tensor::FromStrings("", {4}, {"monday", "tuesday", "wednesday", "thursday"});
             Tensor y = string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/true,
                                          {"monday"});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_strnormalizer_export_monday_empty_output", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x = Tensor::FromStrings("", {2}, {"monday", "monday"});
             Tensor y = string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/true,
                                          {"monday"});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node),
           "test_cc_strnormalizer_export_monday_insensintive_upper_twodim", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(10);

             const KernelContext string_normalizer_ctx{opset};
             const onnx_kernels::kernel::StringNormalizer string_normalizer{string_normalizer_ctx};

             Tensor x = Tensor::FromStrings(
                 "", {1, 6}, {"Monday", "tuesday", "wednesday", "Monday", "tuesday", "wednesday"});
             Tensor y = string_normalizer(x, CaseChangeAction::kUpper, /*is_case_sensitive=*/false,
                                          {"monday"});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

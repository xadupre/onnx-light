// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/text/include_text_kernels.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// ASCII-only lowercase, matching the "C" locale semantics used by the
// upstream onnx reference implementation. Multi-byte UTF-8 sequences are
// passed through untouched (their leading bytes are >= 0x80 and therefore
// are not affected by ``std::tolower``).
std::string AsciiToLower(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

std::string AsciiToUpper(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    out.push_back(static_cast<char>(std::toupper(c)));
  }
  return out;
}

// Returns the ``[C]`` slice of a ``[C]`` or ``[1, C]`` input shape.
// Throws std::invalid_argument when the shape is neither.
int64_t ExtractC(const std::vector<int64_t> &shape) {
  if (shape.size() == 1) {
    return shape[0];
  }
  if (shape.size() == 2 && shape[0] == 1) {
    return shape[1];
  }
  throw std::invalid_argument(
      "kernel::StringNormalizer only accepts [C] or [1, C] string tensors.");
}

} // namespace

StringNormalizer::CaseChangeAction
StringNormalizer::ParseCaseChangeAction(const std::string &value) {
  if (value == "NONE") {
    return CaseChangeAction::kNone;
  }
  if (value == "LOWER") {
    return CaseChangeAction::kLower;
  }
  if (value == "UPPER") {
    return CaseChangeAction::kUpper;
  }
  throw std::invalid_argument("kernel::StringNormalizer: invalid case_change_action '" + value +
                              "'. Valid values are \"NONE\", \"LOWER\", \"UPPER\".");
}

std::vector<int64_t> StringNormalizer::ComputeOutputShape(const std::vector<int64_t> &input_shape,
                                                          int64_t kept) {
  ExtractC(input_shape);
  const int64_t out_c = std::max<int64_t>(kept, 1);
  if (input_shape.size() == 1) {
    return {out_c};
  }
  return {1, out_c};
}

Tensor StringNormalizer::operator()(const Tensor &x, CaseChangeAction case_change_action,
                                    bool is_case_sensitive,
                                    const std::vector<std::string> &stopwords) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::STRING),
                      "kernel::StringNormalizer only supports STRING tensors.");
  const int64_t c = ExtractC(x.shape);
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == c,
                      "kernel::StringNormalizer input string_data size does not match its shape.");

  // Build the stopword lookup set, lowercasing every entry when
  // ``is_case_sensitive`` is false to match the case-insensitive
  // comparison performed below.
  std::unordered_set<std::string> stopword_set;
  stopword_set.reserve(stopwords.size());
  for (const auto &word : stopwords) {
    stopword_set.insert(is_case_sensitive ? word : AsciiToLower(word));
  }

  // First pass: filter out stopwords.
  std::vector<std::string> kept;
  kept.reserve(static_cast<size_t>(c));
  for (int64_t i = 0; i < c; ++i) {
    const std::string &s = x.string_data[static_cast<size_t>(i)];
    const std::string &lookup = is_case_sensitive ? s : AsciiToLower(s);
    if (stopword_set.find(lookup) == stopword_set.end()) {
      kept.push_back(s);
    }
  }

  // Second pass: apply the case-change action.
  switch (case_change_action) {
  case CaseChangeAction::kLower:
    for (auto &s : kept) {
      s = AsciiToLower(s);
    }
    break;
  case CaseChangeAction::kUpper:
    for (auto &s : kept) {
      s = AsciiToUpper(s);
    }
    break;
  case CaseChangeAction::kNone:
    break;
  }

  std::vector<int64_t> out_shape = ComputeOutputShape(x.shape, static_cast<int64_t>(kept.size()));
  std::vector<std::string> out_data;
  if (kept.empty()) {
    // All elements dropped → produce a single empty string.
    out_data.emplace_back();
  } else {
    out_data = std::move(kept);
  }
  return Tensor::MakeString("", out_shape, std::move(out_data));
}

void StringNormalizer::operator()(const Tensor &x, CaseChangeAction case_change_action,
                                  bool is_case_sensitive, const std::vector<std::string> &stopwords,
                                  Tensor &output) const {
  Tensor computed = (*this)(x, case_change_action, is_case_sensitive, stopwords);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::STRING),
                      "kernel::StringNormalizer preallocated output must be a STRING tensor.");
  EXT_ENFORCE_INVALID(output.shape == computed.shape,
                      "kernel::StringNormalizer preallocated output shape must match the "
                      "computed output shape.");
  EXT_ENFORCE_INVALID(output.string_data.size() == computed.string_data.size(),
                      "kernel::StringNormalizer preallocated output string_data has unexpected "
                      "size.");
  output.string_data = std::move(computed.string_data);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

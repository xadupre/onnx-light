// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/text/include_text_kernels.h"

#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Compiles ``pattern`` with ECMAScript syntax. Throws
// ``std::invalid_argument`` if the pattern is not a valid
// ``std::regex`` (e.g. uses RE2-specific syntax not supported by the
// C++ standard library). The message preserves the original
// ``std::regex_error`` description so callers can diagnose the
// offending pattern.
std::regex CompileRegexPattern(const std::string &pattern) {
  try {
    return std::regex(pattern);
  } catch (const std::regex_error &e) {
    throw std::invalid_argument("kernel::RegexFullMatch: invalid regex pattern \"" + pattern +
                                "\": " + e.what());
  }
}

void CheckRegexFullMatchInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(x.data_type == DataType::STRING,
                      "kernel::RegexFullMatch only supports STRING tensors.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == x.element_count(),
                      "kernel::RegexFullMatch input ``x`` string_data size does not match its "
                      "shape.");
}

} // namespace

Tensor RegexFullMatch::operator()(const Tensor &x, const std::string &pattern) const {
  CheckRegexFullMatchInput(x);
  const int64_t n = x.element_count();
  Tensor out = Tensor::FromBool("", x.shape, std::vector<uint8_t>(static_cast<size_t>(n), 0));
  (*this)(x, pattern, out);
  return out;
}

void RegexFullMatch::operator()(const Tensor &x, const std::string &pattern, Tensor &output) const {
  CheckRegexFullMatchInput(x);
  EXT_ENFORCE_INVALID(output.data_type == DataType::BOOL,
                      "kernel::RegexFullMatch preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::RegexFullMatch preallocated output shape must match the input "
                      "shape.");
  const int64_t n = x.element_count();
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.data.size()) ==
                          n * static_cast<int64_t>(sizeof(uint8_t)),
                      "kernel::RegexFullMatch preallocated output data has unexpected size.");
  const std::regex re = CompileRegexPattern(pattern);
  uint8_t *const dst = output.AsBool();
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = std::regex_match(x.string_data[static_cast<size_t>(i)], re) ? uint8_t{1} : uint8_t{0};
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

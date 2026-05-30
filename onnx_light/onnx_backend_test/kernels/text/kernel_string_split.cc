// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/text/include_text_kernels.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

std::vector<std::string> SplitOnDelimiter(const std::string &text, const std::string &delimiter,
                                          int64_t maxsplit) {
  if (maxsplit == 0) {
    return {text};
  }
  std::vector<std::string> parts;
  std::size_t start = 0;
  int64_t splits = 0;
  while (maxsplit < 0 || splits < maxsplit) {
    const std::size_t pos = text.find(delimiter, start);
    if (pos == std::string::npos) {
      break;
    }
    parts.push_back(text.substr(start, pos - start));
    start = pos + delimiter.size();
    ++splits;
  }
  parts.push_back(text.substr(start));
  return parts;
}

std::vector<std::string> SplitOnWhitespace(const std::string &text, int64_t maxsplit) {
  std::vector<std::string> parts;
  const std::size_t n = text.size();
  std::size_t i = 0;
  int64_t splits = 0;
  while (true) {
    while (i < n && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
      ++i;
    }
    if (i >= n) {
      break;
    }
    if (maxsplit >= 0 && splits >= maxsplit) {
      parts.push_back(text.substr(i));
      break;
    }
    std::size_t j = i;
    while (j < n && std::isspace(static_cast<unsigned char>(text[j])) == 0) {
      ++j;
    }
    parts.push_back(text.substr(i, j - i));
    ++splits;
    i = j;
  }
  return parts;
}

std::vector<std::string> SplitString(const std::string &text, const std::string &delimiter,
                                     int64_t maxsplit) {
  if (delimiter.empty()) {
    return SplitOnWhitespace(text, maxsplit);
  }
  return SplitOnDelimiter(text, delimiter, maxsplit);
}

} // namespace

std::pair<Tensor, Tensor> StringSplit::operator()(const Tensor &x, const std::string &delimiter,
                                                  int64_t maxsplit) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(TensorProto::DataType::STRING),
                      "kernel::StringSplit only supports STRING tensors.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == x.element_count(),
                      "kernel::StringSplit input string_data size does not match its shape.");

  const std::vector<std::string> &input = x.AsStrings();
  std::vector<std::vector<std::string>> split_lists(static_cast<std::size_t>(x.element_count()));
  std::vector<int64_t> lengths(static_cast<std::size_t>(x.element_count()));
  int64_t max_length = 0;
  for (int64_t i = 0; i < x.element_count(); ++i) {
    split_lists[static_cast<std::size_t>(i)] =
        SplitString(input[static_cast<std::size_t>(i)], delimiter, maxsplit);
    const int64_t size = static_cast<int64_t>(split_lists[static_cast<std::size_t>(i)].size());
    lengths[static_cast<std::size_t>(i)] = size;
    max_length = std::max(max_length, size);
  }

  std::vector<int64_t> y_shape = x.shape;
  y_shape.push_back(max_length);
  std::vector<std::string> y_data;
  y_data.reserve(static_cast<std::size_t>(x.element_count() * max_length));
  for (const auto &parts : split_lists) {
    y_data.insert(y_data.end(), parts.begin(), parts.end());
    y_data.insert(y_data.end(), static_cast<std::size_t>(max_length) - parts.size(), "");
  }

  Tensor y = Tensor::MakeString("", std::move(y_shape), std::move(y_data));
  Tensor z = Tensor::FromInt64("", x.shape, lengths);
  return std::pair<Tensor, Tensor>(std::move(y), std::move(z));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
